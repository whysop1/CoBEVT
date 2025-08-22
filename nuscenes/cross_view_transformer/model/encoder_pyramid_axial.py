import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional
from .decoder import  DecoderBlock

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):
    def __init__(
            self,
            dim: int,
            sigma: int,
            bev_height: int,
            bev_width: int,
            h_meters: int,
            w_meters: int,
            offset: int,
            upsample_scales: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters,
                            offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()  # 3 3

        for i, scale in enumerate(upsample_scales):
            # each decoder block upsamples the bev embedding by a factor of 2
            h = bev_height // scale
            w = bev_width // scale

            # bev coordinates
            grid = generate_grid(h, w).squeeze(0)
            grid[0] = bev_width * grid[0]
            grid[1] = bev_height * grid[1]

            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')  # 3 (h w)
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)  # 3 h w
            # egocentric frame
            self.register_buffer('grid%d'%i, grid, persistent=False)

            # 3 h w
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim,
                                bev_height//upsample_scales[0],
                                bev_width//upsample_scales[0]))  # d h w

    def get_prior(self):
        return self.learned_features


# ============= Nyström-based Cross-Window Attention =============
def rank_from_count(count: int,
                    r_min: int = 16,
                    r_max: int = 128,
                    k: float = 6.0,
                    c0: float = 16.0) -> int:
    """
    Monotone rank scheduler: r = clamp(r_min + k * log2(1 + count/c0), r_min, r_max)
    """
    import math
    r = int(r_min + k * math.log2(1.0 + float(count)/float(c0)))
    return max(r_min, min(r, r_max))


class CrossWinAttentionNystrom(nn.Module):
    """
    Cross-window attention with Nyström approximation of softmax(QK^T)V.

    Shapes follow the original CrossWinAttention:
      q: (b n X Y W1 W2 d)
      k: (b n x y w1 w2 d)
      v: (b n x y w1 w2 d)
    Internally we flatten per (X*Y) windows and group heads into batch.

    Rank m (landmarks) is adaptive and can be updated via set_rank().
    """
    def __init__(self, dim, heads, dim_head, qkv_bias,
                 rel_pos_emb: bool = False,
                 norm=nn.LayerNorm,
                 base_rank: int = 32):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.rel_pos_emb = rel_pos_emb

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)

        self._rank = base_rank
        self.eps = 1e-6

    @torch.no_grad()
    def set_rank(self, rank: int):
        self._rank = int(max(1, rank))

    def add_rel_pos_emb(self, x):
        # Placeholder for relative positional bias (kept as identity to match original)
        return x

    def _select_landmarks(self, K_len: int, m: int, device: torch.device):
        """
        Deterministic uniform landmark indices for stability:
        [0, ..., K_len-1] -> m indices approximately evenly spaced.
        """
        if m >= K_len:
            idx = torch.arange(K_len, device=device)
        else:
            idx = torch.linspace(0, K_len - 1, steps=m, device=device).round().long().unique()
            # ensure we have exactly m indices (pad or trim)
            if idx.numel() < m:
                pad = torch.arange(0, m - idx.numel(), device=device)
                pad = (pad % K_len)
                idx = torch.unique(torch.cat([idx, pad]))[:m]
            elif idx.numel() > m:
                idx = idx[:m]
        return idx  # (m,)

    def forward(self, q, k, v, skip=None):
        """
        Returns: (b X Y W1 W2 d)
        """
        assert k.shape == v.shape
        b, n, q_height, q_width, q_win_height, q_win_width, d = q.shape
        _, _, kv_height, kv_width, kv_win_height, kv_win_width, _ = k.shape
        assert q_height * q_width == kv_height * kv_width

        device = q.device
        # flatten for (x*y)
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')

        # linear projections with heads
        q = self.to_q(q)  # b L Q (heads*dim_head)
        k = self.to_k(k)  # b L K (heads*dim_head)
        v = self.to_v(v)  # b L K (heads*dim_head)

        # group heads into batch
        h = self.heads
        dh = self.dim_head
        # q: b L Q (h*dh) -> (b*h) L Q dh
        q = rearrange(q, 'b L Q (h d) -> (b h) L Q d', h=h, d=dh)
        k = rearrange(k, 'b L K (h d) -> (b h) L K d', h=h, d=dh)
        v = rearrange(v, 'b L K (h d) -> (b h) L K d', h=h, d=dh)

        # scale queries
        q = q * self.scale

        BHL, L, Q, d_ = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
        K_len = k.shape[2]
        m = min(self._rank, K_len)

        # choose landmarks indices once per forward (shared across (B*h, L) for vectorization)
        idx = self._select_landmarks(K_len, m, device)  # (m,)

        # gather landmark keys
        k_land = k[:, :, idx, :]  # (B*h) L m d
        v_land = v[:, :, idx, :]  # (B*h) L m d  (not strictly needed but can be handy)

        # Nyström components
        # K_qL = softmax(q @ k_land^T)   -> (B*h) L Q m
        # A    = softmax(k_land @ k_land^T) -> (B*h) L m m
        # P    = softmax(k_land @ k^T)  -> (B*h) L m K
        # att ≈ K_qL @ pinv(A) @ P
        # out ≈ att @ v  -> (B*h) L Q d

        # compute sims with numerical stability (row-wise)
        def row_softmax(x):
            x_max = x.max(dim=-1, keepdim=True).values
            x = x - x_max
            expx = torch.exp(x)
            return expx / (expx.sum(dim=-1, keepdim=True) + self.eps)

        # (B*h) L Q m
        K_qL = einsum('b l q d, b l m d -> b l q m', q, k_land)
        K_qL = row_softmax(K_qL)

        # (B*h) L m m
        A = einsum('b l m d, b l n d -> b l m n', k_land, k_land)
        A = row_softmax(A)

        # (B*h) L m K
        P = einsum('b l m d, b l k d -> b l m k', k_land, k)
        P = row_softmax(P)

        # pinv with damping for stability
        I = torch.eye(m, device=device).unsqueeze(0).unsqueeze(0)  # 1 1 m m
        A_damped = A + 1e-4 * I
        A_inv = torch.linalg.pinv(A_damped)  # (B*h) L m m  (batched pinv)

        # att approx: (B*h) L Q K
        att_approx = einsum('b l q m, b l m n, b l n k -> b l q k', K_qL, A_inv, P)

        if self.rel_pos_emb:
            att_approx = self.add_rel_pos_emb(att_approx)

        # out: (B*h) L Q d
        out = einsum('b l q k, b l k d -> b l q d', att_approx, v)

        # merge heads and restore spatial
        out = rearrange(out, '(b h) L (nq) d -> b nq L (h d)', h=h, b=b, nq=1)
        out = rearrange(out, 'b 1 (x_y) (hd) -> b x y (hd)',
                        x=q_height, y=q_width, hd=h*dh)

        out = self.proj(out)  # b x y d
        out = rearrange(out, 'b x y d -> b x y 1 1 d')
        # Now expand back to (b X Y W1 W2 d)
        out = out.expand(-1, q_height, q_width, q_win_height, q_win_width, -1)
        return out


class CrossViewSwapAttention(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        index: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        q_win_size: list,
        feat_win_size: list,
        heads: list,
        dim_head: list,
        bev_embedding_flag: list,
        rel_pos_emb: bool = False,  # to-do
        no_image_features: bool = False,
        skip: bool = True,
        norm=nn.LayerNorm,
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed_flag = bev_embedding_flag[index]
        if self.bev_embed_flag:
            self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.q_win_size = q_win_size[index]
        self.feat_win_size = feat_win_size[index]
        self.rel_pos_emb = rel_pos_emb

        # >>> Replaced by Nyström attention <<<
        self.cross_win_attend_1 = CrossWinAttentionNystrom(dim, heads[index], dim_head[index], qkv_bias,
                                                           rel_pos_emb=rel_pos_emb, norm=norm, base_rank=32)
        self.cross_win_attend_2 = CrossWinAttentionNystrom(dim, heads[index], dim_head[index], qkv_bias,
                                                           rel_pos_emb=rel_pos_emb, norm=norm, base_rank=32)
        self.skip = skip

        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def pad_divisble(self, x, win_h, win_w):
        """Pad the x to be divible by window size."""
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h) // win_h) * win_h, ((w + win_w) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    @torch.no_grad()
    def _update_attention_rank(self, object_count: Optional[torch.Tensor]):
        """
        Decide rank from object_count and update both attention blocks.
        Use a single rank for the whole batch to keep vectorization simple.
        """
        if object_count is None:
            r = 32
        else:
            # use max across batch for safety (monotone non-degrading)
            max_count = int(object_count.max().item())
            r = rank_from_count(max_count, r_min=16, r_max=128, k=6.0, c0=16.0)
        self.cross_win_attend_1.set_rank(r)
        self.cross_win_attend_2.set_rank(r)

    def forward(
        self,
        index: int,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
        object_count: Optional[torch.Tensor] = None, #object_count
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """

        # rank update by object_count
        self._update_attention_rank(object_count)

        #디버깅(원하면 주석처리)
        if object_count is not None:
            print(">> object_count(crossviewswapattention):", object_count.shape, object_count)
            for bi in range(min(8, object_count.numel())):
                print(f"Batch {bi} object count: {int(object_count[bi].item())}")
        else:
            print(">> object_count(crossviewswapattention) is None")

        b, n, _, _, _ = feature.shape
        _, _, H, W = x.shape

        pixel = self.image_plane                                                # b n 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]                                                     # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                                                # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
        d = E_inv @ cam                                                         # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 h w
        d_embed = self.img_embed(d_flat)                                        # (b n) d h w

        img_embed = d_embed - c_embed                                           # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w

        # todo: some hard-code for now.
        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        elif index == 3:
            world = bev.grid3[:2]

        if self.bev_embed_flag:
            # 2 H W
            w_embed = self.bev_embed(world[None])                                   # 1 d H W
            bev_embed = w_embed - c_embed                                           # (b n) d H W
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)      # b n d H W

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)              # (b n) d h w
        else:
            key_flat = img_embed                                                # (b n) d h w

        val_flat = self.feature_linear(feature_flat)                            # (b n) d h w

        # Expand + refine the BEV embedding
        if self.bev_embed_flag:
            query = query_pos + x[:, None]
        else:
            query = x[:, None]  # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w

        # pad divisible
        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])

        # local-to-local cross-attention
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])  # window partition
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # window partition
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # window partition

        query = rearrange(self.cross_win_attend_1(query, key, val,
                                                skip=rearrange(x,
                                                            'b d (x w1) (y w2) -> b x y w1 w2 d',
                                                             w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None),
                       'b x y w1 w2 d  -> b (x w1) (y w2) d')    # reverse window to feature

        query = query + self.mlp_1(self.prenorm_1(query))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)              # b n x y d

        # local-to-global cross-attention
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])  # window partition
        key = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')  # reverse window to feature
        key = rearrange(key, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # grid partition
        val = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')  # reverse window to feature
        val = rearrange(val, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # grid partition

        query = rearrange(self.cross_win_attend_2(query,
                                                  key,
                                                  val,
                                                  skip=rearrange(x_skip,
                                                            'b (x w1) (y w2) d -> b x y w1 w2 d',
                                                            w1=self.q_win_size[0],
                                                            w2=self.q_win_size[1])
                                                  if self.skip else None),
                       'b x y w1 w2 d  -> b (x w1) (y w2) d')  # reverse grid to feature

        query = query + self.mlp_2(self.prenorm_2(query))

        query = self.postnorm(query)

        query = rearrange(query, 'b H W d -> b d H W')

        return query


class PyramidAxialEncoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            cross_view_swap: dict,
            bev_embedding: dict,
            self_attn: dict,
            dim: list,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()
        downsample_layers = list()

        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)])
            layers.append(layer)

            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(dim[i], dim[i] // 2,
                                  kernel_size=3, stride=1,
                                  padding=1, bias=False),
                        nn.PixelUnshuffle(2),
                        nn.Conv2d(dim[i+1], dim[i+1],
                                  3, padding=1, bias=False),
                        nn.BatchNorm2d(dim[i+1]),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(dim[i+1],
                                  dim[i+1], 1, padding=0, bias=False),
                        nn.BatchNorm2d(dim[i+1])
                        )))

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        # self.self_attn = Attention(dim[-1], **self_attn)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # b n c h w
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4

        # ✅ object_count (shape [b]) 지원
        object_count = batch.get('object_count', None)

        #디버깅(원하면 주석처리)
        if object_count is not None:
            print(">> object_count(pyramid axial encoder):", object_count.shape, object_count)
        else:
            print(">> object_count(pyramid axial encoder) is None")

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W

        for i, (cross_view, feature, layer) in \
                enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features)-1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)

        # x = self.self_attn(x)

        return x


if __name__ == "__main__":
    import os
    import re
    import yaml
    def load_yaml(file):
        stream = open(file, 'r')
        loader = yaml.Loader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        param = yaml.load(stream, Loader=loader)
        if "yaml_parser" in param:
            param = eval(param["yaml_parser"])(param)
        return param

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    block = CrossWinAttentionNystrom(dim=128,
                                     heads=4,
                                     dim_head=32,
                                     qkv_bias=True)
    block.cuda()
    test_q = torch.rand(1, 6, 5, 5, 5, 5, 128, device='cuda')
    test_k = torch.rand(1, 6, 5, 5, 6, 12, 128, device='cuda')
    test_v = torch.rand(1, 6, 5, 5, 6, 12, 128, device='cuda')

    output = block(test_q, test_k, test_v)
    print("CrossWinAttentionNystrom output:", output.shape)

    # Dummy encoder forward (requires actual backbone/params to truly run)
    # This section is illustrative; integrate with your training script.
