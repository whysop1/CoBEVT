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


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 25
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, _, height, width, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b d h w -> b (h w) d')

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b m (h w) d -> b h w (m d)',
                        h = height, w = width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, 'b h w d -> b d h w')


class CrossWinAttention(nn.Module):
    """
    ABR-ready cross-window attention:
      - keeps maximum heads fixed at init time
      - allows per-call 'head_gate' (shape: [H]) to reallocate effective head usage
    """
    def __init__(self, dim, heads, dim_head, qkv_bias, rel_pos_emb=False, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads              # maximum heads available
        self.dim_head = dim_head
        self.rel_pos_emb = rel_pos_emb

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)

    def add_rel_pos_emb(self, x):
        return x

    def forward(self, q, k, v, skip=None, head_gate: Optional[torch.Tensor] = None):
        """
        q: (b n X Y W1 W2 d)
        k: (b n x y w1 w2 d)
        v: (b n x y w1 w2 d)
        head_gate: optional [H] tensor to reweight heads (ABR)
        return: (b X Y W1 W2 d)
        """
        assert k.shape == v.shape
        _, view_size, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        _, _, kv_height, kv_width, _, _, _ = k.shape
        assert q_height * q_width == kv_height * kv_width

        # flattening
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')


        # Project
        q = self.to_q(q)                                # b (X Y) (n W1 W2) (H*Dh)
        k = self.to_k(k)                                # b (X Y) (n w1 w2) (H*Dh)
        v = self.to_v(v)                                # b (X Y) (n w1 w2) (H*Dh)

        # Reshape to heads
        H, Dh = self.heads, self.dim_head
        q = rearrange(q, 'b L Q (h d) -> b h L Q d', h=H, d=Dh)  # b H (X Y) (n W1 W2) Dh
        k = rearrange(k, 'b L K (h d) -> b h L K d', h=H, d=Dh)
        v = rearrange(v, 'b L K (h d) -> b h L K d', h=H, d=Dh)

        # ---- ABR head gating ----
        if head_gate is not None:
            # head_gate: [H] in [0,1], broadcast to q/k/v
            gate = head_gate.view(1, H, 1, 1, 1)  # b_dim, h, L, Q/K, d
            q = q * gate
            k = k * gate
            v = v * gate

        # Group the head dim with batch dim for attention
        q = rearrange(q, 'b h L Q d -> (b h) L Q d')
        k = rearrange(k, 'b h L K d -> (b h) L K d')
        v = rearrange(v, 'b h L K d -> (b h) L K d')

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)  # (b*H) (X Y) (n W1 W2) (n w1 w2)

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)  # (b*H) (X Y) (n W1 W2) Dh
        a = rearrange(a, '(b h) L Q d -> b L Q (h d)', h=H, d=Dh)  # b (X Y) (n W1 W2) (H*Dh)
        a = rearrange(a, ' b (x_y) (n_w1w2) d -> b n x y w1 w2 d',
            x=q_height, y=q_width, w1=q_win_height, w2=q_win_width, n=view_size)

        # Combine multiple heads
        z = self.proj(a)

        # reduce cameras: (b n X Y W1 W2 d) -> (b X Y W1 W2 d)
        z = z.mean(1)

        # Optional skip connection
        if skip is not None:
            z = z + skip
        return z


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

        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.skip = skip

        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

        # ---- ABR hyper-parameters (you can expose these to config) ----
        self.abr_min_frac = 0.25   # 최소 local 혹은 global에 줄 수 있는 헤드 비율
        self.abr_max_frac = 0.75   # 최대 비율
        self.abr_mid_count = 30.0  # object_count가 이 값일 때 local/global 균형
        self.abr_sharpness = 0.15  # 선형 맵핑 기울기 (0~1 사이에서 얼마나 빨리 치우칠지)

    def pad_divisble(self, x, win_h, win_w):
        """Pad the x to be divible by window size."""
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h) // win_h) * win_h, ((w + win_w) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    # ---------------- ABR utilities ----------------
    @staticmethod
    def _normalize_count(obj_cnt: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """accepts [B] (int/long ok) -> float [B] scaled to ~[0,1]"""
        if obj_cnt is None:
            return None
        if obj_cnt.dim() > 1:
            s = obj_cnt.reshape(obj_cnt.shape[0], -1).sum(dim=1)
        else:
            s = obj_cnt
        s = s.to(torch.float32)
        # robust scaling: x / (mean + std + eps)
        mean = s.mean().clamp(min=1.0)
        std  = s.std().clamp(min=1.0)
        x = s / (mean + std + 1e-6)
        x = x.clamp(0.0, 2.0) * 0.5  # ~[0,1]
        return x

    def _abr_head_gates(self, object_count: Optional[torch.Tensor], device, heads: int):
        """
        Returns head gates (H,) for local(attend_1) and global(attend_2)
        Keep the total 'effective' heads roughly constant by complementary gates.
        """
        H = heads
        if object_count is None:
            # neutral split
            frac_local = 0.5
        else:
            cnt = self._normalize_count(object_count)  # [B] in ~[0,1]
            # use batch mean to keep tensor shapes uniform across batch
            c = float(cnt.mean().item())
            # map c in [0,1] to [abr_min_frac, abr_max_frac] with mild slope
            base = 0.5 + (c - 0.5) * (2 * self.abr_sharpness)  # around 0.5
            frac_local = float(torch.tensor(base).clamp(self.abr_min_frac, self.abr_max_frac).item())

        # local gets frac_local, global gets (1 - frac_local)
        h_local  = max(1, int(round(frac_local * H)))
        h_global = max(1, int(round((1.0 - frac_local) * H)))

        # build monotone gates: first h heads are active(=1), rest 0
        gate_local  = torch.zeros(H, device=device, dtype=torch.float32)
        gate_global = torch.zeros(H, device=device, dtype=torch.float32)
        gate_local[:h_local]   = 1.0
        gate_global[:h_global] = 1.0

        # L1-normalize to keep magnitude balanced before projection
        gate_local  = gate_local * (H / gate_local.sum().clamp(min=1.0))
        gate_global = gate_global * (H / gate_global.sum().clamp(min=1.0))
        return gate_local, gate_global, h_local, h_global

    # ---------------- forward ----------------
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

        # 디버깅 (선택)
        if object_count is not None:
            print(">> object_count(crossviewswapattention):", object_count.shape, object_count)

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

        # ---------------- ABR: decide head gates for two attentions ----------------
        gate1, gate2, h_loc, h_glo = self._abr_head_gates(object_count, device=x.device,
                                                          heads=self.cross_win_attend_1.heads)

        # ---------------- local-to-local cross-attention ----------------
        query_ll = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                             w1=self.q_win_size[0], w2=self.q_win_size[1])  # window partition
        key_ll = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                           w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # window partition
        val_ll = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                           w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # window partition

        query = rearrange(
            self.cross_win_attend_1(
                query_ll, key_ll, val_ll,
                skip=rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                               w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None,
                head_gate=gate1
            ),
            'b x y w1 w2 d  -> b (x w1) (y w2) d'
        )    # reverse window to feature

        query = query + self.mlp_1(self.prenorm_1(query))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)              # b n x y d

        # ---------------- local-to-global cross-attention ----------------
        query_lg = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                             w1=self.q_win_size[0], w2=self.q_win_size[1])  # window partition
        key_g = rearrange(key_ll, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')  # reverse window to feature
        key_g = rearrange(key_g, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                          w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # grid partition
        val_g = rearrange(val_ll, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')  # reverse window to feature
        val_g = rearrange(val_g, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                          w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # grid partition

        query = rearrange(
            self.cross_win_attend_2(
                query_lg, key_g, val_g,
                skip=rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d',
                               w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None,
                head_gate=gate2
            ),
            'b x y w1 w2 d  -> b (x w1) (y w2) d'
        )  # reverse grid to feature

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

        # object_count: expected shape [B] (e.g., tensor([16, 47, ...]))
        object_count = batch.get('object_count', None)

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

    block = CrossWinAttention(dim=128,
                              heads=4,
                              dim_head=32,
                              qkv_bias=True,)
    block.cuda()
    test_q = torch.rand(1, 6, 5, 5, 5, 5, 128).cuda()
    test_k = torch.rand(1, 6, 5, 5, 6, 12, 128).cuda()
    test_v = test_k.clone()

    output = block(test_q, test_k, test_v, head_gate=torch.tensor([1.,1.,0.,0.], device=test_q.device))
    print(output.shape)

    image = torch.rand(1, 6, 128, 28, 60).cuda()            # b n c h w
    I_inv = torch.rand(1, 6, 3, 3).cuda()                   # b n 3 3
    E_inv = torch.rand(1, 6, 4, 4).cuda()                   # b n 4 4
    feature = torch.rand(1, 6, 128, 25, 25).cuda()
    x = torch.rand(1, 128, 25, 25).cuda()                   # b d H W

    params = load_yaml('config/model/cvt_pyramid_swap.yaml')
    print(params)
