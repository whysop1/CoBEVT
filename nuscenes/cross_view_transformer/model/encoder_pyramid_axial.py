# encoder_pyramid_axial.py
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional
from .decoder import DecoderBlock

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
        [0., -sw, w / 2.],
        [-sh, 0., h * offset + h / 2.],
        [0., 0., 1.]
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
        """
        super().__init__()

        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()  # 3 3

        for i, scale in enumerate(upsample_scales):
            h = bev_height // scale
            w = bev_width // scale

            grid = generate_grid(h, w).squeeze(0)
            grid[0] = bev_width * grid[0]
            grid[1] = bev_height * grid[1]

            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')  # 3 (h w)
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)  # 3 h w
            self.register_buffer('grid%d' % i, grid, persistent=False)

        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim,
                                bev_height // upsample_scales[0],
                                bev_width // upsample_scales[0]))  # d h w

    def get_prior(self):
        return self.learned_features


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            window_size=25
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, _, height, width, device, h = *x.shape, x.device, self.heads

        x = rearrange(x, 'b d h w -> b (h w) d')
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b m (h w) d -> b h w (m d)', h=height, w=width)
        out = self.to_out(out)
        return rearrange(out, 'b h w d -> b d h w')


class CrossWinAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, rel_pos_emb=False, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head
        self.rel_pos_emb = rel_pos_emb

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)

    def add_rel_pos_emb(self, x):
        return x

    def forward(self, q, k, v, skip=None):
        """
        q: (b n X Y W1 W2 d)
        k: (b n x y w1 w2 d)
        v: (b n x y w1 w2 d)
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

        # Project with multiple heads
        q = self.to_q(q)                                # b (X Y) (n W1 W2) (heads dim_head)
        k = self.to_k(k)                                # b (X Y) (n w1 w2) (heads dim_head)
        v = self.to_v(v)                                # b (X Y) (n w1 w2) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)  # b (X Y) (n W1 W2) (n w1 w2)

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)  # b (X Y) (n W1 W2) d
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)
        a = rearrange(a, ' b (x y) (n w1 w2) d -> b n x y w1 w2 d',
                      x=q_height, y=q_width, w1=q_win_height, w2=q_win_width)

        # Combine multiple heads
        z = self.proj(a)

        # reduce n: (b n X Y W1 W2 d) -> (b X Y W1 W2 d)
        z = z.mean(1)  # for sequential usage, we cannot reduce it!

        # Optional skip connection
        if skip is not None:
            z = z + skip
        return z


class DeformableBEVRefine(nn.Module):
    """
    Lightweight deformable attention applied on BEV map.
    For each BEV location, predict K offsets (in pixels) and K attention weights,
    sample values at those offsets using grid_sample and compute weighted sum.
    """

    def __init__(self, dim, K=4, offset_scale=1.0):
        """
        dim: channel dim of BEV features
        K: number of sample points per BEV location
        offset_scale: initial scale multiplier for predicted offsets (in pixels)
        """
        super().__init__()
        self.dim = dim
        self.K = int(K)
        self.offset_scale = float(offset_scale)

        # offset conv: output 2*K channels (dx,dy for each sample)
        self.offset_conv = nn.Conv2d(dim, 2 * self.K, kernel_size=3, padding=1)
        # weight conv: output K attention logits
        self.weight_conv = nn.Conv2d(dim, self.K, kernel_size=3, padding=1)

        # small projection after aggregation
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # init offsets small
        nn.init.constant_(self.offset_conv.weight, 0.0)
        if self.offset_conv.bias is not None:
            nn.init.constant_(self.offset_conv.bias, 0.0)

    def forward(self, bev: torch.Tensor):
        """
        bev: (b, d, H, W)
        returns: (b, d, H, W) refined
        """
        b, d, H, W = bev.shape
        device = bev.device

        # offsets in pixel units: (b, 2K, H, W)
        offsets = self.offset_conv(bev) * self.offset_scale  # initially near zero
        # weights logits: (b, K, H, W)
        w_logits = self.weight_conv(bev)
        w = F.softmax(w_logits.view(b, self.K, -1), dim=1).view(b, self.K, H, W)  # softmax over K

        # base normalized grid in [-1,1]
        # grid_x: shape (W,), grid_y: shape (H,)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')  # W x H
        # but we want shape H x W, so swap later
        base_x = grid_x.t()  # H x W
        base_y = grid_y.t()  # H x W
        base_grid = torch.stack((base_x, base_y), dim=-1)  # H x W x 2
        base_grid = base_grid.unsqueeze(0).repeat(b, 1, 1, 1)  # b H W 2

        # offsets are in pixels. Convert to normalized coordinates:
        # offsets: (b, 2K, H, W) -> dx_pixels, dy_pixels per K
        offsets = offsets.view(b, self.K, 2, H, W)  # b K 2 H W
        dx = offsets[:, :, 0, :, :]  # b K H W
        dy = offsets[:, :, 1, :, :]  # b K H W

        # convert pixel offsets to normalized [-1,1]
        # dx_norm = dx * (2/(W-1)); dy_norm = dy * (2/(H-1))
        if W > 1:
            dx_norm = dx * (2.0 / float(W - 1))
        else:
            dx_norm = dx * 0.0
        if H > 1:
            dy_norm = dy * (2.0 / float(H - 1))
        else:
            dy_norm = dy * 0.0

        # produce K grids and sample
        sampled = []
        bev_reshape = bev  # (b,d,H,W)
        for k in range(self.K):
            # grid_k: b H W 2
            grid_k = torch.stack((base_grid[..., 0] + dx_norm[:, k, :, :],
                                  base_grid[..., 1] + dy_norm[:, k, :, :]), dim=-1)
            # grid_sample expects shape (b, H_out, W_out, 2) with coords in [-1,1], x is width axis
            sampled_k = F.grid_sample(bev_reshape, grid_k, mode='bilinear', padding_mode='border', align_corners=True)
            # sampled_k: (b, d, H, W)
            sampled.append(sampled_k)

        # stack: b K d H W
        sampled = torch.stack(sampled, dim=1)
        # w: b K H W -> view for broadcast
        w = w.unsqueeze(2)  # b K 1 H W
        aggregated = (sampled * w).sum(dim=1)  # b d H W

        out = self.proj(aggregated)  # b d H W
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
        # Deformable params
        deform_K: int = 4,
        deform_offset_scale: float = 1.0,
        # SRA params
        alpha_cap: float = 0.5,
        alpha_k: float = 0.2,
        alpha_c0: float = 16.0,
    ):
        super().__init__()

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

        # cross-attention blocks (same as original)
        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.skip = skip

        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

        # Deformable BEV refinement
        self.deform_refine = DeformableBEVRefine(dim=dim, K=deform_K, offset_scale=deform_offset_scale)

        # SRA params
        self.alpha_cap = float(alpha_cap)
        self.alpha_k = float(alpha_k)
        self.alpha_c0 = float(alpha_c0)

    def pad_divisble(self, x, win_h, win_w):
        """Pad the x to be divible by window size."""
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h) // win_h) * win_h, ((w + win_w) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    def _alpha_from_count(self, object_count: Optional[torch.Tensor], device: torch.device):
        """
        Compute an alpha scalar in [0, alpha_cap] monotonically increasing with max(object_count).
        Uses a sigmoid-shaped schedule centered at alpha_c0 with slope alpha_k.
        If object_count is None -> return 0.0 (safe: no adapter).
        """
        if object_count is None:
            return torch.tensor(0.0, device=device, dtype=torch.float32)
        try:
            if object_count.dim() == 1:
                counts = object_count.float()
            else:
                counts = object_count.float().sum(dim=-1)
            max_count = float(counts.max().item())
        except Exception:
            return torch.tensor(0.0, device=device, dtype=torch.float32)
        x = self.alpha_k * (max_count - self.alpha_c0)
        sigmoid = 1.0 / (1.0 + math.exp(-x))
        alpha = self.alpha_cap * sigmoid
        return torch.tensor(alpha, device=device, dtype=torch.float32)

    def forward(
        self,
        index: int,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
        object_count: Optional[torch.Tensor] = None, # object_count
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
        # optional debugging prints (comment out for speed)
        # if object_count is not None:
        #     print(">> object_count(crossviewswapattention):", object_count.shape, object_count)

        b, n, _, _, _ = feature.shape
        _, _, H, W = x.shape

        pixel = self.image_plane  # 1 1 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]  # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]  # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)  # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')  # 1 1 3 (h w)
        cam = I_inv @ pixel_flat  # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)  # b n 4 (h w)
        d = E_inv @ cam  # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)  # (b n) 4 h w
        d_embed = self.img_embed(d_flat)  # (b n) d h w

        img_embed = d_embed - c_embed  # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)  # (b n) d h w

        # world grid selection
        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        elif index == 3:
            world = bev.grid3[:2]
        else:
            world = bev.grid0[:2]

        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])  # 1 d H W
            bev_embed = w_embed - c_embed  # (b n) d H W
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)  # (b n) d H W
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)  # b n d H W

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')  # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)  # (b n) d h w
        else:
            key_flat = img_embed  # (b n) d h w

        val_flat = self.feature_linear(feature_flat)  # (b n) d h w

        # Expand + refine the BEV embedding
        if self.bev_embed_flag:
            query = query_pos + x[:, None]
        else:
            query = x[:, None]  # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)  # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)  # b n d h w

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

        # -------------------------
        # Deformable BEV refinement + SRA blending
        # -------------------------
        p_base = query  # b,d,H,W

        # compute alpha
        alpha = self._alpha_from_count(object_count, device=p_base.device)  # scalar tensor
        alpha_ = alpha.view(1, 1, 1, 1)

        # if alpha == 0 -> skip refine for safety
        if float(alpha.item()) <= 0.0:
            return p_base

        deform_out = self.deform_refine(p_base)  # b,d,H,W
        p_out = (1.0 - alpha_) * p_base + alpha_ * deform_out

        return p_out


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

        object_count = batch.get('object_count', None)

        # optional debug
        # if object_count is not None:
        #     print(">> object_count(pyramid axial encoder):", object_count.shape, object_count)
        # else:
        #     print(">> object_count(pyramid axial encoder) is None")

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
    # quick smoke test for module import & shapes (doesn't run full training)
    block = CrossWinAttention(dim=128, heads=4, dim_head=32, qkv_bias=True)
    print("Module loaded.")
