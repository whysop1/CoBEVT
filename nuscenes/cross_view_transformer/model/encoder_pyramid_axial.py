# encoder_pyramid_axial.py
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional, Tuple

from .decoder import DecoderBlock  # 원본 구성 유지

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


# -----------------------------
# Utils
# -----------------------------
def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)  # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)              # 3 h w
    indices = indices[None]                                            # 1 3 h w
    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters
    return [
        [0., -sw, w/2.],
        [-sh, 0., h*offset + h/2.],
        [0., 0., 1.]
    ]


def safe_obj_count_tensor(
    object_count: Optional[torch.Tensor],
    b: int,
    n: int,
    device: torch.device
) -> torch.Tensor:
    """
    object_count를 (b, n) 형태로 안전히 반환.
    다양한 입력 길이(1, b, n, b*n 등)를 허용.
    """
    if object_count is None:
        return torch.zeros(b, n, device=device, dtype=torch.float32)

    x = object_count
    try:
        x = x.to(device=device, dtype=torch.float32).view(-1)
    except Exception:
        return torch.zeros(b, n, device=device, dtype=torch.float32)

    if x.numel() == 1:
        return x.view(1, 1).expand(b, n)
    if x.numel() == b * n:
        return x.view(b, n)
    if x.numel() == b:
        return x.view(b, 1).expand(b, n)
    if x.numel() == n:
        return x.view(1, n).expand(b, n)
    return torch.zeros(b, n, device=device, dtype=torch.float32)


# -----------------------------
# Basic modules
# -----------------------------
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
        self.kwargs = {'stride': stride, 'padding': padding}

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
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)
        V_inv = torch.FloatTensor(V).inverse()
        for i, scale in enumerate(upsample_scales):
            h = bev_height // scale
            w = bev_width // scale
            grid = generate_grid(h, w).squeeze(0)
            grid[0] = bev_width * grid[0]
            grid[1] = bev_height * grid[1]
            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)
            self.register_buffer(f'grid{i}', grid, persistent=False)
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, bev_height // upsample_scales[0], bev_width // upsample_scales[0])
        )

    def get_prior(self):
        return self.learned_features


# -----------------------------
# Object-Aware Positional Encoding
# -----------------------------
class ObjectAwarePE(nn.Module):
    def __init__(self, dim: int, bias_scale: float = 1.0):
        super().__init__()
        self.bias_scale = bias_scale
        self.scalar_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.logit_bias = nn.Sequential(nn.Linear(1, 1))

    def forward(self, obj_count_bn: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tanh(torch.log1p(obj_count_bn.clamp(min=0.0)) * 0.5)  # (b,n)
        s = self.scalar_mlp(x.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)  # (b,n,d,1,1)
        s = s.expand(-1, -1, -1, H, W)  # (b,n,d,H,W)
        lb = self.logit_bias(x.unsqueeze(-1)) * self.bias_scale  # (b,n,1)
        lb = lb.unsqueeze(-1)  # (b,n,1,1)
        return s, lb


# -----------------------------
# Attention / CrossWinAttention
# -----------------------------
class Attention(nn.Module):
    def __init__(self, dim, dim_head=32, dropout=0., window_size=25):
        super().__init__()
        assert (dim % dim_head) == 0
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        b, _, h, w = x.shape
        H = self.heads
        x = rearrange(x, 'b d h w -> b (h w) d')
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=H), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b m (h w) d -> b h w (m d)', h=h, w=w)
        out = self.to_out(out)
        return rearrange(out, 'b h w d -> b d h w')


class CrossWinAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        qkv_bias,
        rel_pos_emb=False,
        norm=nn.LayerNorm,
        topk_ratio: float = 0.25,
        min_topk: int = 32,
        query_keep_ratio: float = 0.75,
        min_query_keep: int = 64,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.rel_pos_emb = rel_pos_emb

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.proj = nn.Linear(heads * dim_head, dim)

        self.topk_ratio = topk_ratio
        self.min_topk = min_topk
        self.query_keep_ratio = query_keep_ratio
        self.min_query_keep = min_query_keep

    @staticmethod
    def _sparse_topk_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
        K = logits.size(-1)
        if k >= K:
            return torch.ones_like(logits, dtype=torch.bool)
        topk = torch.topk(logits, k=k, dim=-1).indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, topk, True)
        return mask

    def _dynamic_query_selector(self, q_tokens: torch.Tensor, keep: int) -> torch.Tensor:
        sal = q_tokens.pow(2).sum(dim=-1)  # (B*,L,Q)
        Q = q_tokens.size(2)
        keep = min(max(keep, 1), Q)
        idx = torch.topk(sal, k=keep, dim=-1).indices  # (B*,L,keep)
        mask = torch.zeros_like(sal, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        mask = mask.unsqueeze(-1)
        return q_tokens * mask

    def forward(
        self,
        q, k, v,
        skip=None,
        logit_additive_bias: Optional[torch.Tensor] = None
    ):
        assert k.shape == v.shape
        b, n, qH, qW, qw1, qw2, d = q.shape
        _, _, kH, kW, kw1, kw2, _ = k.shape
        assert qH * qW == kH * kW

        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        H = self.heads
        Dh = self.dim_head
        q = rearrange(q, 'b l q (h d) -> (b h) l q d', h=H, d=Dh)
        k = rearrange(k, 'b l k (h d) -> (b h) l k d', h=H, d=Dh)
        v = rearrange(v, 'b l k (h d) -> (b h) l k d', h=H, d=Dh)

        Q = q.size(2)
        keep_q = max(int(Q * self.query_keep_ratio), self.min_query_keep)
        q_pruned = self._dynamic_query_selector(q, keep=keep_q)

        q_scaled = q_pruned * self.scale
        logits = torch.einsum('b l q d, b l k d -> b l q k', q_scaled, k)

        if logit_additive_bias is not None:
            lb = logit_additive_bias.mean(dim=1, keepdim=True)
            lb = lb.repeat(1, logits.size(1), logits.size(2), logits.size(3))
            times = logits.size(0) // b
            lb = lb.repeat(times, 1, 1, 1)
            logits = logits + lb

        K = logits.size(-1)
        topk = max(int(K * self.topk_ratio), self.min_topk)
        mask = self._sparse_topk_mask(logits, k=topk)
        logits = logits.masked_fill(~mask, float('-inf'))

        att = torch.softmax(logits, dim=-1)
        a = torch.einsum('b l q k, b l k d -> b l q d', att, v)

        a = rearrange(a, '(b h) l (n w1 w2) d -> b n l w1 w2 (h d)', b=b, h=H, n=n, w1=qw1, w2=qw2)
        a = rearrange(a, 'b n (x y) w1 w2 d -> b n x y w1 w2 d', x=qH, y=qW)
        z = self.proj(a)

        z = z.mean(1)
        if skip is not None:
            z = z + skip
        return z


# -----------------------------
# CrossViewSwapAttention (with OAPE + sparse/dynamic)
# -----------------------------
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
        rel_pos_emb: bool = False,
        no_image_features: bool = False,
        skip: bool = True,
        norm=nn.LayerNorm,
        topk_ratio: float = 0.25,
        min_topk: int = 32,
        query_keep_ratio: float = 0.75,
        min_query_keep: int = 64,
        oape_bias_scale: float = 1.0,
    ):
        super().__init__()

        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False)
        )

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False)
            )

        self.bev_embed_flag = bev_embedding_flag[index]
        if self.bev_embed_flag:
            self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.q_win_size = q_win_size[index]
        self.feat_win_size = feat_win_size[index]
        self.rel_pos_emb = rel_pos_emb

        self.cross_win_attend_1 = CrossWinAttention(
            dim, heads[index], dim_head[index], qkv_bias,
            rel_pos_emb=rel_pos_emb, norm=norm,
            topk_ratio=topk_ratio, min_topk=min_topk,
            query_keep_ratio=query_keep_ratio, min_query_keep=min_query_keep,
        )
        self.cross_win_attend_2 = CrossWinAttention(
            dim, heads[index], dim_head[index], qkv_bias,
            rel_pos_emb=rel_pos_emb, norm=norm,
            topk_ratio=topk_ratio, min_topk=min_topk,
            query_keep_ratio=query_keep_ratio, min_query_keep=min_query_keep,
        )
        self.skip = skip

        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

        self.oape = ObjectAwarePE(dim=dim, bias_scale=oape_bias_scale)

    @staticmethod
    def pad_divisble(x, win_h, win_w):
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h) // win_h) * win_h, ((w + win_w) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    def forward(
        self,
        index: int,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
        object_count: Optional[torch.Tensor] = None,
    ):
        b, n, _, _, _ = feature.shape
        _, d, H, W = x.shape
        device = x.device

        obj_bn = safe_obj_count_tensor(object_count, b, n, device)
        oape_query_bias, oape_logit_bias = self.oape(obj_bn, H, W)  # (b,n,d,H,W), (b,n,1,1)

        pixel = self.image_plane
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]
        c_embed = self.cam_embed(c_flat)

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        cam = I_inv @ pixel_flat
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)
        d_dir = E_inv @ cam
        d_flat = rearrange(d_dir, 'b n d (h w) -> (b n) d h w', h=h, w=w)
        d_embed = self.img_embed(d_flat)

        img_embed = d_embed - c_embed
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)

        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        else:
            world = bev.grid3[:2]

        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])  # 1,d,H,W
            bev_embed = w_embed - c_embed
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)
            query_pos = query_pos + oape_query_bias

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)
        else:
            key_flat = img_embed
        val_flat = self.feature_linear(feature_flat)

        if self.bev_embed_flag:
            query = query_pos + x[:, None]
        else:
            query = x[:, None]

        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)

        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])

        query_w = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                            w1=self.q_win_size[0], w2=self.q_win_size[1])
        key_w = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_w = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        skip_local = None
        if self.skip:
            skip_local = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                                   w1=self.q_win_size[0], w2=self.q_win_size[1])

        query_ll = self.cross_win_attend_1(
            query_w, key_w, val_w,
            skip=skip_local,
            logit_additive_bias=oape_logit_bias
        )
        query_ll = rearrange(query_ll, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query_ll = query_ll + self.mlp_1(self.prenorm_1(query_ll))

        x_skip = query_ll
        query_glb = repeat(query_ll, 'b x y d -> b n x y d', n=n)
        query_glb = rearrange(query_glb, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                              w1=self.q_win_size[0], w2=self.q_win_size[1])

        key_g = rearrange(key_w, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key_g = rearrange(key_g, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                          w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_g = rearrange(val_w, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val_g = rearrange(val_g, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                          w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        skip_glb = None
        if self.skip:
            skip_glb = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d',
                                 w1=self.q_win_size[0], w2=self.q_win_size[1])

        query_glb = self.cross_win_attend_2(
            query_glb, key_g, val_g,
            skip=skip_glb,
            logit_additive_bias=oape_logit_bias
        )
        query_glb = rearrange(query_glb, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query_glb = query_glb + self.mlp_2(self.prenorm_2(query_glb))

        out = self.postnorm(query_glb)
        out = rearrange(out, 'b H W d -> b d H W')
        return out


# -----------------------------
# PyramidAxialEncoder (reg 제거)
# -----------------------------
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

        cross_views = []
        layers = []
        downsample_layers = []

        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)])
            layers.append(layer)

            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(dim[i], dim[i] // 2, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.PixelUnshuffle(2),
                        nn.Conv2d(dim[i+1], dim[i+1], 3, padding=1, bias=False),
                        nn.BatchNorm2d(dim[i+1]),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(dim[i+1], dim[i+1], 1, padding=0, bias=False),
                        nn.BatchNorm2d(dim[i+1])
                    )
                ))

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # (b*n, c, h, w)
        I_inv = batch['intrinsics'].inverse()           # (b, n, 3, 3)
        E_inv = batch['extrinsics'].inverse()           # (b, n, 4, 4)

        object_count = batch.get('object_count', None)

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()              # (d, H, W)
        x = repeat(x, '... -> b ...', b=b)              # (b, d, H, W)

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features) - 1:
                x = self.downsample_layers[i](x)

        return x


# -----------------------------
# (Optional) quick smoke test at script run
# -----------------------------
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

    # 간단 shape 테스트 (실제 config로 바꿔 사용하세요)
    B, N = 2, 6
    d_in = 128
    Hq, Wq = 25, 25
    hf, wf = 28, 60

    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_shapes = [(B * N, d_in, Hq, Wq)]
        def forward(self, x):
            return [torch.rand(B * N, d_in, Hq, Wq)]

    backbone = DummyBackbone()
    cross_view = {'image_height': hf, 'image_width': wf}
    cross_view_swap = {
        'q_win_size': [[5, 5]],
        'feat_win_size': [[6, 12]],
        'heads': [4],
        'dim_head': [32],
        'bev_embedding_flag': [True],
    }
    bev_embedding = {
        'sigma': 0.01, 'bev_height': Hq, 'bev_width': Wq,
        'h_meters': 50, 'w_meters': 50, 'offset': 0.0, 'upsample_scales': [1]
    }
    self_attn = {}

    enc = PyramidAxialEncoder(backbone, cross_view, cross_view_swap, bev_embedding, self_attn, dim=[128], middle=[1])
    batch = {
        'image': torch.rand(B, N, 3, hf, wf),
        'intrinsics': torch.eye(3).view(1,1,3,3).repeat(B,N,1,1),
        'extrinsics': torch.eye(4).view(1,1,4,4).repeat(B,N,1,1),
        'object_count': torch.tensor([3]).repeat(B*N)  # 샘플용
    }
    out = enc(batch)
    print('encoder out shape:', out.shape)
