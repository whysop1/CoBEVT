# encoder_pyramid_axial.py (Temporal Fusion 적용 + DDP unused-parameter 보호용 패치 포함)
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List, Optional

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
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
            self.register_buffer('grid%d' % i, grid, persistent=False)

        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, bev_height // upsample_scales[0], bev_width // upsample_scales[0])
        )

    def get_prior(self):
        return self.learned_features


class Attention(nn.Module):
    def __init__(self, dim, dim_head=32, dropout=0., window_size=25):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

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
        batch, _, height, width, device, h = *x.shape, x.device, self.heads
        x = rearrange(x, 'b d h w -> b (h w) d')
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
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
        assert k.shape == v.shape
        _, view_size, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        _, _, kv_height, kv_width, _, _, _ = k.shape
        assert q_height * q_width == kv_height * kv_width

        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        dot = self.scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)
        att = dot.softmax(dim=-1)
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)
        a = rearrange(a, ' b (x y) (n w1 w2) d -> b n x y w1 w2 d',
            x=q_height, y=q_width, w1=q_win_height, w2=q_win_width)

        z = self.proj(a)
        z = z.mean(1)
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
        rel_pos_emb: bool = False,
        no_image_features: bool = False,
        skip: bool = True,
        norm=nn.LayerNorm,
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

        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.skip = skip

        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def pad_divisble(self, x, win_h, win_w):
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
        _, _, H, W = x.shape

        pixel = self.image_plane
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]
        c_embed = self.cam_embed(c_flat)

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        cam = I_inv @ pixel_flat
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)
        d = E_inv @ cam
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)
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
            w_embed = self.bev_embed(world[None])
            bev_embed = w_embed - c_embed
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)

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

        # local-to-local
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        query = rearrange(self.cross_win_attend_1(query, key, val,
                                                skip=rearrange(x,
                                                            'b d (x w1) (y w2) -> b x y w1 w2 d',
                                                             w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None),
                       'b x y w1 w2 d  -> b (x w1) (y w2) d')

        query = query + self.mlp_1(self.prenorm_1(query))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)

        # local-to-global
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key = rearrange(key, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val = rearrange(val, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        query = rearrange(self.cross_win_attend_2(query,
                                                  key,
                                                  val,
                                                  skip=rearrange(x_skip,
                                                            'b (x w1) (y w2) d -> b x y w1 w2 d',
                                                            w1=self.q_win_size[0],
                                                            w2=self.q_win_size[1])
                                                  if self.skip else None),
                       'b x y w1 w2 d  -> b (x w1) (y w2) d')

        query = query + self.mlp_2(self.prenorm_2(query))
        query = self.postnorm(query)
        query = rearrange(query, 'b H W d -> b d H W')

        return query


# ---------------- Temporal modules ----------------

class ConvGRUCell(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.reset = nn.Conv2d(dim * 2, dim, kernel_size, padding=pad)
        self.update = nn.Conv2d(dim * 2, dim, kernel_size, padding=pad)
        self.out = nn.Conv2d(dim * 2, dim, kernel_size, padding=pad)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros_like(x)
        inp = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.reset(inp))
        z = torch.sigmoid(self.update(inp))
        n = torch.tanh(self.out(torch.cat([x, r * h], dim=1)))
        h_new = (1 - z) * n + z * h
        return h_new


class TemporalBEVModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gru = ConvGRUCell(dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def warp_bev(prev: torch.Tensor, A: Optional[torch.Tensor]) -> torch.Tensor:
        if A is None:
            return prev
        b, d, H, W = prev.shape
        # A expected as (b, 2, 3) mapping output coords to input coords
        grid = F.affine_grid(A, size=(b, d, H, W), align_corners=True)
        return F.grid_sample(prev, grid, align_corners=True)

    def forward(self, cur: torch.Tensor, prev: Optional[torch.Tensor] = None, prev2cur: Optional[torch.Tensor] = None):
        if prev is None:
            return self.gru(cur, None)
        prev_warp = self.warp_bev(prev, prev2cur)
        combined = torch.cat([cur, prev_warp], dim=1)
        gated = self.gate(combined)
        h = self.gru(gated, prev_warp)
        return h


# ---------------- PyramidAxialEncoder with temporal fusion ----------------

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
                    nn.Conv2d(dim[i], dim[i] // 2, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.PixelUnshuffle(2),
                    nn.Conv2d(dim[i+1], dim[i+1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim[i+1], dim[i+1], 1, padding=0, bias=False),
                    nn.BatchNorm2d(dim[i+1])
                ))

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)

        # temporal modules, one per scale
        self.temporal_modules = nn.ModuleList([TemporalBEVModule(dim[i]) for i in range(len(middle))])

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()

        prev_bev = batch.get('prev_bev', None)       # optional (b, d, H, W)
        prev2cur = batch.get('prev2cur_bev', None)   # optional (b, 2, 3) affine

        object_count = batch.get('object_count', None)

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()
        x = repeat(x, '... -> b ...', b=b)  # b d H W

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)

            # Temporal fusion: always call module (even if prev_bev=None) to ensure module params are used.
            if prev_bev is not None:
                # resize prev_bev if necessary to match x spatial
                if prev_bev.shape[-2:] != x.shape[-2:]:
                    prev_resized = F.interpolate(prev_bev, size=x.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    prev_resized = prev_bev
                x = self.temporal_modules[i](x, prev_resized, prev2cur)
            else:
                # still call with prev=None
                x = self.temporal_modules[i](x, None, None)

            if i < len(features) - 1:
                x = self.downsample_layers[i](x)

        # ----------------------------
        # IMPORTANT: DDP safety net for "unused parameter" errors.
        # Make sure all model parameters participate in graph during forward,
        # by creating an aux scalar that depends on every parameter (multiplied by 0).
        # This does not change numerical outputs but ensures DDP sees them as used.
        # ----------------------------
        aux = 0.0
        for p in self.parameters():
            # p.sum()*0.0 ensures parameter appears in graph but contributes 0.
            aux = aux + (p.sum() * 0.0)
        # add aux to x (broadcast): doesn't change x numerically
        x = x + aux

        return x


if __name__ == "__main__":
    # quick smoke test (CPU)
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            # emulate 2-scale outputs (shapes only)
            self.output_shapes = [(1, 128, 28, 60), (1, 128, 14, 30)]
            self.conv = nn.Conv2d(3, 128, 3, padding=1)

        def forward(self, x):
            out1 = F.interpolate(self.conv(x), size=(28, 60), mode='bilinear', align_corners=False)
            out2 = F.interpolate(self.conv(x), size=(14, 30), mode='bilinear', align_corners=False)
            return [out1, out2]

    dim = [128, 128]
    backbone = DummyBackbone()
    cross_view = dict(image_height=224, image_width=480, qkv_bias=True)
    cross_view_swap = dict(q_win_size=[[5, 5], [5, 5]],
                           feat_win_size=[[6, 12], [6, 12]],
                           heads=[4, 4], dim_head=[32, 32],
                           bev_embedding_flag=[True, False])
    bev_embedding = dict(dim=128, sigma=0.02, bev_height=25, bev_width=25, h_meters=100, w_meters=100, offset=0.0, upsample_scales=[1, 2])

    enc = PyramidAxialEncoder(
        backbone=backbone,
        cross_view=cross_view,
        cross_view_swap=cross_view_swap,
        bev_embedding=bev_embedding,
        self_attn={},
        dim=dim,
        middle=[1, 1],
        scale=1.0
    )

    b, n = 2, 6
    image = torch.rand(b, n, 3, 224, 480)
    intr = torch.eye(3).view(1, 1, 3, 3).repeat(b, n, 1, 1)
    extr = torch.eye(4).view(1, 1, 4, 4).repeat(b, n, 1, 1)

    batch = {
        'image': image,
        'intrinsics': intr,
        'extrinsics': extr,
        # optionally provide prev_bev if available:
        # 'prev_bev': torch.zeros(b, dim[0], bev_embedding['bev_height']//bev_embedding['upsample_scales'][0], bev_embedding['bev_width']//bev_embedding['upsample_scales'][0]),
        # 'prev2cur_bev': None
    }

    out = enc(batch)
    print("smoke output shape:", out.shape)
