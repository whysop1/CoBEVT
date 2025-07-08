'''

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List

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
        num_clusters: int = 10,
    ):
        super().__init__()

        self.num_clusters = num_clusters
        self.dim = dim
        h = bev_height // upsample_scales[0]
        w = bev_width // upsample_scales[0]

        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)
        V_inv = torch.FloatTensor(V).inverse()

        for i, scale in enumerate(upsample_scales):
            hh = bev_height // scale
            ww = bev_width // scale
            grid = generate_grid(hh, ww).squeeze(0)
            grid[0] = bev_width * grid[0]
            grid[1] = bev_height * grid[1]
            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')
            grid = rearrange(grid, 'd (h w) -> d h w', h=hh, w=ww)
            self.register_buffer(f'grid{i}', grid, persistent=False)

        self.learned_features = nn.Parameter(
            sigma * torch.randn(num_clusters, dim, h, w)
        )

    def get_prior(self, cluster_ids: torch.Tensor):
        """
        cluster_ids: Tensor of shape (B, N) with cluster indices
        Returns: Tensor of shape (B, D, H, W), averaged embeddings over N clusters
        """
        # 체크: cluster_ids 타입과 범위
        assert cluster_ids.dtype == torch.long, "cluster_ids must be LongTensor"
        assert cluster_ids.max() < self.num_clusters, "cluster_ids contains out-of-bound indices"

        B, N = cluster_ids.shape

        # 1차원 인덱스로 변환
        flat_ids = cluster_ids.view(-1)  # (B * N)

        # learned_features에서 인덱싱
        selected = self.learned_features[flat_ids]  # (B * N, D, H, W)

        # 다시 원래 배치 크기로 reshape
        selected = selected.view(B, N, *self.learned_features.shape[1:])  # (B, N, D, H, W)

        # N개 클러스터 임베딩 평균
        embeddings = selected.mean(dim=1)  # (B, D, H, W)

        return embeddings


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
        # dot = rearrange(dot, 'b l n Q K -> b l Q (n K)')  # b (X Y) (W1 W2) (n w1 w2)

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
        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.feature_proj = None if no_image_features else nn.Sequential(
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
        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.skip = skip
        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

        self.register_buffer('image_plane', generate_grid(feat_height, feat_width)[None], persistent=False)
        self.image_plane[:, :, 0] *= image_width
        self.image_plane[:, :, 1] *= image_height

    def pad_divisible(self, x, win_h, win_w):
        _, _, _, h, w = x.shape
        h_pad = ((h + win_h - 1) // win_h) * win_h
        w_pad = ((w + win_w - 1) // win_w) * win_w
        padh = h_pad - h
        padw = w_pad - w
        return F.pad(x, (0, padw, 0, padh), value=0)

    def forward(
        self,
        index: int,
        x: torch.Tensor,
        bev: BEVEmbedding,
        feature: torch.Tensor,
        I_inv: torch.Tensor,
        E_inv: torch.Tensor,
        cluster_ids: torch.Tensor,
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

        if self.bev_embed_flag:
            grid = getattr(bev, f'grid{index}')[:2]
            w_embed = self.bev_embed(grid[None])
            bev_embed = w_embed - c_embed
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        key_flat = img_embed + self.feature_proj(feature_flat) if self.feature_proj else img_embed
        val_flat = self.feature_linear(feature_flat)

        # 클러스터 기반 positional embedding 적용
        if self.bev_embed_flag:
            cluster_bev = bev.get_prior(cluster_ids)
            query = query_pos + cluster_bev[:, None]
        else:
            query = x[:, None]  # (B, N, D, H, W)

        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)

        # 나머지 attention 연산 (cross_win_attend_1, 2)...
        # 아래에 계속됨
         # pad divisible
                # padding to fit window size
        key = self.pad_divisible(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisible(val, self.feat_win_size[0], self.feat_win_size[1])

        # window partition
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        # cross attention 1
        x_skip = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                           w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        query = self.cross_win_attend_1(query, key, val, skip=x_skip)
        query = rearrange(query, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_1(self.prenorm_1(query))
        x_skip = query

        # cross attention 2
        query = repeat(query, 'b x y d -> b n x y d', n=n)
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key = rearrange(key, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        x_skip = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d',
                           w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        query = self.cross_win_attend_2(query, key, val, skip=x_skip)
        query = rearrange(query, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_2(self.prenorm_2(query))
        query = self.postnorm(query)
        return rearrange(query, 'b H W d -> b d H W')


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
        self.down = lambda x: F.interpolate(x, scale_factor=scale) if scale < 1.0 else x

        cross_views = []
        layers = []
        downsample_layers = []

        for i, (feat_shape, num_layers) in enumerate(zip(backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
            cross_views.append(cva)
            layers.append(nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)]))

            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Conv2d(dim[i], dim[i] // 2, 3, padding=1, bias=False),
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
        # self.self_attn = Attention(dim[-1], **self_attn)






    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        print(f"[DEBUG] batch['image'].shape: {batch['image'].shape} → B={b}, N={n}")
    
        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()

        cluster_ids = batch['camera_cluster_ids']  # (B, N) or list
        print(f"[DEBUG] Type of cluster_ids: {type(cluster_ids)}")

        if isinstance(cluster_ids, list):
            print(f"[DEBUG] cluster_ids is a list of length {len(cluster_ids)}")
            for i, c in enumerate(cluster_ids):
                if isinstance(c, torch.Tensor):
                    print(f"[DEBUG] cluster_ids[{i}] shape: {c.shape}")
        
            # 리스트 내부가 텐서이고 각 텐서가 (N,) shape일 경우
            cluster_ids = torch.cat(cluster_ids, dim=0)  # (B * N,)
            print(f"[DEBUG] cluster_ids after torch.cat: shape = {cluster_ids.shape}")
            cluster_ids = cluster_ids.view(b, n)  # reshape to (B, N)
            print(f"[DEBUG] cluster_ids reshaped to (B, N): shape = {cluster_ids.shape}")
            cluster_ids = cluster_ids.to(dtype=torch.long, device=image.device)

        elif isinstance(cluster_ids, torch.Tensor):
            print(f"[DEBUG] cluster_ids is a tensor: shape = {cluster_ids.shape}")
            cluster_ids = cluster_ids.to(dtype=torch.long, device=image.device)

        else:
            raise TypeError(f"[ERROR] Unexpected type for cluster_ids: {type(cluster_ids)}")

        features = [self.down(f) for f in self.backbone(self.norm(image))]
        x = self.bev_embedding.get_prior(cluster_ids)  # (B, D, H, W)

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, cluster_ids)
            x = layer(x)

            if i < len(features) - 1:
                x = self.downsample_layers[i](x)

        return x






if __name__ == "__main__":
    import os
    import re
    import yaml

    def load_yaml(file):
        stream = open(file, 'r')
        loader = yaml.Loader

        float_pattern = (
            r'^('
            r'[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?'
            r'|[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)'
            r'|\.[0-9_]+(?:[eE][-+][0-9]+)?'
            r'|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*'
            r'|[-+]?\.(?:inf|Inf|INF)'
            r'|\.(?:nan|NaN|NAN)'
            r')$'
        )

        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(float_pattern, re.X),
            list(u'-+0123456789.')
        )

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
    test_q = torch.rand(1, 6, 5, 5, 5, 5, 128)
    test_k = test_v = torch.rand(1, 6, 5, 5, 6, 12, 128)
    test_q = test_q.cuda()
    test_k = test_k.cuda()
    test_v = test_v.cuda()

    # test pad divisible
    # output = block.pad_divisble(x=test_data, win_h=6, win_w=12)
    output = block(test_q, test_k, test_v)
    print(output.shape)

    # block = CrossViewSwapAttention(
    #     feat_height=28,
    #     feat_width=60,
    #     feat_dim=128,
    #     dim=128,
    #     index=0,
    #     image_height=25,
    #     image_width=25,
    #     qkv_bias=True,
    #     q_win_size=[5, 5],
    #     feat_win_size=[6, 12],
    #     heads=[4,],
    #     dim_head=[32,],
    #     qkv_bias=True,)

    image = torch.rand(1, 6, 128, 28, 60)            # b n c h w
    I_inv = torch.rand(1, 6, 3, 3)           # b n 3 3
    E_inv = torch.rand(1, 6, 4, 4)           # b n 4 4

    feature = torch.rand(1, 6, 128, 25, 25)

    x = torch.rand(1, 128, 25, 25)                     # b d H W

    # output = block(0, x, self.bev_embedding, feature, I_inv, E_inv)
    block.cuda()

    ##### EncoderSwap
    params = load_yaml('config/model/cvt_pyramid_swap.yaml')

    print(params)

    batch = {}
    batch['image'] = image
    batch['intrinsics'] = I_inv
    batch['extrinsics'] = E_inv

    out = encoder(batch)

    print(out.shape)

'''

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import math#임의로 추가

from torch import einsum
from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List

from .decoder import DecoderBlock

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)
    indices = indices[None]
    return indices

def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters
    return [
        [ 0., -sw, w/2.],
        [-sh,  0., h*offset + h/2.],
        [ 0.,   0., 1.],
    ]

class Normalize(nn.Module):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean)[None,:,None,None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None,:,None,None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std

class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_size: tuple,      # (bev_height, bev_width) 형태로 받도록 변경
        h_meters: int,
        w_meters: int,
        offset: int,
        upsample_scales: list,
        num_clusters: int = 10,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        bev_height, bev_width = bev_size
        h = bev_height // upsample_scales[0]
        w = bev_width // upsample_scales[0]
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)
        V_inv = torch.FloatTensor(V).inverse()
        for i, scale in enumerate(upsample_scales):
            hh = bev_height // scale
            ww = bev_width // scale
            grid = generate_grid(hh, ww).squeeze(0)
            grid[0] *= bev_width
            grid[1] *= bev_height
            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')
            grid = rearrange(grid, 'd (h w) -> d h w', h=hh, w=ww)
            self.register_buffer(f'grid{i}', grid, persistent=False)
        self.learned_features = nn.Parameter(sigma * torch.randn(num_clusters, dim, h, w))

    def get_prior(self, cluster_ids):#클러스터링 적용 x ->클러스터링하고 싶으면 해당 함수 수정 필요
        batch_size = cluster_ids.shape[0]  # 여기 추가

        print(f"[get_prior] cluster_ids.shape: {cluster_ids.shape}")
        print(f"[get_prior] learned_features.shape: {self.learned_features.shape}")
        print(f"[get_prior] returning shape: {(batch_size, ) + self.learned_features.shape[1:]}")

        # 기존 코드
        return self.learned_features.mean(dim=0, keepdim=True).expand(batch_size, -1, -1, -1)





class CrossWinAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, rel_pos_emb=False, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head
        self.rel_pos_emb = rel_pos_emb

        # nn.LayerNorm(dim) 을 사용하여 dim 차원 기준 norm 하도록 수정
        self.to_q = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

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
        # dot = rearrange(dot, 'b l n Q K -> b l Q (n K)')  # b (X Y) (W1 W2) (n w1 w2)

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

class CrossViewSwapAttention(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        index: int,  # 유지되지만 인덱싱 용도로만 사용하지 않음
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        q_win_size: list,         # e.g., [8, 8]
        feat_win_size: list,      # e.g., [16, 16]
        heads: int,               # ✅ 이제 int
        dim_head: int,            # ✅ 이제 int
        bev_embedding_flag: bool, # ✅ 이제 bool
        rel_pos_emb: bool = False,
        no_image_features: bool = False,
        skip: bool = True,
        norm=nn.LayerNorm,
    ):
        super().__init__()

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.feature_proj = None if no_image_features else nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed_flag = bev_embedding_flag
        if self.bev_embed_flag:
            self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.q_win_size = q_win_size
        self.feat_win_size = feat_win_size

        print("dim:", dim)
        print("heads:", heads)
        print("dim_head:", dim_head)
        print("index:", index)

        self.cross_win_attend_1 = CrossWinAttention(dim, heads, dim_head, qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip
        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

        self.register_buffer('image_plane', generate_grid(feat_height, feat_width)[None], persistent=False)
        self.image_plane[:, :, 0] *= image_width
        self.image_plane[:, :, 1] *= image_height

   

    def pad_divisible(self, x, win_h, win_w):
        if x.dim() == 5:
            _, _, _, h, w = x.shape
        elif x.dim() == 4:
            _, _, h, w = x.shape
        else:
            raise ValueError(f"Unsupported tensor shape in pad_divisible: {x.shape}")

        h_pad = math.ceil(h / win_h) * win_h
        w_pad = math.ceil(w / win_w) * win_w
        padh = h_pad - h
        padw = w_pad - w
        x_padded = F.pad(x, (0, padw, 0, padh), value=0)
        print(f"pad_divisible input shape: {x.shape} -> output shape: {x_padded.shape}")
        return x_padded





    def forward(
        self,
        index: int,
        x: torch.Tensor,
        bev: BEVEmbedding,
        feature: torch.Tensor,
        I_inv: torch.Tensor,
        E_inv: torch.Tensor,
        cluster_ids: torch.Tensor,
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
        img_embed = F.interpolate(img_embed, size=feature.shape[-2:], mode='bilinear', align_corners=False)
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)

        if self.bev_embed_flag:
            grid = getattr(bev, f'grid{index}')[:2]
            w_embed = self.bev_embed(grid[None])
            bev_embed = w_embed - c_embed
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        key_flat = img_embed + self.feature_proj(feature_flat) if self.feature_proj else img_embed
        val_flat = self.feature_linear(feature_flat)

        if self.bev_embed_flag:
            cluster_bev = bev.get_prior(cluster_ids)
            query = query_pos + cluster_bev[:, None]
        else:
            query = x[:, None]  # (B, N, D, H, W)

        # 🔧 모든 feature size를 맞춘다
        query = rearrange(query, 'b n d h w -> (b n) d h w')
        key = key_flat
        val = val_flat

        target_h = max(query.shape[-2], key.shape[-2], val.shape[-2])
        target_w = max(query.shape[-1], key.shape[-1], val.shape[-1])

        query = F.interpolate(query, size=(target_h, target_w), mode='bilinear', align_corners=False)
        key = F.interpolate(key, size=(target_h, target_w), mode='bilinear', align_corners=False)
        val = F.interpolate(val, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # 🔧 윈도우 크기만큼 나눌 수 있도록 패딩
        query = self.pad_divisible(query, self.q_win_size[0], self.q_win_size[1])
        key = self.pad_divisible(key, self.q_win_size[0], self.q_win_size[1])
        val = self.pad_divisible(val, self.q_win_size[0], self.q_win_size[1])
        
        query = rearrange(query, '(b n) d h w -> b n d h w', b=b, n=n)
        key = rearrange(key, '(b n) d h w -> b n d h w', b=b, n=n)
        val = rearrange(val, '(b n) d h w -> b n d h w', b=b, n=n)

        # 🔍 디버깅용 출력 (원하면 주석 처리 가능)
        print("query shape after pad/interp:", query.shape)
        print("key shape after pad/interp:", key.shape)
        print("val shape after pad/interp:", val.shape)

        # ===== 디버깅 출력 =====
        print("query shape after padding:", query.shape)
        print("key shape after padding:", key.shape)
        print("val shape after padding:", val.shape)
        # ======================
        
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])


        #내가 임의로 추가한 코드
        if self.skip:
            target_h, target_w = query.shape[-2], query.shape[-1]  # ✅ 이 라인 추가
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            x = self.pad_divisible(x, self.q_win_size[0], self.q_win_size[1])
            x_skip = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                               w1=self.q_win_size[0], w2=self.q_win_size[1])
        else:
            x_skip = None



        
        
        x_skip = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                           w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None

        query = self.cross_win_attend_1(query, key, val, skip=x_skip)
        query = rearrange(query, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_1(self.prenorm_1(query))
        x_skip = query

        query = repeat(query, 'b x y d -> b n x y d', n=n)
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key = rearrange(key, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        x_skip = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d',
                           w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        query = self.cross_win_attend_2(query, key, val, skip=x_skip)
        query = rearrange(query, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_2(self.prenorm_2(query))
        query = self.postnorm(query)
        return rearrange(query, 'b H W d -> b d H W')



class PyramidAxialEncoder(nn.Module):
    def __init__(self, backbone, cross_view, cross_view_swap, bev_embedding,
                 self_attn, dim: List[int], middle: List[int], scale: float = 1.0):
        super().__init__()

        self.norm = Normalize()
                     
        # backbone이 DictConfig라면 instantiate를 통해 객체화
        if isinstance(backbone, dict) or 'DictConfig' in str(type(backbone)):
            backbone = instantiate(backbone)
            
        self.backbone = backbone
        self.down = lambda x: F.interpolate(x, scale_factor=scale) if scale < 1.0 else x
        self.cross_views = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        print("Backbone output_shapes:", backbone.output_shapes)

        for i, (feat_shape, num_layers) in enumerate(zip(backbone.output_shapes, middle)):
            # Create dummy input to determine feature dimensions
            if len(feat_shape) == 3:
                dummy_input = torch.zeros(1, *feat_shape)  # Ensure batch dimension
            else:
                dummy_input = torch.zeros(feat_shape)

            _, feat_dim, H, W = self.down(dummy_input).shape

            cva = CrossViewSwapAttention(
                feat_height=H,
                feat_width=W,
                feat_dim=feat_dim,
                dim=dim[i],               # 단일 int 값
                index=i,
                image_height=feat_shape[1],
                image_width=feat_shape[2],
                qkv_bias=True,
                heads=cross_view['heads'][i],       # 리스트에서 i번째 값 전달
                dim_head=cross_view['dim_head'][i], # 리스트에서 i번째 값 전달
                q_win_size=cross_view['q_win_size'],
                feat_win_size=cross_view['feat_win_size'],
                bev_embedding_flag=cross_view['bev_embedding_flag'],
                no_image_features=cross_view.get('no_image_features', False),
                skip=True
            )

            self.cross_views.append(cva)
            self.layers.append(nn.Sequential(*[
                ResNetBottleNeck(dim[i]) for _ in range(num_layers)
            ]))

            # Downsampling for multi-level features
            if i < len(middle) - 1:
                self.downsample_layers.append(nn.Sequential(
                    nn.Conv2d(dim[i], dim[i] // 2, 3, padding=1, bias=False),
                    nn.PixelUnshuffle(2),
                    nn.Conv2d(dim[i + 1], dim[i + 1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim[i + 1], dim[i + 1], 1, bias=False),
                    nn.BatchNorm2d(dim[i + 1])
                ))

        self.bev_embedding = bev_embedding
        self.self_attn = self_attn  # Optional
        self.dim = dim
        self.middle = middle

    def forward(self, batch):
        bev_obj_count = batch.get('bev_obj_count', None)
        repeat_times = self._decide_repeats(bev_obj_count)

        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()
        cluster_ids = batch['camera_cluster_ids']
        if isinstance(cluster_ids, list):
            cluster_ids = torch.cat(cluster_ids, dim=0).view(b, n).long().to(image.device)
        else:
            cluster_ids = cluster_ids.long().to(image.device)

        # backbone에 입력 넣기 전 이미지 정규화
        images_norm = self.norm(image)

        # backbone 실행
        backbone_outputs = self.backbone(images_norm)

        # 점검용 출력 추가
        print(f"batch size (b): {b}, number of views (n): {n}")
        for i, f in enumerate(backbone_outputs):
            print(f"features[{i}] shape before view: {f.shape}")
            expected_shape = (b, n, *f.shape[1:])
            print(f"expected shape after view: {expected_shape}")
            expected_numel = b * n * f.shape[1] * f.shape[2] * f.shape[3]
            if f.numel() != expected_numel:
                print(f"Mismatch in number of elements: got {f.numel()}, expected {expected_numel}")

        # Feature extraction
        features = []
        for i, f in enumerate(backbone_outputs):
            f = f.view(b, n, *f.shape[1:])             # (B, N, C, H, W)
            f = f.flatten(0, 1)                        # (B*N, C, H, W)
            f = self.down(f)                           # (B*N, C', H', W')
            _, c, h, w = f.shape
            f = f.view(b, n, c, h, w)                  # (B, N, C', H', W')
            features.append(f)

        x = self.bev_embedding.get_prior(cluster_ids)

        # Cross-view and residual layers
        for _ in range(repeat_times):
            for i, (cva, layer) in enumerate(zip(self.cross_views, self.layers)):
                feat = features[i]
                x = cva(i, x, self.bev_embedding, feat, I_inv, E_inv, cluster_ids)
                x = layer(x)
                if i < len(self.downsample_layers):
                    x = self.downsample_layers[i](x)

        # Optional self-attention
        if self.self_attn:
            x = self.self_attn(x)

        return x


    def _decide_repeats(self, bev_obj_count):
        if bev_obj_count is None:
            return 2  #원래 1
        if bev_obj_count < 5:
            return 2  #원래 1
        elif bev_obj_count < 10:
            return 3  #원래 2
        else:
            return 4  #원래 3


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
    test_q = torch.rand(1, 6, 5, 5, 5, 5, 128)
    test_k = test_v = torch.rand(1, 6, 5, 5, 5, 5, 128)  # 윈도우 크기 맞춤
    test_q = test_q.cuda()
    test_k = test_k.cuda()
    test_v = test_v.cuda()

    output = block(test_q, test_k, test_v)
    print(output.shape)

    # test pad divisible
    # output = block.pad_divisble(x=test_data, win_h=6, win_w=12)
    output = block(test_q, test_k, test_v)
    print(output.shape)

    # block = CrossViewSwapAttention(
    #     feat_height=28,
    #     feat_width=60,
    #     feat_dim=128,
    #     dim=128,
    #     index=0,
    #     image_height=25,
    #     image_width=25,
    #     qkv_bias=True,
    #     q_win_size=[5, 5],
    #     feat_win_size=[6, 12],
    #     heads=[4,],
    #     dim_head=[32,],
    #     qkv_bias=True,)

    image = torch.rand(1, 6, 128, 28, 60)            # b n c h w
    I_inv = torch.rand(1, 6, 3, 3)           # b n 3 3
    E_inv = torch.rand(1, 6, 4, 4)           # b n 4 4

    feature = torch.rand(1, 6, 128, 25, 25)

    x = torch.rand(1, 128, 25, 25)                     # b d H W

    # output = block(0, x, self.bev_embedding, feature, I_inv, E_inv)
    block.cuda()

    ##### EncoderSwap
    params = load_yaml('config/model/cvt_pyramid_swap.yaml')

    print(params)

    batch = {}
    batch['image'] = image
    batch['intrinsics'] = I_inv
    batch['extrinsics'] = E_inv

    out = encoder(batch)

    print(out.shape)

