'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List


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
        [0., -sw, w/2.],
        [-sh, 0., h*offset + h/2.],
        [0., 0., 1.]
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
        decoder_blocks: list,
        num_clusters: int = 10
    ):
        super().__init__()

        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))

        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)
        V_inv = torch.FloatTensor(V).inverse()
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)

        self.register_buffer('grid', grid, persistent=False)

        self.num_clusters = num_clusters
        self.embeddings = nn.Parameter(
            sigma * torch.randn(num_clusters, dim, h, w)
        )

    def get_prior(self, cluster_ids: torch.Tensor):
        """
        cluster_ids: Tensor of shape (B, N) with values in [0, num_clusters)
        Returns: Tensor of shape (B, N, D, H, W)
        """
        b, n = cluster_ids.shape
        device = cluster_ids.device
        out = []

        for i in range(b):
            per_sample = self.embeddings[cluster_ids[i]]  # (N, D, H, W)
            out.append(per_sample)

        return torch.stack(out, dim=0)  # (B, N, D, H, W)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        _, _, _, H, W = q.shape

        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        z = self.proj(a)

        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z


class CrossViewAttention(nn.Module):
    def __init__(
        self,
        feat_height, feat_width, feat_dim, dim,
        image_height, image_width,
        qkv_bias, heads=4, dim_head=32,
        no_image_features=False, skip=True
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

        self.feature_proj = None if no_image_features else nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(self, x, bev: BEVEmbedding, feature, I_inv, E_inv, cluster_ids):
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

        w_embed = self.bev_embed(bev.grid[:2][None])
        bev_embed = w_embed - c_embed
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)

        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')

        key_flat = img_embed + self.feature_proj(feature_flat) if self.feature_proj else img_embed
        val_flat = self.feature_linear(feature_flat)

        query = query_pos + bev.get_prior(cluster_ids)  # shape: (B, N, D, H, W)
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)

        return self.cross_attend(query, key, val, skip=x if self.skip else None)


class Encoder(nn.Module):
    def __init__(self, backbone, cross_view: dict, bev_embedding: dict,
                 dim=128, middle: List[int] = [2, 2], scale=1.0):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone
        self.down = lambda x: F.interpolate(x, scale_factor=scale) if scale < 1.0 else x

        cross_views = []
        layers = []

        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)
            layers.append(nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)]))

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()
        cluster_ids = batch['camera_cluster_ids']  # (B, N)

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior(cluster_ids).mean(1)  # average across N cameras for init
        # shape: (B, D, H, W)

        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv, cluster_ids)
            x = layer(x)

        return x


'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List


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
        [0., -sw, w/2.],
        [-sh, 0., h*offset + h/2.],
        [0., 0., 1.]
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
        decoder_blocks: list,
        num_clusters: int = 10
    ):
        super().__init__()

        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))

        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)
        V_inv = torch.FloatTensor(V).inverse()
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)

        self.register_buffer('grid', grid, persistent=False)

        self.num_clusters = num_clusters
        self.embeddings = nn.Parameter(
            sigma * torch.randn(num_clusters, dim, h, w)
        )

    def get_prior(self, cluster_ids: torch.Tensor):
        """
        cluster_ids: Tensor of shape (B, N) with values in [0, num_clusters)
        Returns: Tensor of shape (B, N, D, H, W)
        """
        b, n = cluster_ids.shape
        device = cluster_ids.device
        out = []

        for i in range(b):
            per_sample = self.embeddings[cluster_ids[i]]  # (N, D, H, W)
            out.append(per_sample)

        return torch.stack(out, dim=0)  # (B, N, D, H, W)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        _, _, _, H, W = q.shape

        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        z = self.proj(a)

        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z


class CrossViewAttention(nn.Module):
    def __init__(
        self,
        feat_height, feat_width, feat_dim, dim,
        image_height, image_width,
        qkv_bias, heads=4, dim_head=32,
        no_image_features=False, skip=True
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

        self.feature_proj = None if no_image_features else nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(self, x, bev: BEVEmbedding, feature, I_inv, E_inv, cluster_ids):
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

        w_embed = self.bev_embed(bev.grid[:2][None])
        bev_embed = w_embed - c_embed
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)

        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')

        key_flat = img_embed + self.feature_proj(feature_flat) if self.feature_proj else img_embed
        key = rearrange(key_flat, '(b n) d h w -> b n d h w', b=b, n=n)
        value = rearrange(feature_flat, '(b n) d h w -> b n d h w', b=b, n=n)

        x = self.cross_attend(query_pos, key, value, skip=x if self.skip else None)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        dim=128,
        bev_height=200,
        bev_width=200,
        h_meters=100,
        w_meters=100,
        offset=0,
        backbone=None,
        cross_views=None,
        layers=None,
        bev_embedding=None,
    ):
        super().__init__()

        self.bev_embedding = bev_embedding
        self.backbone = backbone
        self.cross_views = cross_views
        self.layers = layers

    def forward(self, batch, bev_obj_count=None):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()
        cluster_ids = batch['camera_cluster_ids']

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior(cluster_ids).mean(1)

        # 반복 횟수 결정 (bev_obj_count에 따라)
        repeat_times = 3
        if bev_obj_count is not None:
            if bev_obj_count < 5:
                repeat_times = 2
            elif bev_obj_count < 10:
                repeat_times = 3
            else:
                repeat_times = 4

        for _ in range(repeat_times):
            for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
                feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
                x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv, cluster_ids)
                x = layer(x)

        return x


# 추가적으로 필요한 norm과 down 샘플링 정의 (예시)
class NormLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, D, H, W)
        # rearrange -> (B, H*W, D)
        b, d, h, w = x.shape
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# 아래는 Encoder 초기화 예시 (사용자 환경에 맞게 변경 필요)
def build_encoder():
    # 예시: backbone, cross_views, layers, bev_embedding을 사용자 환경에 맞게 초기화
    backbone = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        ),
        # 추가 레이어들
    ])

    cross_views = nn.ModuleList([
        CrossViewAttention(
            feat_height=25, feat_width=25, feat_dim=64, dim=128,
            image_height=200, image_width=200,
            qkv_bias=True, heads=4, dim_head=32
        )
        # 더 필요한 만큼 추가
    ])

    layers = nn.ModuleList([
        NormLayer(128),
        # 더 필요한 만큼 추가
    ])

    bev_embedding = BEVEmbedding(
        dim=128,
        sigma=0.01,
        bev_height=200,
        bev_width=200,
        h_meters=100,
        w_meters=100,
        offset=0,
        decoder_blocks=[2],  # 예시
        num_clusters=10
    )

    encoder = Encoder(
        dim=128,
        bev_height=200,
        bev_width=200,
        h_meters=100,
        w_meters=100,
        offset=0,
        backbone=backbone,
        cross_views=cross_views,
        layers=layers,
        bev_embedding=bev_embedding
    )
    encoder.norm = Normalize()
    encoder.down = DownsampleLayer(64, 128)
    return encoder

