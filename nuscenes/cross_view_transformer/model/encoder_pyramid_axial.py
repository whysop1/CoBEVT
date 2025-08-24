import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional

# from .decoder import DecoderBlock

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)      # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)              # 3 h w
    indices = indices[None]                                             # 1 3 h w

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


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++ NEW MODULE START ++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ObjectCountAdapter(nn.Module):
    """
    Takes object_count per view and generates a gating weight.
    The gate is sigmoid-activated and scaled by 2, allowing it to suppress (0),
    pass-through (1), or amplify (2) features.
    """
    def __init__(self, in_features=1, hidden_features=32, out_features=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, object_count):
        # object_count: (b, n)
        b, n = object_count.shape
        oc = object_count.float().reshape(-1, 1)
        gate_weights = self.mlp(oc)
        gate_weights = gate_weights.reshape(b, n, -1)
        return torch.sigmoid(gate_weights) * 2.0
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++ NEW MODULE END +++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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


class CrossWinAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, rel_pos_emb=False, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)

    def forward(self, q, k, v, skip=None):
        """
        q, k, v: (b, n, num_patches_h, num_patches_w, patch_h, patch_w, d)
        return: (b, num_patches_h, num_patches_w, patch_h, patch_w, d)
        """
        assert k.shape == v.shape, "Key and Value shapes must match"
        _, _, q_height, q_width, _, _, _ = q.shape
        _, _, kv_height, kv_width, _, _, _ = k.shape
        assert q_height * q_width == kv_height * kv_width, \
            f"Query and Key must have the same number of patches. Got Q={q_height*q_width}, K={kv_height*kv_width}"

        # Flatten patches
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')

        # Project to queries, keys, values
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # Split heads
        q = rearrange(q, 'b l T (h d) -> (b h) l T d', h=self.heads)
        k = rearrange(k, 'b l T (h d) -> (b h) l T d', h=self.heads)
        v = rearrange(v, 'b l T (h d) -> (b h) l T d', h=self.heads)

        # Attention
        sim = self.scale * einsum('b l Q d, b l K d -> b l Q K', q, k)
        attn = sim.softmax(dim=-1)

        # Aggregate values
        out = einsum('b l Q K, b l K d -> b l Q d', attn, v)

        # Merge heads
        out = rearrange(out, '(b h) l T d -> b l T (h d)', h=self.heads)
        
        # Project output
        out = self.proj(out)

        # Un-flatten patches
        out = rearrange(out, 'b (x y) (n w1 w2) d -> b n x y w1 w2 d',
                        x=q_height, y=q_width, n=v.shape[2] // (k.shape[-2] * k.shape[-3] if len(k.shape) > 3 else k.shape[-2]))

        # Average across views
        out = out.mean(1)

        if skip is not None:
            out = out + skip
        return out


class CrossViewSwapAttention(nn.Module):
    def __init__(
        self, feat_height: int, feat_width: int, feat_dim: int, dim: int, index: int,
        image_height: int, image_width: int, qkv_bias: bool, q_win_size: list,
        feat_win_size: list, heads: list, dim_head: list, bev_embedding_flag: list,
        skip: bool = True, norm=nn.LayerNorm,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.index = index
        self.skip = skip
        self.q_win_size = q_win_size[index]
        self.feat_win_size = feat_win_size[index]

        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
        self.feature_proj = nn.Sequential(nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed_flag = bev_embedding_flag[index]
        if self.bev_embed_flag:
            self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)

        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def pad_divisible(self, x, win_h, win_w):
        b, n, c, h, w = x.shape
        pad_h = (win_h - h % win_h) % win_h
        pad_w = (win_w - w % win_w) % win_w
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(
        self, index: int, x: torch.FloatTensor, bev: BEVEmbedding, feature: torch.FloatTensor,
        I_inv: torch.FloatTensor, E_inv: torch.FloatTensor, gate_weights: Optional[torch.Tensor] = None,
    ):
        b, n, _, h_feat, w_feat = feature.shape
        _, _, H, W = x.shape

        pixel = self.image_plane
        c = E_inv[..., -1:]
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]
        c_embed = self.cam_embed(c_flat)

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        cam = I_inv @ pixel_flat
        cam = F.pad(cam, (0, 0, 0, 1), value=1)
        d = E_inv @ cam
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h_feat, w=w_feat)
        d_embed = self.img_embed(d_flat)
        img_embed = d_embed - c_embed
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)

        grid_name = f'grid{self.index}'
        world = getattr(bev, grid_name, None)[:2]
        if world is None:
            raise AttributeError(f"BEVEmbedding has no attribute {grid_name}")

        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])
            bev_embed = w_embed - c_embed
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        key_flat = img_embed + self.feature_proj(feature_flat)
        val_flat = self.feature_linear(feature_flat)

        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)

        query = x[:, None]
        if self.bev_embed_flag:
            query = query_pos + query
        
        if gate_weights is not None:
            val = val * gate_weights.unsqueeze(-1).unsqueeze(-1)

        key = self.pad_divisible(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisible(val, self.feat_win_size[0], self.feat_win_size[1])

        # Dimension `d` to last for partitioning
        query = rearrange(query, 'b n d h w -> b n h w d')
        key = rearrange(key, 'b n d h w -> b n h w d')
        val = rearrange(val, 'b n d h w -> b n h w d')

        # local-to-local attention
        query_win = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key_win = rearrange(key, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_win = rearrange(val, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        x_skip_win = rearrange(rearrange(x, 'b d h w -> b h w d'), 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        
        attended = self.cross_win_attend_1(query_win, key_win, val_win, skip=x_skip_win if self.skip else None)
        attended = rearrange(attended, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        
        attended = attended + self.mlp_1(self.prenorm_1(attended))
        x_skip = attended

        # local-to-global attention
        query = repeat(attended, 'b h w d -> b n h w d', n=n)
        
        num_patches_h = H // self.q_win_size[0]
        num_patches_w = W // self.q_win_size[1]
        
        query_win = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', x=num_patches_h, y=num_patches_w)

        # Partition key/val to match query's patch count
        key_grid = rearrange(key, 'b n (p1 h) (p2 w) d -> b n p1 p2 h w d', p1=num_patches_h, p2=num_patches_w)
        val_grid = rearrange(val, 'b n (p1 h) (p2 w) d -> b n p1 p2 h w d', p1=num_patches_h, p2=num_patches_w)
        
        x_skip_win = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        
        attended = self.cross_win_attend_2(query_win, key_grid, val_grid, skip=x_skip_win if self.skip else None)
        attended = rearrange(attended, 'b x y w1 w2 d -> b (x w1) (y w2) d')

        attended = attended + self.mlp_2(self.prenorm_2(attended))
        attended = self.postnorm(attended)

        return rearrange(attended, 'b H W d -> b d H W')


class PyramidAxialEncoder(nn.Module):
    def __init__(
            self, backbone, cross_view: dict, cross_view_swap: dict, bev_embedding: dict,
            dim: list, middle: List[int] = [2, 2], scale: float = 1.0, **kwargs
    ):
        super().__init__()
        self.norm = Normalize()
        self.backbone = backbone
        self.down = (lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)) if scale < 1.0 else (lambda x: x)

        cross_views, layers, downsample_layers = [], [], []

        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(1, *feat_shape[1:])).shape
            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
            cross_views.append(cva)
            layers.append(nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)]))
            if i < len(middle) - 1:
                downsample_layers.append(nn.Conv2d(dim[i], dim[i+1], kernel_size=3, stride=2, padding=1))

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        self.adapter = ObjectCountAdapter()

    def _normalize_object_count(self, object_count, b, n, device):
        if object_count is None: return None
        oc = object_count if torch.is_tensor(object_count) else torch.tensor(object_count, device=device)
        oc = oc.to(device)

        if oc.dim() == 0: return oc.view(1, 1).expand(b, n).float()
        if oc.dim() == 1:
            L = oc.numel()
            if L == b: return oc.view(b, 1).expand(b, n).float()
            if L == n: return oc.view(1, n).expand(b, n).float()
            if L == b * n: return oc.view(b, n).float()
        if oc.dim() == 2 and oc.shape == (b, n): return oc.float()

        print(f"Warning: object_count shape {oc.shape} incompatible. Using zeros.")
        return torch.zeros(b, n, device=device, dtype=torch.float32)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()
        
        object_count = batch.get('object_count', None)
        norm_obj = self._normalize_object_count(object_count, b, n, image.device)
        gate_weights = self.adapter(norm_obj) if norm_obj is not None else None

        features = [self.down(y) for y in self.backbone(self.norm(image))]
        x = repeat(self.bev_embedding.get_prior(), '... -> b ...', b=b)

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, gate_weights=gate_weights)
            x = layer(x)
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)

        return x
