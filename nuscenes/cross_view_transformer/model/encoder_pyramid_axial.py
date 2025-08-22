import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import math
import copy

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional

# Deformable Attention에 필요한 추가 import
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# =================================================================================================
# Deformable Attention 모듈 및 헬퍼 함수 추가 시작
# 출처: https://github.com/fundamentalvision/Deformable-DETR
# =================================================================================================

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# This is a dummy ms_deform_attn_core_pytorch function for compatibility.
# It is recommended to install and compile the CUDA version for performance.
def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                        mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

# =================================================================================================
# Deformable Attention 모듈 및 헬퍼 함수 추가 끝
# =================================================================================================


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
        [ 0., -sw,        w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,          1.]
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
            sigma: float,
            bev_height: int,
            bev_width: int,
            h_meters: float,
            w_meters: float,
            offset: float,
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
            sigma * torch.randn(dim, bev_height // upsample_scales[0], bev_width // upsample_scales[0]))

    def get_prior(self):
        return self.learned_features


class CrossWinAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.proj = nn.Linear(heads * dim_head, dim)

    def forward(self, q, k, v, skip=None):
        _, _, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q, k, v = map(lambda t: rearrange(t, 'b ... (m d) -> (b m) ... d', m=self.heads), (q, k, v))
        dot = self.scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)
        att = dot.softmax(dim=-1)
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads)
        a = rearrange(a, 'b (x y) (n w1 w2) d -> b n x y w1 w2 d',
                      x=q_height, y=q_width, w1=q_win_height, w2=q_win_width)
        z = self.proj(a)
        z = z.mean(1)
        if skip is not None:
            z = z + skip
        return z


class CrossViewSwapAttention(nn.Module):
    def __init__(
        self, feat_height, feat_width, feat_dim, dim, index, image_height, image_width,
        qkv_bias, q_win_size, feat_win_size, heads, dim_head, bev_embedding_flag,
        no_image_features=False, skip=True, norm=nn.LayerNorm,
        **kwargs
    ):
        super().__init__()
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)
        self.feature_linear = nn.Sequential(nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
        self.feature_proj = None if no_image_features else nn.Sequential(nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
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

    def pad_divisible(self, x, win_h, win_w):
        _, _, _, h, w = x.shape
        padh = (win_h - h % win_h) % win_h
        padw = (win_w - w % win_w) % win_w
        return F.pad(x, (0, padw, 0, padh))

    def forward(self, index, x, bev, feature, I_inv, E_inv, object_count=None):
        b, n, _, _, _ = feature.shape
        pixel = self.image_plane
        _, _, _, h, w = pixel.shape
        c = E_inv[..., -1:]
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]
        c_embed = self.cam_embed(c_flat)
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        cam = I_inv @ pixel_flat
        cam = F.pad(cam, (0, 0, 0, 1), value=1)
        d = E_inv @ cam
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)
        d_embed = self.img_embed(d_flat)
        img_embed = d_embed - c_embed
        img_embed = F.normalize(img_embed, dim=1)
        
        world = getattr(bev, f'grid{index}')[:2]

        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])
            bev_embed = w_embed - c_embed
            bev_embed = F.normalize(bev_embed, dim=1)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b)
        
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        key_flat = img_embed + (self.feature_proj(feature_flat) if self.feature_proj else 0)
        val_flat = self.feature_linear(feature_flat)
        
        query = (query_pos if self.bev_embed_flag else 0) + x[:, None]
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b)
        
        key = self.pad_divisible(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisible(val, self.feat_win_size[0], self.feat_win_size[1])
        
        query_w = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key_w = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_w = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        
        x_skip_w = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        
        query_out_w = self.cross_win_attend_1(query_w, key_w, val_w, skip=x_skip_w if self.skip else None)
        query = rearrange(query_out_w, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_1(self.prenorm_1(query))
        
        x_skip = query
        query_g = repeat(query, 'b x y d -> b n x y d', n=n)
        
        query_g_w = rearrange(query_g, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key_g = rearrange(key_w, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key_g_w = rearrange(key_g, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_g = rearrange(val_w, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val_g_w = rearrange(val_g, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        
        x_skip_g_w = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])

        query_out_g_w = self.cross_win_attend_2(query_g_w, key_g_w, val_g_w, skip=x_skip_g_w if self.skip else None)
        query = rearrange(query_out_g_w, 'b x y w1 w2 d -> b (x w1) (y w2) d')
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
            self_attn: dict, # ERROR FIX: Changed 'self_attn_deformable' back to 'self_attn' to match config file
            dim: list,
            middle: List[int] = (2, 2),
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        self.down = (lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)) if scale < 1.0 else (lambda x: x)

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = []
        layers = []
        downsample_layers = []

        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(1, *feat_shape[1:])).shape
            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
            cross_views.append(cva)
            layers.append(nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)]))
            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Conv2d(dim[i], dim[i] // 2, 3, 1, 1, bias=False),
                    nn.PixelUnshuffle(2),
                    nn.Conv2d(dim[i+1], dim[i+1], 3, 1, 1, bias=False),
                    nn.BatchNorm2d(dim[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim[i+1], dim[i+1], 1, 0, 0, bias=False),
                    nn.BatchNorm2d(dim[i+1])
                ))

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)

        # Deformable Self-Attention for BEV Feature Refinement
        # ERROR FIX: Using 'self_attn' dictionary from the config to initialize the module
        self.self_attn_deformable = MSDeformAttn(d_model=dim[-1], **self_attn)

    def get_reference_points(self, spatial_shapes, device):
        reference_points_list = []
        for H, W in spatial_shapes:
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, device=device),
                                          torch.linspace(0.5, W - 0.5, W, device=device), indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()
        object_count = batch.get('object_count')
        
        features = [self.down(y) for y in self.backbone(self.norm(image))]
        x = repeat(self.bev_embedding.get_prior(), '... -> b ...', b=b)

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features) - 1:
                x = self.downsample_layers[i](x)

        # Deformable Self-Attention Application
        b, c, h, w = x.shape
        spatial_shapes = x.new_tensor([(h, w)], dtype=torch.long)
        level_start_index = x.new_tensor([0], dtype=torch.long)
        
        query = rearrange(x, 'b c h w -> b (h w) c')
        reference_points = self.get_reference_points(spatial_shapes, device=x.device).repeat(b, 1, 1, 1)
        
        x = self.self_attn_deformable(query, reference_points, query, spatial_shapes, level_start_index)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x
