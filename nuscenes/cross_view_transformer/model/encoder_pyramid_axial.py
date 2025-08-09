import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List
from .decoder import  DecoderBlock

from typing import Optional


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

'''
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
        # self.proj = nn.Linear(2 * dim, dim)

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

        #디버깅
        if object_count is not None:
            print(">> object_count(crossviewswapattention):", object_count.shape, object_count) #각 인덱스가 특정 종류(차, 트럭, 보행자)의 객체 수임
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
'''


'''
class CrossViewSwapAttention(nn.Module):
    # __init__ 메서드 및 다른 메서드들은 제공된 코드와 동일하게 유지합니다.
    # ...
    # get_attention_rounds 메서드도 그대로 유지합니다.
    # ...
    
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

        # 이 부분은 제공된 코드와 동일합니다.
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

    def pad_divisble(self, x, win_h, win_w):
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h) // win_h) * win_h, ((w + win_w) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    def get_attention_rounds(self, object_count):
        if object_count < 10:
            return 2
        elif object_count < 30:
            return 4
        else:
            return 6

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
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)
        object_count: (b,) - 각 배치의 객체 수

        Returns: (b, d, H, W)
        """
        # ... (forward 메서드의 초기 설정 부분은 동일) ...
        b, n, _, _, _ = feature.shape
        _, d, H, W = x.shape
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
        
        if index == 0: world = bev.grid0[:2]
        elif index == 1: world = bev.grid1[:2]
        elif index == 2: world = bev.grid2[:2]
        elif index == 3: world = bev.grid3[:2]

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
            initial_query = query_pos + x[:, None]
        else:
            initial_query = x[:, None]

        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)
        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])

        if object_count is not None:
            # 각 데이터에 대한 attention 라운드 횟수를 계산
            # [1, 3, 2, 1, ...] 형태의 텐서가 됨
            attention_rounds_per_item = torch.tensor([self.get_attention_rounds(oc) for oc in object_count], device=x.device)
            max_rounds = attention_rounds_per_item.max().item()

            # 최종 결과를 저장할 텐서. 초기값은 원본 x
            final_query_result = rearrange(x, 'b d H W -> b H W d')
            # 현재 라운드의 입력을 저장할 텐서
            current_query_input = initial_query

            # 최대 라운드 횟수만큼 반복
            for round_idx in range(max_rounds):
                # 이번 라운드에서 attention을 수행해야 할 데이터의 인덱스를 찾음
                # 예: round_idx가 1(두 번째 라운드)일 때, attention_rounds_per_item이 2 또는 3인 데이터
                active_indices = (attention_rounds_per_item > round_idx).nonzero(as_tuple=True)[0]

                # 처리할 데이터가 없으면 다음 라운드로
                if len(active_indices) == 0:
                    continue
                
                # Attention을 적용할 데이터(query, key, val)를 선택
                active_query = current_query_input[active_indices]
                active_key = key[active_indices]
                active_val = val[active_indices]
                active_x = x[active_indices]
                
                b_active, n_active = active_query.shape[0], active_query.shape[1]
                
                # ------------------ Attention 1: Local-to-Local ------------------
                # 윈도우 형태로 변경
                query_win = rearrange(active_query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', 
                                      w1=self.q_win_size[0], w2=self.q_win_size[1])
                key_win = rearrange(active_key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', 
                                    w1=self.feat_win_size[0], w2=self.feat_win_size[1])
                val_win = rearrange(active_val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', 
                                    w1=self.feat_win_size[0], w2=self.feat_win_size[1])

                # 첫 라운드에만 skip connection 적용
                skip1 = rearrange(active_x, 'b d (x w1) (y w2) -> b x y w1 w2 d', 
                                  w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip and round_idx == 0 else None

                # Cross-attention 수행
                query_res = self.cross_win_attend_1(query_win, key_win, val_win, skip=skip1)
                query_res = rearrange(query_res, 'b x y w1 w2 d -> b (x w1) (y w2) d')
                
                query_res = query_res + self.mlp_1(self.prenorm_1(query_res))
                x_skip = query_res # for a later skip connection

                # ------------------ Attention 2: Local-to-Global ------------------
                query_exp = repeat(query_res, 'b H W d -> b n H W d', n=n_active)
                query_exp = rearrange(query_exp, 'b n H W d -> b n d H W')
                query_win = rearrange(query_exp, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', 
                                      w1=self.q_win_size[0], w2=self.q_win_size[1])
                
                # Grid 형태로 변환
                key_grid = rearrange(key_win, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
                key_grid = rearrange(key_grid, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
                val_grid = rearrange(val_win, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
                val_grid = rearrange(val_grid, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])

                skip2 = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', 
                                  w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None

                # Cross-attention 수행
                query_res = self.cross_win_attend_2(query_win, key_grid, val_grid, skip=skip2)
                query_res = rearrange(query_res, 'b x y w1 w2 d -> b (x w1) (y w2) d')

                query_res = query_res + self.mlp_2(self.prenorm_2(query_res))

                # --- 결과 업데이트 ---
                # 최종 결과 텐서에 현재 라운드에서 계산된 데이터를 업데이트
                final_query_result[active_indices] = query_res

                # 다음 라운드의 입력으로 사용할 query 업데이트
                # b H W d -> b n d H W
                if round_idx < max_rounds - 1:
                    query_res_exp = repeat(query_res, 'b H W d -> b n H W d', n=n_active)
                    # 전체 current_query_input에서 해당 부분만 업데이트
                    current_query_input[active_indices] = rearrange(query_res_exp, 'b n H W d -> b n d H W')
            
            query = final_query_result

        else:
            # ... (기존의 object_count가 None일 때의 로직은 동일) ...
            query = initial_query
            query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
            key_w = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
            val_w = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
            skip1 = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
            query = rearrange(self.cross_win_attend_1(query, key_w, val_w, skip=skip1), 'b x y w1 w2 d -> b (x w1) (y w2) d')
            query = query + self.mlp_1(self.prenorm_1(query))
            x_skip = query
            query = repeat(query, 'b x y d -> b n x y d', n=n)
            query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
            key_g = rearrange(key_w, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
            key_g = rearrange(key_g, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
            val_g = rearrange(val_w, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
            val_g = rearrange(val_g, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
            skip2 = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
            query = rearrange(self.cross_win_attend_2(query, key_g, val_g, skip=skip2), 'b x y w1 w2 d -> b (x w1) (y w2) d')
            query = query + self.mlp_2(self.prenorm_2(query))

        query = self.postnorm(query)
        query = rearrange(query, 'b H W d -> b d H W')

        return query
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# assume CrossWinAttention, BEVEmbedding, generate_grid exist (unchanged)

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

    def get_attention_rounds(self, object_count_scalar: int):
        # 너가 말했듯이 반복 횟수는 높게 가져가도 된다고 했으니
        # 아래는 예시 규칙 — 필요하면 더 세분화하거나 scale 기반으로 조정해봐
        if object_count_scalar < 10:
            return 2
        elif object_count_scalar < 30:
            return 4
        elif object_count_scalar < 60:
            return 6
        else:
            return 8

    def forward(
        self,
        index: int,
        x: torch.FloatTensor,
        bev: 'BEVEmbedding',
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
        object_count: Optional[torch.Tensor] = None,
    ):
        """
        x: (b, d, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)
        object_count: (b,) or (b, k)  - per-sample counts or per-class counts
        """

        b, n, _, _, _ = feature.shape
        _, d, H, W = x.shape

        # --- embed camera/world/image as before ---
        pixel = self.image_plane
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]
        c_embed = self.cam_embed(c_flat)

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        cam = I_inv @ pixel_flat
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)
        d_coords = E_inv @ cam
        d_flat = rearrange(d_coords, 'b n d (h w) -> (b n) d h w', h=h, w=w)
        d_embed = self.img_embed(d_flat)
        img_embed = d_embed - c_embed
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)

        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        elif index == 3:
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
            current_query_input = query_pos + x[:, None]  # b n d H W
        else:
            current_query_input = x[:, None]  # b n d H W

        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)

        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])

        # If no object_count provided -> fall back to original fixed 2-round path
        if object_count is None:
            # --- original 2-round behavior (same as 1번) ---
            query = current_query_input  # b n d H W

            # local-to-local
            query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                              w1=self.q_win_size[0], w2=self.q_win_size[1])
            key_w = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                              w1=self.feat_win_size[0], w2=self.feat_win_size[1])
            val_w = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                              w1=self.feat_win_size[0], w2=self.feat_win_size[1])

            skip1 = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                              w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None

            query = rearrange(self.cross_win_attend_1(query, key_w, val_w, skip=skip1),
                              'b x y w1 w2 d -> b (x w1) (y w2) d')
            query = query + self.mlp_1(self.prenorm_1(query))
            x_skip = query

            query = repeat(query, 'b x y d -> b n x y d', n=n)
            query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                              w1=self.q_win_size[0], w2=self.q_win_size[1])

            key_g = rearrange(key_w, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
            key_g = rearrange(key_g, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                               w1=self.feat_win_size[0], w2=self.feat_win_size[1])
            val_g = rearrange(val_w, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
            val_g = rearrange(val_g, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                               w1=self.feat_win_size[0], w2=self.feat_win_size[1])

            skip2 = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d',
                              w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None

            query = rearrange(self.cross_win_attend_2(query, key_g, val_g, skip=skip2),
                              'b x y w1 w2 d -> b (x w1) (y w2) d')
            query = query + self.mlp_2(self.prenorm_2(query))

            query = self.postnorm(query)
            query = rearrange(query, 'b H W d -> b d H W')
            return query

        # ---------------- Adaptive path ----------------
        # Normalize object_count to per-sample scalar if user passed vector (e.g., per-class counts)
        # Expect object_count shape (b,) or (b, k)
        oc = object_count
        if oc.dim() > 1:
            oc = oc.view(b, -1).sum(dim=1)  # sum across classes to get total per sample
        else:
            oc = oc.view(b)

        # compute rounds per sample
        rounds_tensor = torch.tensor([self.get_attention_rounds(int(o.item())) for o in oc], device=x.device)
        max_rounds = int(rounds_tensor.max().item())

        # store previous results as (b, H, W, d) for easy masked blending
        prev_result = rearrange(x, 'b d H W -> b H W d')  # identity baseline

        # ensure current_query_input shape is b n d H W
        # loop over rounds, compute full-batch attention, then selectively update per-sample via mask
        for round_idx in range(max_rounds):
            # boolean mask for samples that should be updated this round
            active_mask = (rounds_tensor > round_idx).to(x.device)  # (b,)
            if active_mask.sum() == 0:
                # nobody active this round
                continue

            # Prepare query for cross_win_attend_1: b n x y w1 w2 d
            query_win = rearrange(current_query_input,
                                  'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                                  w1=self.q_win_size[0], w2=self.q_win_size[1])
            key_win = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                                w1=self.feat_win_size[0], w2=self.feat_win_size[1])
            val_win = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                                w1=self.feat_win_size[0], w2=self.feat_win_size[1])

            # only use original x as skip1 on first round, to match baseline behaviour
            skip1 = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                              w1=self.q_win_size[0], w2=self.q_win_size[1]) if (self.skip and round_idx == 0) else None

            # cross attend 1 -> returns shape b x y w1 w2 d (CrossWinAttention reduces camera dim)
            q1 = self.cross_win_attend_1(query_win, key_win, val_win, skip=skip1)
            q1 = rearrange(q1, 'b x y w1 w2 d -> b (x w1) (y w2) d')  # -> b H W d
            q1 = q1 + self.mlp_1(self.prenorm_1(q1))
            x_skip = q1  # to be used as skip for second attend

            # prepare for second cross-attend: expand q1 to cameras dimension
            q_exp = repeat(q1, 'b H W d -> b n H W d', n=n)
            q_exp = rearrange(q_exp, 'b n H W d -> b n d H W')
            q_win2 = rearrange(q_exp, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                               w1=self.q_win_size[0], w2=self.q_win_size[1])

            # convert key_win to grid form for second attend (same as original)
            key_grid = rearrange(key_win, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
            key_grid = rearrange(key_grid, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                                 w1=self.feat_win_size[0], w2=self.feat_win_size[1])
            val_grid = rearrange(val_win, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
            val_grid = rearrange(val_grid, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                                 w1=self.feat_win_size[0], w2=self.feat_win_size[1])

            skip2 = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d',
                              w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None

            q2 = self.cross_win_attend_2(q_win2, key_grid, val_grid, skip=skip2)
            q2 = rearrange(q2, 'b x y w1 w2 d -> b (x w1) (y w2) d')  # -> b H W d
            q2 = q2 + self.mlp_2(self.prenorm_2(q2))

            # Masked update: only update prev_result where active_mask is True
            mask = active_mask.view(b, 1, 1, 1).to(q2.dtype)  # (b,1,1,1)
            new_result = mask * q2 + (1.0 - mask) * prev_result
            prev_result = new_result

            # ---------- 여기서 오류가 났음: rearrange(q2, 'b H W d -> b n d H W', n=n)
            # 오류 해결: einops.repeat 사용
            q2_exp_for_update = repeat(q2, 'b H W d -> b n d H W', n=n)
            mask_bn = mask.view(b, 1, 1, 1, 1)
            current_query_input = mask_bn * q2_exp_for_update + (1.0 - mask_bn) * current_query_input

        # after rounds, apply postnorm and reshape to b d H W
        out = self.postnorm(prev_result)  # expects last dim to be feature dim
        out = rearrange(out, 'b H W d -> b d H W')
        return out









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

        # ✅ 여기서 object_count 가져오기
        object_count = batch.get('object_count', None)

        #디버깅
        if object_count is not None:
            print(">> object_count(pyramid axial encoder):", object_count.shape, object_count) #각 인덱스가 특정 종류(차, 트럭, 보행자)의 객체 수임
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
