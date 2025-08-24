import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional, Tuple


from .decoder import DecoderBlock  # (미사용이지만 원본 그대로 유지)

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
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


def safe_obj_count_tensor(
    object_count: Optional[torch.Tensor],
    b: int,
    n: int,
    device: torch.device
) -> torch.Tensor:
    """
    object_count를 (b, n) 스칼라 텐서로 안전하게 변환.
    - None이면 0으로 채움
    - 길이가 1이면 전체에 브로드캐스트
    - 길이가 b면 (b, 1)로 보고 n에 브로드캐스트
    - 길이가 n이면 (1, n)로 보고 b에 브로드캐스트
    - 길이가 b*n이면 (b,n)로 reshape (최우선)
    그 외 길이는 0으로 처리
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

    # 못 맞추면 0
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

        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3x3
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
    """
    object_count(b,n)을 기반으로 쿼리/로짓에 주입할 추가 positional signal 생성
    - scalar embed: (b,n)->(b,n,d) -> (b,n,d,H,W) 로 브로드캐스트
    - logit bias:   (b,n)->(b,n,1,1) 로 만들어 어텐션 로짓에 additive bias
    """
    def __init__(self, dim: int, bias_scale: float = 1.0):
        super().__init__()
        self.bias_scale = bias_scale
        self.scalar_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        # 로짓 바이어스는 스칼라 한 번 더 변환
        self.logit_bias = nn.Sequential(
            nn.Linear(1, 1),
        )

    def forward(
        self,
        obj_count_bn: torch.Tensor,  # (b,n)
        H: int,
        W: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 정규화: log1p 후 tanh로 안정화
        x = torch.tanh(torch.log1p(obj_count_bn.clamp(min=0.0)) * 0.5)  # (b,n)
        s = self.scalar_mlp(x.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)  # (b,n,d,1,1)
        s = s.expand(-1, -1, -1, H, W)                                    # (b,n,d,H,W)

        lb = self.logit_bias(x.unsqueeze(-1)) * self.bias_scale           # (b,n,1)
        lb = lb.unsqueeze(-1)                                             # (b,n,1,1)
        return s, lb


# -----------------------------
# Dense attention (windowed)
# -----------------------------
class Attention(nn.Module):
    def __init__(self, dim, dim_head=32, dropout=0., window_size=25):
        super().__init__()
        assert (dim % dim_head) == 0, 'dim must be divisible by dim_head'
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
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=H), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b m (h w) d -> b h w (m d)', h=h, w=w)
        out = self.to_out(out)
        return rearrange(out, 'b h w d -> b d h w')


# -----------------------------
# Sparse Cross-Window Attention with Dynamic Token Selection + OAPE
# -----------------------------
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

        # sparse control
        self.topk_ratio = topk_ratio
        self.min_topk = min_topk
        self.query_keep_ratio = query_keep_ratio
        self.min_query_keep = min_query_keep

    def add_rel_pos_emb(self, x):
        return x

    @staticmethod
    def _sparse_topk_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        logits: (B, L, Q, K)
        return mask with True for kept positions; False for pruned
        """
        K = logits.size(-1)
        if k >= K:
            return torch.ones_like(logits, dtype=torch.bool)
        topk = torch.topk(logits, k=k, dim=-1).indices  # (B,L,Q,k)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, topk, True)
        return mask

    def _dynamic_query_selector(self, q_tokens: torch.Tensor, keep: int) -> torch.Tensor:
        """
        q_tokens: (B, L, Q, D): saliency top-keep만 남기고 나머지는 0으로 (residual만 통과)
        """
        # saliency = L2 norm
        sal = q_tokens.pow(2).sum(dim=-1)  # (B,L,Q)
        Q = q_tokens.size(2)
        keep = min(max(keep, 1), Q)
        idx = torch.topk(sal, k=keep, dim=-1).indices  # (B,L,keep)

        mask = torch.zeros_like(sal, dtype=torch.bool)  # (B,L,Q)
        mask.scatter_(-1, idx, True)
        mask = mask.unsqueeze(-1)  # (B,L,Q,1)
        return q_tokens * mask  # pruned queries are zeroed (still contribute through skip/residual)

    def forward(
        self,
        q, k, v,
        skip=None,
        logit_additive_bias: Optional[torch.Tensor] = None  # (b,n,1,1) -> broadcasted later
    ):
        """
        q: (b n X Y W1 W2 d)
        k: (b n x y w1 w2 d)
        v: (b n x y w1 w2 d)
        return: (b X Y W1 W2 d)
        """
        assert k.shape == v.shape
        b, n, qH, qW, qw1, qw2, d = q.shape
        _, _, kH, kW, kw1, kw2, _ = k.shape
        assert qH * qW == kH * kW

        # flatten windows
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')  # B, L, Q, d
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')  # B, L, K, d
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')

        # project
        q = self.to_q(q)  # B,L,Q,H*Dh
        k = self.to_k(k)
        v = self.to_v(v)

        # merge heads with batch
        H = self.heads
        Dh = self.dim_head
        q = rearrange(q, 'b l q (h d) -> (b h) l q d', h=H, d=Dh)
        k = rearrange(k, 'b l k (h d) -> (b h) l k d', h=H, d=Dh)
        v = rearrange(v, 'b l k (h d) -> (b h) l k d', h=H, d=Dh)

        # dynamic query selection (keep top-T% by saliency)
        Q = q.size(2)
        keep_q = max(int(Q * self.query_keep_ratio), self.min_query_keep)
        q_pruned = self._dynamic_query_selector(q, keep=keep_q)

        # attention logits
        q_scaled = q_pruned * self.scale
        logits = torch.einsum('b l q d, b l k d -> b l q k', q_scaled, k)  # (B*,L,Q,K)

        # optional additive logit bias from OAPE: broadcast to (B*,L,Q,K)
        if logit_additive_bias is not None:
            # logit_additive_bias: (b, n, 1, 1)
            # 재배열해서 (b, 1, Q, K) 형태로 브로드캐스트 -> head 병합 B*에 맞춰 expand
            # 여기서는 view/window 축 L, Q, K에 모두 브로드캐스트
            lb = logit_additive_bias.mean(dim=1, keepdim=True)  # (b,1,1,1)
            lb = lb.repeat(1, logits.size(1), logits.size(2), logits.size(3))  # (b,L,Q,K)
            # heads가 합쳐져 있으므로 repeat에 맞게 expand
            times = logits.size(0) // b
            lb = lb.repeat(times, 1, 1, 1)
            logits = logits + lb

        # sparse top-k on keys
        K = logits.size(-1)
        topk = max(int(K * self.topk_ratio), self.min_topk)
        mask = self._sparse_topk_mask(logits, k=topk)
        logits = logits.masked_fill(~mask, float('-inf'))

        att = torch.softmax(logits, dim=-1)
        a = torch.einsum('b l q k, b l k d -> b l q d', att, v)  # (B*,L,Q,Dh)

        # reshape back, combine heads
        a = rearrange(a, '(b h) l (n w1 w2) d -> b n l w1 w2 (h d)', b=b, h=H, n=n, w1=qw1, w2=qw2)
        a = rearrange(a, 'b n (x y) w1 w2 d -> b n x y w1 w2 d', x=qH, y=qW)
        z = self.proj(a)  # (b,n,x,y,w1,w2,d)

        # reduce camera dim (평균)
        z = z.mean(1)  # (b,x,y,w1,w2,d)

        if skip is not None:
            z = z + skip
        return z


# -----------------------------
# Cross-View Swap Attention block (with OAPE + sparse/dynamic)
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
        # sparse/dynamic knobs
        topk_ratio: float = 0.25,
        min_topk: int = 32,
        query_keep_ratio: float = 0.75,
        min_query_keep: int = 64,
        oape_bias_scale: float = 1.0,
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

        # Cross-window attention with sparse/dynamic selection
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

        # OAPE
        self.oape = ObjectAwarePE(dim=dim, bias_scale=oape_bias_scale)

    @staticmethod
    def pad_divisble(x, win_h, win_w):
        """Pad x to be divisible by window size."""
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h) // win_h) * win_h, ((w + win_w) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    def forward(
        self,
        index: int,
        x: torch.FloatTensor,          # (b, d, H, W)
        bev: BEVEmbedding,
        feature: torch.FloatTensor,    # (b, n, dim_in, h, w)
        I_inv: torch.FloatTensor,      # (b, n, 3, 3)
        E_inv: torch.FloatTensor,      # (b, n, 4, 4)
        object_count: Optional[torch.Tensor] = None,
    ):
        b, n, _, _, _ = feature.shape
        _, d, H, W = x.shape
        device = x.device

        # ---- OAPE 준비: (b,n) 정규화 스칼라 -> 쿼리 bias, 로짓 bias 생성
        obj_bn = safe_obj_count_tensor(object_count, b, n, device)
        oape_query_bias, oape_logit_bias = self.oape(obj_bn, H, W)   # (b,n,d,H,W), (b,n,1,1)

        # ---- 카메라/이미지 임베딩
        pixel = self.image_plane
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]                                  # (b,n,4,1)
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]    # (b n,4,1,1)
        c_embed = self.cam_embed(c_flat)                     # (b n,d,1,1)

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')       # (1,1,3,hw)
        cam = I_inv @ pixel_flat                             # (b,n,3,hw)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)  # (b,n,4,hw)
        d_dir = E_inv @ cam                                  # (b,n,4,hw)
        d_flat = rearrange(d_dir, 'b n d (h w) -> (b n) d h w', h=h, w=w)  # (b n,4,h,w)
        d_embed = self.img_embed(d_flat)                     # (b n,d,h,w)

        img_embed = d_embed - c_embed                        # (b n,d,h,w)
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)

        # ---- BEV grid 선택
        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        else:
            world = bev.grid3[:2]

        # ---- 쿼리/키/값 만들기
        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])                  # (1,d,H,W)
            bev_embed = w_embed - c_embed                          # (b n,d,H,W) via broadcast
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)  # (b,n,d,H,W)
            # OAPE 쿼리 바이어스 주입
            query_pos = query_pos + oape_query_bias                # (b,n,d,H,W)

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')  # (b n,dim_in,h,w)
        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat) # (b n,d,h,w)
        else:
            key_flat = img_embed
        val_flat = self.feature_linear(feature_flat)               # (b n,d,h,w)

        if self.bev_embed_flag:
            query = query_pos + x[:, None]                         # (b,n,d,H,W)
        else:
            query = x[:, None]

        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)  # (b,n,d,h,w)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)  # (b,n,d,h,w)

        # pad divisible
        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])

        # ---- Local-to-Local cross-attention (sparse/dynamic + OAPE)
        query_w = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                            w1=self.q_win_size[0], w2=self.q_win_size[1])
        key_w   = rearrange(key,   'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                            w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_w   = rearrange(val,   'b n d (x w1) (y w2) -> b n x y w1 w2 d',
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

        # ---- Local-to-Global cross-attention (grid)
        x_skip = query_ll
        query_glb = repeat(query_ll, 'b x y d -> b n x y d', n=n)
        query_glb = rearrange(query_glb, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                              w1=self.q_win_size[0], w2=self.q_win_size[1])

        key_g   = rearrange(key_w, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key_g   = rearrange(key_g, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                            w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_g   = rearrange(val_w, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val_g   = rearrange(val_g, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
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
# Encoder
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

            cva = CrossViewSwapAttention(
                feat_height, feat_width, feat_dim, dim[i], i,
                **cross_view, **cross_view_swap
            )
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)])
            layers.append(layer)

            if i < len(middle) - 1:
                downsample_layers.append(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(dim[i], dim[i] // 2, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.PixelUnshuffle(2),
                            nn.Conv2d(dim[i+1], dim[i+1], 3, padding=1, bias=False),
                            nn.BatchNorm2d(dim[i+1]),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(dim[i+1], dim[i+1], 1, padding=0, bias=False),
                            nn.BatchNorm2d(dim[i+1])
                        )
                    )
                )

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        # self.self_attn = Attention(dim[-1], **self_attn)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)       # (b*n, c, h, w)
        I_inv = batch['intrinsics'].inverse()      # (b, n, 3, 3)
        E_inv = batch['extrinsics'].inverse()      # (b, n, 4, 4)

        object_count = batch.get('object_count', None)

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()         # (d, H, W)
        x = repeat(x, '... -> b ...', b=b)         # (b, d, H, W)

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features) - 1:
                x = self.downsample_layers[i](x)

        # x = self.self_attn(x)
        return x


# -----------------------------
# Local test (optional)
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    # quick shape smoke test
    B, N = 2, 6
    d_in = 128
    Hq, Wq = 50, 50
    hf, wf = 28, 60

    # fake inputs
    image = torch.rand(B, N, d_in, hf, wf)
    I_inv = torch.eye(3).view(1,1,3,3).repeat(B, N, 1, 1)
    E_inv = torch.eye(4).view(1,1,4,4).repeat(B, N, 1, 1)
    feature = torch.rand(B, N, d_in, Hq, Wq)
    x = torch.rand(B, d_in, Hq, Wq)
    object_count = torch.tensor([3, 1, 5, 0, 2, 4]).repeat(B)  # (B*N, ) 예시

    # CrossWinAttention quick test
    cwa = CrossWinAttention(dim=128, heads=4, dim_head=32, qkv_bias=True)
    cwa.cuda()
    test_q = torch.rand(1, 6, 5, 5, 5, 5, 128).cuda()
    test_k = test_v = torch.rand(1, 6, 5, 5, 6, 12, 128).cuda()
    out = cwa(test_q, test_k, test_v)
    print('CrossWinAttention out:', out.shape)

    # params = load_yaml('config/model/cvt_pyramid_swap.yaml')
    # print(params)

    # 주 실행 파이프라인은 실제 프로젝트에서 Encoder를 초기화한 뒤 사용하세요.
