import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List, Optional, Tuple

# ------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------

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
    Ego(x,y,1) -> BEV(pixel)  (row=y, col=x)
    """
    sh = h / h_meters
    sw = w / w_meters
    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


def ego_to_bev_xy(xy: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    xy: (..., 2) in ego meters
    V : (3,3) ego->bev matrix
    return: (..., 2) bev pixel coords (x=col, y=row)
    """
    ones = torch.ones_like(xy[..., :1])
    homo = torch.cat([xy, ones], dim=-1)                      # (...,3)
    pix = homo @ V.T                                          # (...,3)
    pix = pix[..., :2] / pix[..., 2:].clamp(min=1e-7)         # (...,2)
    return pix


def normalize_grid_uv(uv: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    uv: (..., 2) in pixel coords (u=x, v=y) for feature map of size (h,w)
    return: (..., 2) normalized to [-1,1] for grid_sample (x,y)
    """
    x = (uv[..., 0] / (w - 1)) * 2 - 1
    y = (uv[..., 1] / (h - 1)) * 2 - 1
    return torch.stack([x, y], dim=-1)


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


# ------------------------------------------------------------------------------------
# BEV Embedding (store V and V_inv)
# ------------------------------------------------------------------------------------

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

        V = torch.FloatTensor(get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset))  # 3x3 ego->bev
        V_inv = V.inverse()

        self.register_buffer('V', V, persistent=False)
        self.register_buffer('V_inv', V_inv, persistent=False)

        for i, scale in enumerate(upsample_scales):
            h = bev_height // scale
            w = bev_width // scale
            grid = generate_grid(h, w).squeeze(0)
            grid[0] = bev_width * grid[0]
            grid[1] = bev_height * grid[1]
            # bev(pixel) -> ego(meters)
            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')   # 3 (h w)
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w) # 3 h w
            self.register_buffer('grid%d'%i, grid, persistent=False)

        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, bev_height//upsample_scales[0], bev_width//upsample_scales[0])
        )

    def get_prior(self):
        return self.learned_features


# ------------------------------------------------------------------------------------
# Depth-aware head + Lift-Splat (ground plane)
# ------------------------------------------------------------------------------------

class DepthHead(nn.Module):
    def __init__(self, in_dim: int, n_bins: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, n_bins, 1, bias=True)
        )
        self.n_bins = n_bins

    def forward(self, x):
        # x: (b*n, C, h, w)
        logits = self.head(x)                                 # (b*n, D, h, w)
        prob = logits.softmax(dim=1)
        return prob


class LiftSplatBEV(nn.Module):
    """
    Lift camera features to 3D along rays and splat onto ground plane (Z=0) BEV.
    """
    def __init__(self, V: torch.Tensor, bev_h: int, bev_w: int):
        super().__init__()
        # Use BEVEmbedding's V to guarantee identical mapping
        self.register_buffer('V', V, persistent=False)
        self.bev_h = bev_h
        self.bev_w = bev_w

    @staticmethod
    def _ray_to_ground_intersection(c_xyz: torch.Tensor, d_xyz: torch.Tensor) -> torch.Tensor:
        """
        c_xyz: (..., 3) camera center in ego
        d_xyz: (..., 3) ray direction in ego (not normalized ok)
        returns XY at Z=0
        """
        dz = d_xyz[..., 2].clone()
        dz[dz.abs() < 1e-6] = 1e-6
        s = (-c_xyz[..., 2]) / dz
        xy = c_xyz[..., :2] + d_xyz[..., :2] * s[..., None]
        return xy

    def forward(
        self,
        feat_bn: torch.Tensor,          # (b*n, C, h, w)
        depth_prob: torch.Tensor,       # (b*n, D, h, w)
        I_inv: torch.Tensor,            # (b, n, 3, 3)
        E_inv: torch.Tensor,            # (b, n, 4, 4)
        n: int
    ):
        device = feat_bn.device
        b_n, C, h, w = feat_bn.shape
        b = b_n // n

        # Build pixel grid on feature map
        xs = torch.linspace(0, w - 1, w, device=device)
        ys = torch.linspace(0, h - 1, h, device=device)
        u, v = torch.meshgrid(xs, ys, indexing='xy')           # w, h
        pix = torch.stack([u, v, torch.ones_like(u)], dim=0)   # 3, w, h
        pix = rearrange(pix, 'd w h -> d (h w)')               # 3, HW

        # project rays to ego using inverse intrinsics/extrinsics
        I_inv_bn = rearrange(I_inv, 'b n ... -> (b n) ...', b=b, n=n)  # (b*n, 3,3)
        E_inv_bn = rearrange(E_inv, 'b n ... -> (b n) ...', b=b, n=n)  # (b*n, 4,4)

        cam = I_inv_bn @ pix                                           # (b*n, 3, HW)
        cam = F.pad(cam, (0, 0, 0, 1), value=1.0)                      # (b*n, 4, HW)
        d = E_inv_bn @ cam                                             # (b*n, 4, HW)
        d = rearrange(d, 'bn d (hw) -> bn (hw) d', hw=h*w)[..., :3]    # (b*n, HW, 3)

        c = E_inv_bn[..., :3, 3]                                       # (b*n, 3)
        c = c[:, None, :].expand(-1, h*w, -1)                          # (b*n, HW, 3)

        # Depth confidence (max over bins as a soft confidence)
        conf = depth_prob.max(dim=1).values                             # (b*n, h, w)
        conf = rearrange(conf, 'bn h w -> bn (h w)')                    # (b*n, HW)

        # Intersect ray with ground plane Z=0
        xy_ego = self._ray_to_ground_intersection(c, d)                # (b*n, HW, 2)

        # ego->bev pixel
        bev_xy = ego_to_bev_xy(xy_ego, self.V)                         # (b*n, HW, 2)

        # Normalize to pixel coords
        px = bev_xy[..., 0].clamp(0, self.bev_w - 1e-4)
        py = bev_xy[..., 1].clamp(0, self.bev_h - 1e-4)

        x0 = px.floor()
        y0 = py.floor()
        x1 = (x0 + 1).clamp(max=self.bev_w - 1)
        y1 = (y0 + 1).clamp(max=self.bev_h - 1)

        wa = (x1 - px) * (y1 - py)
        wb = (px - x0) * (y1 - py)
        wc = (x1 - px) * (py - y0)
        wd = (px - x0) * (py - y0)

        feat_flat = rearrange(feat_bn, 'bn c h w -> bn c (h w)')        # (b*n, C, HW)
        bev = torch.zeros(b*n, C, self.bev_h, self.bev_w, device=device)

        def add_weighted(ix, iy, wgt):
            idx = (iy.long() * self.bev_w + ix.long())                  # (b*n, HW)
            w = (wgt * conf).unsqueeze(1)                               # (b*n, 1, HW)
            wfeat = feat_flat * w                                       # (b*n, C, HW)
            bev.view(b*n, C, -1).scatter_add_(2, idx[:, None, :], wfeat)

        add_weighted(x0, y0, wa)
        add_weighted(x1, y0, wb)
        add_weighted(x0, y1, wc)
        add_weighted(x1, y1, wd)

        # merge cameras: mean over n
        bev = rearrange(bev, '(b n) c h w -> b n c h w', b=b, n=n).mean(dim=1)  # (b, C, H, W)
        return bev


# ------------------------------------------------------------------------------------
# Deformable Cross Attention (BEV query -> image features)
# ------------------------------------------------------------------------------------

class DeformableCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32, n_points: int = 8, qkv_bias: bool = True):
        super().__init__()
        assert dim % dim_head == 0
        self.heads = heads
        self.dim_head = dim_head
        self.n_points = n_points
        self.scale = dim_head ** -0.5

        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=qkv_bias)
        self.to_kv = nn.Conv2d(dim, 2 * inner, 1, bias=qkv_bias)
        self.offset_mlp = nn.Sequential(
            nn.Conv2d(dim, inner, 1), nn.ReLU(inplace=True),
            nn.Conv2d(inner, heads * n_points * 2, 1)
        )
        # attention weights from similarity only (stable). If 필요하면 아래 주석 해제 후 사용
        # self.attn_mlp = nn.Sequential(nn.Conv2d(dim, inner, 1), nn.ReLU(inplace=True), nn.Conv2d(inner, heads * n_points, 1))

        self.proj = nn.Linear(inner, dim)

    @staticmethod
    def _project_bev_to_image(world_xy: torch.Tensor, K: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """
        world_xy: (b, n, 2, H, W) in ego meters on Z=0
        K: (b, n, 3, 3)
        E: (b, n, 4, 4)  (ego->cam)
        return uv pixels: (b, n, 2, H, W)
        """
        b, n, _, H, W = world_xy.shape
        ones = torch.ones(b, n, 1, H, W, device=world_xy.device)
        xyz1 = torch.cat([world_xy, torch.zeros_like(ones), ones], dim=2)   # (b,n,4,H,W), Z=0
        xyz1 = rearrange(xyz1, 'b n d h w -> b n h w d')
        xyz1 = rearrange(xyz1, 'b n h w d -> (b n) (h w) d')

        E_bn = rearrange(E, 'b n ... -> (b n) ...')
        K_bn = rearrange(K, 'b n ... -> (b n) ...')

        cam = (E_bn @ xyz1.transpose(1, 2)).transpose(1, 2)                # (b*n, HW, 4)
        cam = cam[..., :3]                                                 # (b*n, HW, 3)
        pix = (K_bn @ cam.transpose(1, 2)).transpose(1, 2)                 # (b*n, HW, 3)

        uv = pix[..., :2] / pix[..., 2:].clamp(min=1e-6)                   # (b*n, HW, 2)
        uv = rearrange(uv, '(b n) (h w) c -> b n c h w', b=b, n=n, h=H, w=W)
        return uv

    def forward(
        self,
        bev: torch.Tensor,                  # (b, d, H, W)  - query
        img_feats: torch.Tensor,            # (b, n, d, h, w)
        K: torch.Tensor,                    # (b, n, 3, 3)
        E: torch.Tensor,                    # (b, n, 4, 4) (ego->cam)
        world_xy: torch.Tensor              # (2, H, W) ego grid from BEVEmbedding.grid{idx}[:2]
    ):
        b, d, H, W = bev.shape
        n = img_feats.shape[1]
        h, w = img_feats.shape[-2:]  # FIX: was '_, _, h, w = img_feats.shape[-3:]'

        # project BEV cell centers to each camera feature plane
        world_xy_bn = world_xy[None, None, ...].expand(b, n, -1, -1, -1)   # (b,n,2,H,W)
        uv = self._project_bev_to_image(world_xy_bn, K, E)                 # (b,n,2,H,W)
        base_grid = normalize_grid_uv(rearrange(uv, 'b n c h w -> b n h w c'), h, w)  # (b,n,H,W,2)

        # prepare q
        inner = self.heads * self.dim_head
        q = rearrange(self.to_q(rearrange(bev, 'b d H W -> b (H W) d')), 'b hw (m d1) -> b m d1 hw', m=self.heads, d1=self.dim_head)

        # prepare k,v (split per head for clean sampling)
        kv = rearrange(img_feats, 'b n d h w -> (b n) d h w')
        kv = self.to_kv(kv)                                                # (b*n, 2*inner, h, w)
        k, v = torch.chunk(kv, 2, dim=1)                                   # (b*n, inner, h, w)
        k = rearrange(k, '(b n) (m d1) h w -> (b n m) d1 h w', b=b, n=n, m=self.heads, d1=self.dim_head)
        v = rearrange(v, '(b n) (m d1) h w -> (b n m) d1 h w', b=b, n=n, m=self.heads, d1=self.dim_head)

        # offsets per head & per sample point (predict from kv)
        off = self.offset_mlp(rearrange(img_feats, 'b n d h w -> (b n) d h w'))   # (b*n, m*p*2, h, w)
        off = rearrange(off, '(b n) (m p c) h w -> b n m p h w c', b=b, n=n, m=self.heads, p=self.n_points, c=2)
        off[..., 0] = off[..., 0] / max(w - 1, 1) * 2
        off[..., 1] = off[..., 1] / max(h - 1, 1) * 2

        # per-head grids with offsets
        grid_rep = base_grid[:, :, None, None, ...].expand(-1, -1, self.heads, self.n_points, -1, -1, -1)  # b n m p H W 2
        samp_grid = (grid_rep + off).clamp(-1, 1)                           # b n m p H W 2

        # sample helper: per head / per p -> use batch trick for grid_sample
        def sample(feat_head):  # (b n m, d1, h, w)
            g = rearrange(samp_grid, 'b n m p H W c -> (b n m p) H W c')
            feat_rep = feat_head.repeat_interleave(self.n_points, dim=0)     # (b n m p, d1, h, w)
            out = F.grid_sample(feat_rep, g, align_corners=True)             # (b n m p, d1, H, W)
            out = rearrange(out, '(b n m p) d1 H W -> b n m p d1 H W', b=b, n=n, m=self.heads, p=self.n_points)
            return out

        k_s = sample(k)   # (b, n, m, p, d1, H, W)
        v_s = sample(v)   # (b, n, m, p, d1, H, W)

        # attention: dot(q,k) per head over d1
        qh = q * self.scale                                         # (b, m, d1, HW)
        k_s = rearrange(k_s, 'b n m p d1 H W -> b n m p d1 (H W)')
        qh = rearrange(qh, 'b m d1 (H W) -> b 1 m d1 (H W)', H=H, W=W)
        sim = (qh * k_s).sum(dim=3)                                 # (b, n, m, p, HW)
        att = sim.softmax(dim=3)                                    # softmax over p

        # weighted sum of v
        v_s = rearrange(v_s, 'b n m p d1 H W -> b n m p d1 (H W)')
        out = (att.unsqueeze(3) * v_s).sum(dim=3)                   # (b, n, m, d1, HW)
        out = out.mean(dim=1)                                       # average over cameras -> (b, m, d1, HW)

        out = rearrange(out, 'b m d1 (H W) -> b (m d1) H W', H=H, W=W)
        out = self.proj(out)                                        # (b, d, H, W)
        return out


# ------------------------------------------------------------------------------------
# CrossView block (Deformable + Depth-aware + MLP)
# ------------------------------------------------------------------------------------

class CrossViewDeformableBlock(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        index: int,
        image_height: int,      # unused but kept for compatibility
        image_width: int,       # unused but kept for compatibility
        qkv_bias: bool,
        q_win_size: list,       # unused
        feat_win_size: list,    # unused
        heads: list,
        dim_head: list,
        bev_embedding_flag: list,
        rel_pos_emb: bool = False,
        no_image_features: bool = False,
        skip: bool = True,
        n_points: int = 8,
        depth_bins: int = 64,
        norm=nn.LayerNorm,
    ):
        super().__init__()

        self.index = index
        self.feat_h = feat_height
        self.feat_w = feat_width

        # feature projection
        self.img_proj = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, dim, 1, bias=False)
        )

        self.deform_attn = DeformableCrossAttention(dim, heads[index], dim_head[index], n_points, qkv_bias)
        self.skip = skip

        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.postnorm = norm(dim)

        # Depth-aware lift-splat
        self.depth_head = DepthHead(feat_dim, n_bins=depth_bins)
        self.lift_splat = None  # set in build() after we know bev size via BEVEmbedding

        # gating to fuse deformable output and depth-lifted BEV
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def build_lift(self, bev: BEVEmbedding):
        # One-time setup with BEV sizes inferred from learned_features
        H, W = bev.learned_features.shape[-2:]
        self.lift_splat = LiftSplatBEV(bev.V, H, W)

    def forward(
        self,
        index: int,
        x: torch.Tensor,                 # (b, d, H, W) current BEV
        bev: BEVEmbedding,
        feature: torch.Tensor,           # (b, n, dim_in, h, w)
        K: torch.Tensor,                 # (b, n, 3, 3)  (intrinsics)
        E: torch.Tensor,                 # (b, n, 4, 4)  (ego->cam)
        I_inv: torch.Tensor,             # (b, n, 3, 3)
        E_inv: torch.Tensor,             # (b, n, 4, 4)
        object_count: Optional[torch.Tensor] = None,
    ):
        b, n, _, h, w = feature.shape
        _, d, H, W = x.shape

        if self.lift_splat is None:
            self.build_lift(bev)

        # world grid (ego meters) from embedding
        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        else:
            world = bev.grid3[:2]

        # image feature proj
        img_f = rearrange(feature, 'b n c h w -> (b n) c h w')
        img_f_proj = self.img_proj(img_f)                                 # (b*n, d, h, w)
        img_f_proj_bn = rearrange(img_f_proj, '(b n) d h w -> b n d h w', b=b, n=n)

        # deformable cross-attention (BEV query -> image)
        x_skip = x
        x = self.deform_attn(x, img_f_proj_bn, K, E, world) + (x_skip if self.skip else 0)
        x = rearrange(x, 'b d H W -> b (H W) d')
        x = x + self.mlp(self.prenorm(x))
        x = self.postnorm(x)
        x = rearrange(x, 'b (H W) d -> b d H W', H=H, W=W)

        # depth-aware lift-splat and fuse
        depth_prob = self.depth_head(img_f)                                # (b*n, D, h, w)
        lift_bev = self.lift_splat(img_f_proj, depth_prob, I_inv, E_inv, n)  # (b, d, H, W)

        x = self.fuse_gate(torch.cat([x, lift_bev], dim=1))
        return x


# ------------------------------------------------------------------------------------
# Temporal Fusion (ConvGRU + optional warping)
# ------------------------------------------------------------------------------------

class ConvGRUCell(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.reset = nn.Conv2d(dim*2, dim, kernel_size, padding=pad)
        self.update = nn.Conv2d(dim*2, dim, kernel_size, padding=pad)
        self.out = nn.Conv2d(dim*2, dim, kernel_size, padding=pad)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros_like(x)
        inp = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.reset(inp))
        z = torch.sigmoid(self.update(inp))
        n = torch.tanh(self.out(torch.cat([x, r*h], dim=1)))
        h_new = (1 - z) * n + z * h
        return h_new


class TemporalBEVGRU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.cell = ConvGRUCell(dim)

    @staticmethod
    def warp_bev(x: torch.Tensor, A: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: (b, d, H, W), A: (b, 2, 3) affine (prev->cur) in BEV pixel coords
        """
        if A is None:
            return x
        b, d, H, W = x.shape
        grid = F.affine_grid(A, size=(b, d, H, W), align_corners=True)
        return F.grid_sample(x, grid, align_corners=True)

    def forward(self, x: torch.Tensor, prev: Optional[torch.Tensor] = None, prev2cur: Optional[torch.Tensor] = None):
        if prev is not None:
            prev = self.warp_bev(prev, prev2cur)
        h = self.cell(x, prev)
        return h


# ------------------------------------------------------------------------------------
# Encoder
# ------------------------------------------------------------------------------------

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
        downs = list()

        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewDeformableBlock(
                feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap
            )
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)])
            layers.append(layer)

            if i < len(middle) - 1:
                downs.append(nn.Sequential(
                    nn.Conv2d(dim[i], dim[i] // 2, 3, stride=1, padding=1, bias=False),
                    nn.PixelUnshuffle(2),
                    nn.Conv2d(dim[i+1], dim[i+1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim[i+1], dim[i+1], 1, bias=False),
                    nn.BatchNorm2d(dim[i+1]),
                ))

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downs)

        # Temporal fusion after each scale
        self.temporal = nn.ModuleList([TemporalBEVGRU(dim[i]) for i in range(len(middle))])

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        # images to backbone
        image = batch['image'].flatten(0, 1)                  # (b*n, c, h, w)
        K = batch['intrinsics']                               # (b, n, 3, 3)
        E = batch['extrinsics']                               # (b, n, 4, 4)
        I_inv = K.inverse()
        E_inv = E.inverse()

        # optional temporal inputs
        prev_bev = batch.get('prev_bev', None)                # (b, d, H, W) or None
        prev2cur = batch.get('prev2cur_bev', None)            # (b, 2, 3) or None

        features = [self.down(y) for y in self.backbone(self.norm(image))]
        x = self.bev_embedding.get_prior()                    # (d, H, W)
        x = repeat(x, '... -> b ...', b=b)                    # (b, d, H, W)

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(i, x, self.bev_embedding, feature, K, E, I_inv, E_inv, batch.get('object_count', None))
            x = layer(x)

            # temporal fusion at each stage
            x = self.temporal[i](x, prev=prev_bev if i == 0 else None, prev2cur=prev2cur if i == 0 else None)

            if i < len(features) - 1:
                x = self.downsample_layers[i](x)

        return x


# ------------------------------------------------------------------------------------
# Quick test (shape-level)
# ------------------------------------------------------------------------------------

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

    # Dummy backbone stub (replace with real one)
    class DummyBackbone(nn.Module):
        def __init__(self, output_shapes):
            super().__init__()
            self.output_shapes = output_shapes
            C = output_shapes[0][1]
            self.net = nn.Conv2d(3, C, 3, padding=1)

        def forward(self, x):
            # produce pyramid features matching output_shapes
            outs = []
            for shape in self.output_shapes:
                _, c, h, w = shape
                feat = F.interpolate(self.net(x), size=(h, w), mode='bilinear', align_corners=False)
                outs.append(feat)
            return outs

    # Minimal config
    dim = [128, 128]
    backbone = DummyBackbone(output_shapes=[(1, 128, 28, 60), (1, 128, 14, 30)])

    cross_view = dict(
        image_height=0, image_width=0, qkv_bias=True,
    )
    cross_view_swap = dict(
        q_win_size=[[5,5],[5,5]], feat_win_size=[[6,12],[6,12]],
        heads=[4,4], dim_head=[32,32], bev_embedding_flag=[True,False],
        rel_pos_emb=False, no_image_features=False, skip=True
    )
    bev_embedding = dict(sigma=0.02, bev_height=50, bev_width=50, h_meters=100, w_meters=100, offset=0.0, upsample_scales=[2,4])

    enc = PyramidAxialEncoder(
        backbone=backbone,
        cross_view=cross_view,
        cross_view_swap=cross_view_swap,
        bev_embedding=dict(dim=dim[0], **bev_embedding),
        self_attn={},
        dim=dim,
        middle=[1,1],
        scale=1.0
    ).cuda()

    b, n = 2, 6
    image = torch.rand(b, n, 3, 224, 480).cuda()
    intr = torch.eye(3).view(1,1,3,3).repeat(b,n,1,1).cuda()
    extr = torch.eye(4).view(1,1,4,4).repeat(b,n,1,1).cuda()

    batch = {
        'image': image,
        'intrinsics': intr,
        'extrinsics': extr,
        'prev_bev': torch.zeros(b, dim[0], bev_embedding['bev_height']//bev_embedding['upsample_scales'][0],
                                bev_embedding['bev_width']//bev_embedding['upsample_scales'][0]).cuda(),
        'prev2cur_bev': torch.tensor([[[1,0,0],[0,1,0]]], dtype=torch.float32).repeat(b,1,1).cuda()
    }

    with torch.no_grad():
        out = enc(batch)
        print('Output:', out.shape
