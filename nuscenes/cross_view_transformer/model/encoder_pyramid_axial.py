# encoder_pyramid_axial.py (에러 수정 전체 파일)
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List, Optional, Tuple

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


def ego_to_bev_xy(xy: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(xy[..., :1])
    homo = torch.cat([xy, ones], dim=-1)
    pix = homo @ V.T
    pix = pix[..., :2] / pix[..., 2:].clamp(min=1e-7)
    return pix


def normalize_grid_uv(uv: torch.Tensor, h: int, w: int) -> torch.Tensor:
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
        V = torch.FloatTensor(get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset))
        V_inv = V.inverse()
        self.register_buffer('V', V, persistent=False)
        self.register_buffer('V_inv', V_inv, persistent=False)

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
        logits = self.head(x)
        prob = logits.softmax(dim=1)
        return prob


class LiftSplatBEV(nn.Module):
    def __init__(self, bev_h: int, bev_w: int, h_m: float = 100.0, w_m: float = 100.0, offset: float = 0.0):
        super().__init__()
        V = torch.FloatTensor(get_view_matrix(bev_h, bev_w, h_m, w_m, offset))
        self.register_buffer('V', V, persistent=False)
        self.bev_h = bev_h
        self.bev_w = bev_w

    @staticmethod
    def _ray_to_ground_intersection(c_xyz: torch.Tensor, d_xyz: torch.Tensor) -> torch.Tensor:
        dz = d_xyz[..., 2].clamp(min=1e-6)
        s = (-c_xyz[..., 2]) / dz
        xy = c_xyz[..., :2] + d_xyz[..., :2] * s[..., None]
        return xy

    def forward(self, feat_bn: torch.Tensor, depth_prob: torch.Tensor, I_inv: torch.Tensor, E_inv: torch.Tensor,
                feat_hw: Tuple[int, int], n: int):
        device = feat_bn.device
        b_n, C, h, w = feat_bn.shape
        b = b_n // n

        xs = torch.linspace(0, w - 1, w, device=device)
        ys = torch.linspace(0, h - 1, h, device=device)
        u, v = torch.meshgrid(xs, ys, indexing='xy')
        pix = torch.stack([u, v, torch.ones_like(u)], dim=0)
        pix = rearrange(pix, 'd w h -> d (h w)')

        I_inv_bn = rearrange(I_inv, 'b n ... -> (b n) ...', b=b, n=n)
        E_inv_bn = rearrange(E_inv, 'b n ... -> (b n) ...', b=b, n=n)

        cam = I_inv_bn @ pix
        cam = F.pad(cam, (0, 0, 0, 1), value=1.0)
        d = E_inv_bn @ cam
        d = rearrange(d, 'bn d (hw) -> bn (hw) d', hw=h*w)[..., :3]

        c = E_inv_bn[..., :3, 3]
        c = c[:, None, :].expand(-1, h * w, -1)

        D = depth_prob.shape[1]
        z_bins = torch.linspace(1.0, 60.0, D, device=device)
        z_exp = (depth_prob * z_bins[None, :, None, None]).sum(dim=1)
        z_exp = rearrange(z_exp, 'bn h w -> bn (h w) 1')

        xy_ego = self._ray_to_ground_intersection(c, d)
        bev_xy = ego_to_bev_xy(xy_ego, self.V)
        gx = (bev_xy[..., 0] / (self.bev_w - 1)) * 2 - 1
        gy = (bev_xy[..., 1] / (self.bev_h - 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)

        conf = depth_prob.max(dim=1).values
        conf = rearrange(conf, 'bn h w -> bn (h w) 1')

        px = (gx + 1) * (self.bev_w - 1) / 2
        py = (gy + 1) * (self.bev_h - 1) / 2

        x0 = px.floor().clamp(0, self.bev_w - 1)
        y0 = py.floor().clamp(0, self.bev_h - 1)
        x1 = (x0 + 1).clamp(0, self.bev_w - 1)
        y1 = (y0 + 1).clamp(0, self.bev_h - 1)

        wa = (x1 - px) * (y1 - py)
        wb = (px - x0) * (y1 - py)
        wc = (x1 - px) * (py - y0)
        wd = (px - x0) * (py - y0)

        feat_flat = rearrange(feat_bn, 'bn c h w -> bn c (h w)')
        bev = torch.zeros(b*n, C, self.bev_h, self.bev_w, device=device)

        def add_weighted(ix, iy, wgt):
            idx = (iy.long() * self.bev_w + ix.long())
            wfeat = feat_flat * (wgt * conf).transpose(1, 0).transpose(1, 2)
            bev.view(b*n, C, -1).scatter_add_(2, idx[:, None, :], wfeat)

        add_weighted(x0, y0, wa)
        add_weighted(x1, y0, wb)
        add_weighted(x0, y1, wc)
        add_weighted(x1, y1, wd)

        bev = rearrange(bev, '(b n) c h w -> b n c h w', b=b, n=n).mean(dim=1)
        return bev


class DeformableCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32, n_points: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.n_points = n_points
        self.scale = dim_head ** -0.5
        inner = heads * dim_head

        self.to_q = nn.Linear(dim, inner, bias=qkv_bias)
        self.to_kv = nn.Conv2d(dim, 2 * inner, 1, bias=qkv_bias)

        self.offset_mlp = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, n_points * 2, 1, bias=True)
        )

        self.proj = nn.Linear(inner, dim)

    def _project_bev_to_image(self, world_xy: torch.Tensor, K: torch.Tensor, E: torch.Tensor):
        b, n, _, H, W = world_xy.shape
        ones = torch.ones(b, n, 1, H, W, device=world_xy.device)
        xyz1 = torch.cat([world_xy, torch.zeros_like(ones), ones], dim=2)
        xyz1 = rearrange(xyz1, 'b n d h w -> (b n) (h w) d')

        E_bn = rearrange(E, 'b n ... -> (b n) ...')
        K_bn = rearrange(K, 'b n ... -> (b n) ...')

        cam = (E_bn @ xyz1.transpose(1, 2)).transpose(1, 2)
        cam = cam[..., :3]
        pix = (K_bn @ cam.transpose(1, 2)).transpose(1, 2)
        uv = pix[..., :2] / pix[..., 2:].clamp(min=1e-6)
        uv = rearrange(uv, '(b n) (h w) c -> b n c h w', b=b, n=n, h=H, w=W)
        return uv

    def forward(self, bev: torch.Tensor, img_feats: torch.Tensor, K: torch.Tensor, E: torch.Tensor, world_xy: torch.Tensor):
        b, d, H, W = bev.shape
        _, n, d_img, h, w = img_feats.shape

        world_xy_bn = world_xy[None, None, ...].expand(b, n, -1, -1, -1)
        uv = self._project_bev_to_image(world_xy_bn, K, E)
        grid = normalize_grid_uv(rearrange(uv, 'b n c h w -> b n h w c'), h, w)

        kv = rearrange(img_feats, 'b n d h w -> (b n) d h w')
        kv = self.to_kv(kv)
        k_proj, v_proj = torch.chunk(kv, 2, dim=1)
        k_proj = rearrange(k_proj, '(b n) c h w -> b n c h w', b=b, n=n)
        v_proj = rearrange(v_proj, '(b n) c h w -> b n c h w', b=b, n=n)

        offset_raw = self.offset_mlp(bev)
        p = self.n_points
        offset_raw = offset_raw.view(b, p, 2, H, W)
        offset_raw = rearrange(offset_raw, 'b p c H W -> b p H W c')
        offset_raw = offset_raw[:, None, ...].expand(-1, n, -1, -1, -1, -1)

        grid_rep = grid[:, :, None, ...].expand(-1, -1, p, -1, -1, -1)
        samp_grid = (grid_rep + offset_raw).clamp(-1, 1)

        k_proj_flat = rearrange(k_proj, 'b n c h w -> (b n) c h w')
        v_proj_flat = rearrange(v_proj, 'b n c h w -> (b n) c h w')

        samp_grid_flat = rearrange(samp_grid, 'b n p H W c -> (b n) p H W c')

        k_samples = []
        v_samples = []
        for pi in range(p):
            gpi = samp_grid_flat[:, pi, ...]
            out_k = F.grid_sample(k_proj_flat, gpi, align_corners=True)
            out_v = F.grid_sample(v_proj_flat, gpi, align_corners=True)
            k_samples.append(out_k)
            v_samples.append(out_v)

        k_stack = torch.stack(k_samples, dim=1)  # (b*n, p, inner, H, W)
        v_stack = torch.stack(v_samples, dim=1)
        k_stack = rearrange(k_stack, '(b n) p c H W -> b n p c H W', b=b, n=n)
        v_stack = rearrange(v_stack, '(b n) p c H W -> b n p c H W', b=b, n=n)

        inner = self.heads * self.dim_head
        k_heads = rearrange(k_stack, 'b n p (m d) H W -> b m n p d H W', m=self.heads, d=self.dim_head)
        v_heads = rearrange(v_stack, 'b n p (m d) H W -> b m n p d H W', m=self.heads, d=self.dim_head)

        q = self.to_q(rearrange(bev, 'b d H W -> b (H W) d'))  # (b, HW, inner)
        HW = H * W
        # SAFE reshape without relying on einops to infer H,W
        q = q.view(b, HW, self.heads, self.dim_head).permute(0, 2, 1, 3)  # b, m, HW, d

        # prepare for dot product
        q_hw = q.permute(0, 1, 3, 2)  # b, m, d, HW
        k_hw = rearrange(k_heads, 'b m n p d H W -> b m n p d (H W)')
        # sim: b m n p HW  (einsum simpler)
        sim = torch.einsum('b m d q, b m n p d q -> b m n p q', q_hw, k_hw)
        att = sim.softmax(dim=3)

        v_hw = rearrange(v_heads, 'b m n p d H W -> b m n p d (H W)')
        weighted = (att.unsqueeze(4) * v_hw).sum(dim=3)  # b m n d HW
        weighted = weighted.mean(dim=2)  # mean over cameras -> b m d HW

        out = rearrange(weighted, 'b m d (H W) -> b (m d) (H W)', H=H, W=W)
        out = rearrange(self.proj(rearrange(out, 'b c (H W) -> b (H W) c', H=H, W=W)), 'b (H W) c -> b c H W', H=H, W=W)
        return out


class CrossViewDeformableBlock(nn.Module):
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
        q_win_size: list = None,
        feat_win_size: list = None,
        heads: list = None,
        dim_head: list = None,
        bev_embedding_flag: list = None,
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

        self.img_proj = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, dim, 1, bias=False)
        )

        if heads is None: heads = [4]
        if dim_head is None: dim_head = [32]
        if bev_embedding_flag is None: bev_embedding_flag = [True]

        self.deform_attn = DeformableCrossAttention(dim, heads[index], dim_head[index], n_points, qkv_bias)
        self.skip = skip

        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.postnorm = norm(dim)

        self.depth_head = DepthHead(feat_dim, n_bins=depth_bins)
        self.lift_splat = None

        self.fuse_gate = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.bev_embed_flag = bev_embedding_flag[index] if index < len(bev_embedding_flag) else False
        if self.bev_embed_flag:
            self.bev_pos_proj = nn.Conv2d(2, dim, 1)

    def build_lift(self, bev: BEVEmbedding):
        H, W = bev.learned_features.shape[-2:]
        self.lift_splat = LiftSplatBEV(bev_h=H, bev_w=W)

    def forward(
        self,
        index: int,
        x: torch.Tensor,
        bev: BEVEmbedding,
        feature: torch.Tensor,
        K: torch.Tensor,
        E: torch.Tensor,
        I_inv: torch.Tensor,
        E_inv: torch.Tensor,
        object_count: Optional[torch.Tensor] = None,
    ):
        b, n, _, h, w = feature.shape
        _, d, H, W = x.shape

        if self.lift_splat is None:
            self.build_lift(bev)

        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        else:
            world = bev.grid3[:2]

        img_f = rearrange(feature, 'b n c h w -> (b n) c h w')
        img_f_proj = rearrange(self.img_proj(img_f), '(b n) c h w -> b n c h w', b=b, n=n)

        x_skip = x
        x = self.deform_attn(x, img_f_proj, K, E, world) + (x_skip if self.skip else 0)
        x_flat = rearrange(x, 'b d H W -> b (H W) d')
        x_flat = x_flat + self.mlp(self.prenorm(x_flat))
        x = rearrange(self.postnorm(x_flat), 'b (H W) d -> b d H W', H=H, W=W)

        depth_prob = self.depth_head(img_f)  # (b*n, D, h, w)
        lift_bev = self.lift_splat(img_f_proj.reshape(b*n, -1, h, w), depth_prob, I_inv, E_inv, (h, w), n)

        fused = self.fuse_gate(torch.cat([x, lift_bev], dim=1))
        return fused


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


class TemporalBEVGRU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.cell = ConvGRUCell(dim)

    @staticmethod
    def warp_bev(x: torch.Tensor, A: Optional[torch.Tensor]) -> torch.Tensor:
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

            cva = CrossViewDeformableBlock(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
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

        self.temporal = nn.ModuleList([TemporalBEVGRU(dim[i]) for i in range(len(middle))])

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1)
        K = batch['intrinsics']
        E = batch['extrinsics']
        I_inv = K.inverse()
        E_inv = E.inverse()

        prev_bev = batch.get('prev_bev', None)
        prev2cur = batch.get('prev2cur_bev', None)

        features = [self.down(y) for y in self.backbone(self.norm(image))]
        x = self.bev_embedding.get_prior()
        x = repeat(x, '... -> b ...', b=b)

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, K, E, I_inv, E_inv, batch.get('object_count', None))
            x = layer(x)
            x = self.temporal[i](x, prev=prev_bev if i == 0 else None, prev2cur=prev2cur if i == 0 else None)
            if i < len(features) - 1:
                x = self.downsample_layers[i](x)

        return x


if __name__ == "__main__":
    import os, re, yaml

    class DummyBackbone(nn.Module):
        def __init__(self, output_shapes):
            super().__init__()
            self.output_shapes = output_shapes
            self.net = nn.Conv2d(3, output_shapes[0][1], 3, padding=1)
        def forward(self, x):
            outs = []
            for shape in self.output_shapes:
                _, c, h, w = shape
                outs.append(F.interpolate(self.net(x), size=(h, w), mode='bilinear', align_corners=False))
            return outs

    dim = [128, 128]
    backbone = DummyBackbone(output_shapes=[(1, 128, 28, 60), (1, 128, 14, 30)])
    cross_view = dict(image_height=0, image_width=0, qkv_bias=True)
    cross_view_swap = dict(q_win_size=[[5,5],[5,5]], feat_win_size=[[6,12],[6,12]],
                           heads=[4,4], dim_head=[32,32], bev_embedding_flag=[True,False],
                           rel_pos_emb=False, no_image_features=False, skip=True)
    bev_embedding = dict(dim=dim[0], sigma=0.02, bev_height=50, bev_width=50, h_meters=100, w_meters=100, offset=0.0, upsample_scales=[2,4])

    enc = PyramidAxialEncoder(
        backbone=backbone,
        cross_view=cross_view,
        cross_view_swap=cross_view_swap,
        bev_embedding=bev_embedding,
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
        print('Output:', out.shape)
