import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from .layers import Norm, MultiHeadAttention, FeedForward, PositionalEncoder
import logging


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def get_targets(model, inputs, cond, denoise_steps, device, segs_mask, bootstrap_every=8, force_t=-1, force_dt=-1):
    model.eval()
    
    batch_size = inputs.shape[0]

    # 1. ========== Sample dt ==========
    bootstrap_batchsize = batch_size // bootstrap_every
    log2_sections = int(math.log2(denoise_steps))

    dt_base = torch.repeat_interleave(log2_sections - 1 - torch.arange(log2_sections), bootstrap_batchsize // log2_sections)
    dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batchsize - dt_base.shape[0],)])

    force_dt_vec = torch.ones(bootstrap_batchsize) * force_dt
    dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base).to(device)
    dt = 1 / (2 ** (dt_base))
    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2

    # 2. ========== Sample t ==========
    dt_sections = 2 ** dt_base
    t = torch.cat([
        torch.randint(low=0, high=int(val.item()), size=(1, )).float() for val in dt_sections
    ]).to(device)
    t = t / dt_sections
    force_t_vec = torch.ones(bootstrap_batchsize, dtype=torch.float32).to(device) * force_t
    t = torch.where(force_t_vec != -1, force_t_vec, t).to(device)
    t_full = t[:, None, None]

    # 3. ========== Generate Bootstrap Targets ==========
    x_1 = inputs[:bootstrap_batchsize]
    cond_bst = cond[:bootstrap_batchsize]
    segs_mask_bst = segs_mask[:bootstrap_batchsize]
    x_0 = torch.randn_like(x_1).masked_fill(segs_mask_bst == 0, 0)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1

    with torch.no_grad():
        v_b1 = model(x_t, t, dt_base_bootstrap, cond_bst, segs_mask_bst)
    t2 = t + dt_bootstrap
    x_t2 = x_t + dt_bootstrap[:, None, None] * v_b1
    x_t2 = torch.clip(x_t2, -4, 4)

    with torch.no_grad():
        v_b2 = model(x_t2, t2, dt_base_bootstrap, cond_bst, segs_mask_bst)

    v_target = (v_b1 + v_b2) / 2
    v_target = torch.clip(v_target, -4, 4)
    v_target = v_target.masked_fill(segs_mask_bst == 0, 0)

    bst_v = v_target
    bst_dt = dt_base
    bst_t = t
    bst_xt = x_t
    
    # 4. ========== Generate Flow-Matching Targets ==========
    # sample t
    t = torch.randint(low=0, high=denoise_steps, size=(inputs.shape[0],), dtype=torch.float32)
    t /= denoise_steps
    force_t_vec = torch.ones(inputs.shape[0]) * force_t
    t = torch.where(force_t_vec != -1, force_t_vec, t).to(device)
    t_full = t[:, None, None]

    # sample flow pairs x_t, v_t
    x_0 = torch.randn_like(inputs).masked_fill(segs_mask == 0, 0)
    x_1 = inputs
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t = x_1 - (1 - 1e-5) * x_0
    v_t = v_t.masked_fill(segs_mask == 0, 0)

    dt_flow = int(math.log2(denoise_steps))
    dt_base = (torch.ones(inputs.shape[0], dtype=torch.int32) * dt_flow).to(device)

    # 5. ========== Merge Flow and Bootstrap ==========
    bst_size = batch_size // bootstrap_every
    bst_size_data = batch_size - bst_size
    x_t = torch.cat([bst_xt, x_t[-bst_size_data:]], dim=0)
    t = torch.cat([bst_t, t[-bst_size_data:]], dim=0)
    dt_base = torch.cat([bst_dt, dt_base[-bst_size_data:]], dim=0)
    v_t = torch.cat([bst_v, v_t[-bst_size_data:]], dim=0)

    return x_t, v_t, t, dt_base


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class DiTBlock(nn.Module):
    def __init__(self, hid_dim, num_heads=4, dropout=0.1):
        super(DiTBlock, self).__init__()
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.cond_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hid_dim, 6 * hid_dim)
        )

        self.norm1 = Norm(hid_dim)
        self.norm2 = Norm(hid_dim)

        self.attn = MultiHeadAttention(num_heads, hid_dim, dropout)
        self.ff = FeedForward(hid_dim, d_ff=hid_dim * 2)

    def forward(self, x, c):
        cond = self.cond_linear(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond, 6, dim=-1)
        
        x_norm1 = self.norm1(x)
        x_modulated = modulate(x_norm1, shift_msa, scale_msa)

        attn_x = self.attn(x_modulated, x_modulated, x_modulated)

        x = x + (gate_msa * attn_x)

        x_norm2 = self.norm2(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)

        mlp_x = self.ff(x_modulated2)
        x = x + (gate_mlp * mlp_x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(OutputLayer, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.cond_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hid_dim, 2 * hid_dim)
        )

        self.norm = Norm(hid_dim)
        self.output_linear = nn.Linear(hid_dim, out_dim)
    
    def forward(self, x, c):
        cond = self.cond_linear(c)
        shift, scale = torch.chunk(cond, 2, dim=-1)

        x_norm = self.norm(x)
        x_modulated = modulate(x_norm, shift, scale)
        x = self.output_linear(x_modulated)
        
        return x


class DiT(nn.Module):
    def __init__(self, out_dim, hid_dim, depth, cond_dim):
        super(DiT, self).__init__()
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.depth = depth

        sinu_pos_emb = SinusoidalPosEmb(hid_dim)
        fourier_dim = hid_dim
        time_dim = hid_dim

        self.pe = PositionalEncoder(hid_dim, max_seq_len=2000)

        self.time_embedder = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.timestep_embedder = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.cond_linear = nn.Linear(cond_dim, hid_dim)

        self.DiTBlocks = nn.ModuleList(
            [
                DiTBlock(hid_dim) for _ in range(depth)
            ]
        )

        self.noise_linear = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.ReLU()
        )

        self.output = OutputLayer(hid_dim, out_dim)

    def forward(self, x, t, dt, cond, segs_mask):
        x = self.noise_linear(x)
        x = self.pe(x)

        # logging.info(f'cond shape: {cond.shape}')
        c = self.cond_linear(cond)
        # logging.info(f'c shape: {c.shape}')
        te = self.time_embedder(t)
        dte = self.timestep_embedder(dt)

        # logging.info(f'te shape: {te[:, None].shape}')
        c = c + te[:, None] + dte[:, None] # (B, 1, d)
        # print(f'condition shape: {c.shape}')

        for i in range(self.depth):
            x = self.DiTBlocks[i](x, c)

        x = self.output(x, cond)

        x = x.masked_fill(segs_mask == 0, 0)
        return x


class ShortCut(nn.Module):
    def __init__(self, model, infer_steps, seq_length, bootstrap_every=8):
        super().__init__()

        self.model = model
        self.infer_steps = infer_steps
        self.seq_length = seq_length
        self.bootstrap_every = bootstrap_every
    
    def forward(self, x_t, v_t, t, dt_base, cond, x_1, segs_mask):
        v_pred = self.model(x_t, t, dt_base, cond, segs_mask)
        # TODO: add bce loss
        x_pred = x_t + v_pred
        mse_loss = F.mse_loss(v_pred, v_t)
        bce_loss = F.binary_cross_entropy(F.softmax(x_pred.masked_fill(segs_mask == 0, -1e9), dim=-1), x_1, reduction='mean')
        loss = mse_loss + bce_loss
        return loss

    @torch.no_grad()
    def inference(self, batch_size, cond, segs_mask):
        device = cond.device
        eps = torch.randn((batch_size, 1, self.seq_length), device=device)

        delta_t = 1.0 / self.infer_steps
        x = eps.masked_fill(segs_mask == 0, 0)

        for ti in range(self.infer_steps):
            t = ti / self.infer_steps
            
            t_vector = torch.full((eps.shape[0],), t).to(device)
            dt_base = torch.ones_like(t_vector).to(device) * math.log2(self.infer_steps)

            v = self.model(x, t_vector, dt_base, cond, segs_mask)

            x = x + v * delta_t
        
        # x = x.masked_fill(segs_mask == 0, 0)
        x = F.softmax(x.masked_fill(segs_mask == 0, -1e9), dim=-1)

        return x
