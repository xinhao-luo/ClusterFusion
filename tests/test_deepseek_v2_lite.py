import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import flashinfer

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

class DeepseekV2LiteVanilla(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_size = 2048
        self.num_heads = 16

        self.qk_rope_head_dim = 64
        self.kv_lora_rank = 512
        self.v_head_dim = 128
        self.qk_nope_head_dim = 128
        self.q_head_dim = 192  # 192 = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.cos = torch.full((1, self.qk_rope_head_dim), 1.0, dtype=torch.float32)
        self.sin = torch.full((1, self.qk_rope_head_dim), 0.0, dtype=torch.float32)

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.q_head_dim,
            bias=False,
        )
        torch.nn.init.normal_(self.q_proj.weight)

        self.kv_a_proj_with_mqa = nn.Linear( # [,2048]-->[, 512+64] W^DKV & W^KR = [2048, 512+64]
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(self.kv_lora_rank)

        # W^UK & W^UV ~ [512, 16*(128+128)]
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads
            * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        torch.nn.init.normal_(self.kv_b_proj.weight)

        # W^O ~ [16*128, 2048]
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )
        torch.nn.init.normal_(self.o_proj.weight)

        self.softmax_scale = self.q_head_dim ** (-0.5)

class DeepseekV2LiteMatAbsorbDecode(nn.Module):

    def __init__(self, mla_vanilla: DeepseekV2LiteVanilla):
        super().__init__()

        self.hidden_size = mla_vanilla.hidden_size  # 2048
        self.num_heads = mla_vanilla.num_heads  # 16

        self.qk_rope_head_dim = mla_vanilla.qk_rope_head_dim  # 64
        self.kv_lora_rank = mla_vanilla.kv_lora_rank  # 512
        self.v_head_dim = mla_vanilla.v_head_dim  # 128
        self.qk_nope_head_dim = mla_vanilla.qk_nope_head_dim  # 128
        self.q_head_dim = (
            mla_vanilla.q_head_dim
        )  # qk_nope_head_dim + qk_rope_head_dim # 128+64=192
        self.kv_a_proj_with_mqa = mla_vanilla.kv_a_proj_with_mqa
        self.kv_a_layernorm = mla_vanilla.kv_a_layernorm

        self.cos = mla_vanilla.cos
        self.sin = mla_vanilla.sin

        self.softmax_scale = mla_vanilla.softmax_scale
        
        self.W_UQR = mla_vanilla.q_proj.weight.t().view(self.hidden_size, self.num_heads*self.q_head_dim)

        # W_UK ~ [16, 512, 128]   W_UV ~ [16, 512, 128]
        self.W_UK, self.W_UV = torch.split(
            mla_vanilla.kv_b_proj.weight.t().view(
                self.kv_lora_rank,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ),
            [self.qk_nope_head_dim, self.v_head_dim],
            -1,
        )
        self.W_UK = self.W_UK.transpose(0, 1)
        self.W_UV = self.W_UV.transpose(0, 1)
        self.W_O = mla_vanilla.o_proj.weight.t().view(
            self.num_heads * self.v_head_dim, self.hidden_size
        )

    def run_proof_of_concept(
        self,
        hidden_states: torch.Tensor,
        compressed_kv_normed_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Q
        q = torch.matmul(hidden_states, self.W_UQR).view(
            -1, self.num_heads, self.q_head_dim
        )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # q_nope ~ [bsz, 16, 512], q_nope ~ [bsz, 16, 64]
        q_nope = torch.bmm(q_nope.transpose(0, 1), self.W_UK.transpose(1, 2)).transpose(0, 1)

        # Compressed KV
        latent_cache = self.kv_a_proj_with_mqa(hidden_states).unsqueeze(1)
        c = latent_cache[..., : self.kv_lora_rank]
        c = self.kv_a_layernorm(c.contiguous())
        k_pe = latent_cache[..., self.kv_lora_rank :]
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, self.cos.to(q_pe.device), self.sin.to(q_pe.device))
        k_pe_cache[:, -1:, :] = k_pe
        compressed_kv_normed_cache[:, -1:, :] = c
        # MQA
        if kv_len % page_size != 0:
            raise ValueError(
                "For simplicity, kv_len should be multiple of page_size."
            )
        num_pages_per_seq = kv_len // page_size
        total_num_pages = num_pages_per_seq * bsz

        q_indptr = torch.arange(0, bsz + 1).to(0).int()
        kv_indptr = torch.arange(0, bsz + 1).to(dev_id).int() * num_pages_per_seq
        kv_indices = torch.arange(0, total_num_pages).to(dev_id).int()
        kv_last_page_len = torch.full((bsz,), page_size, dtype=torch.int32).to(
            dev_id
        )
        kv_lens = torch.full((bsz,), kv_len, dtype=torch.int32).to(0)

        paged_ckv_cache = compressed_kv_normed_cache.reshape(
            total_num_pages, page_size, self.kv_lora_rank
        )
        paged_kpe_cache = k_pe_cache.reshape(
            total_num_pages, page_size, self.qk_rope_head_dim
        )

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(
            dev_id
        )
        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace_buffer)
        wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_lens,
            self.num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            page_size,
            False,
            self.softmax_scale,
            q_nope.dtype,
            paged_ckv_cache.dtype,
        )
        attn_output = wrapper.run(q_nope, q_pe, paged_ckv_cache, paged_kpe_cache)

        # output ~ [bsz, 2048]
        output = torch.bmm(attn_output.transpose(0, 1), self.W_UV).transpose(0, 1)
        output = torch.matmul(output.flatten(1, 2), self.W_O)  

        return output


if __name__ == "__main__":

    dev_id = 0

    torch.manual_seed(666)
    torch.set_grad_enabled(False)

    mla_vanilla = DeepseekV2LiteVanilla().cuda(device=dev_id).half()

    bsz = 1
    kv_len = 4096
    page_size = 16

    hidden_states = torch.randn([bsz, 1, mla_vanilla.hidden_size]).to(dev_id).half()
    compressed_kv_normed_cache = torch.randn(
        [bsz, kv_len, mla_vanilla.kv_lora_rank]
    ).to(dev_id).half()
    k_pe_cache = torch.randn([bsz, kv_len, mla_vanilla.qk_rope_head_dim]).to(dev_id).half()

    mla_mat_absorb = DeepseekV2LiteMatAbsorbDecode(mla_vanilla).cuda(device=dev_id).half()
    output_mat_absorbed_use_flashinfer = mla_mat_absorb.run_proof_of_concept(
        hidden_states.squeeze(1),
        compressed_kv_normed_cache,
        k_pe_cache,
    )
    print(output_mat_absorbed_use_flashinfer, output_mat_absorbed_use_flashinfer.shape)
