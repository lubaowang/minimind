# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             Wormhole-LM Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class WormholeConfig(PretrainedConfig):
    """
    Wormhole-LM 配置类

    基于MiniMind配置，添加虫洞机制相关参数：
    - 螺旋存储参数 (spiral_storage)
    - 虫洞索引参数 (wormhole_index)
    - 相对论注意力参数 (relativistic_attention)
    """
    model_type = "wormhole"

    def __init__(
            self,
            # ========== MiniMind基础配置 ==========
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            # MOE配置
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,

            # ========== 虫洞机制配置 ==========
            # 螺旋存储参数
            spiral_max_layers: int = 16,          # 螺旋最大层数
            spiral_layer_capacity: int = 1000,    # 每层容量
            spiral_growth_rate: float = 0.1,       # 螺旋增长率
            # 重要性势能权重
            alpha_time: float = 0.3,              # 时间势能权重
            beta_freq: float = 0.3,               # 访问势能权重
            gamma_semantic: float = 0.2,          # 语义势能权重
            delta_recency: float = 0.2,           # 近因势能权重

            # 虫洞索引参数
            wormhole_similarity_threshold: float = 0.7,  # 建立虫洞的相似度阈值
            wormhole_min_distance: float = 5.0,    # 最小螺旋距离
            wormhole_max_staleness: float = 86400, # 虫洞失效时间(秒)

            # 相对论注意力参数
            relativistic_attention: bool = True,   # 是否启用相对论注意力
            max_propagation_speed: float = 2.0,    # 最大传播速度
            min_propagation_speed: float = 0.5,   # 最小传播速度
            importance_decay: float = 0.001,      # 重要性衰减率

            # 虫洞检索参数
            enable_wormhole_retrieval: bool = True,  # 是否启用虫洞检索
            retrieval_top_k: int = 10,           # 检索返回数量
            wormhole_traversal_steps: int = 3,   # 虫洞穿越步数

            **kwargs
    ):
        super().__init__(**kwargs)
        # ========== MiniMind基础配置 ==========
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling

        # RoPE外推配置
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None

        self.flash_attn = flash_attn

        # MOE配置
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob

        # ========== 虫洞机制配置 ==========
        # 螺旋存储
        self.spiral_max_layers = spiral_max_layers
        self.spiral_layer_capacity = spiral_layer_capacity
        self.spiral_growth_rate = spiral_growth_rate
        self.alpha_time = alpha_time
        self.beta_freq = beta_freq
        self.gamma_semantic = gamma_semantic
        self.delta_recency = delta_recency

        # 虫洞索引
        self.wormhole_similarity_threshold = wormhole_similarity_threshold
        self.wormhole_min_distance = wormhole_min_distance
        self.wormhole_max_staleness = wormhole_max_staleness

        # 相对论注意力
        self.relativistic_attention = relativistic_attention
        self.max_propagation_speed = max_propagation_speed
        self.min_propagation_speed = min_propagation_speed
        self.importance_decay = importance_decay

        # 虫洞检索
        self.enable_wormhole_retrieval = enable_wormhole_retrieval
        self.retrieval_top_k = retrieval_top_k
        self.wormhole_traversal_steps = wormhole_traversal_steps


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             Wormhole-LM Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union, Dict, Any
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(nn.Module):
    """RMSNorm: Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """预计算RoPE频率矩阵"""
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """应用旋转位置编码"""
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """KV头复制"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


# ========== 螺旋位置编码 ==========

class SpiralPositionalEncoding(nn.Module):
    """
    螺旋位置编码

    将传统的一维位置编码扩展到螺旋坐标系:
    - 径向编码 (n): 距离中心的层级
    - 角度编码 (m): 螺旋角度位置
    - 深度编码 (k): 垂直维度

    这种编码模拟了星系螺旋结构，使重要信息更靠近核心
    """

    def __init__(self, config: WormholeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_layers = config.spiral_max_layers

        # 径向位置编码
        self.radial_freq = nn.Parameter(
            torch.randn(1, 1, self.max_layers, self.head_dim) * 0.02
        )

        # 角度位置编码
        self.angular_freq = nn.Parameter(
            torch.randn(1, 1, 512, self.head_dim) * 0.02  # 512个角度位置
        )

    def forward(self, hidden_states: torch.Tensor, spiral_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        应用螺旋位置编码

        参数:
            hidden_states: [batch, seq_len, hidden_size]
            spiral_coords: [batch, seq_len, 3] - (n, m, k) 螺旋坐标

        返回:
            添加位置编码后的hidden_states
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 如果没有提供螺旋坐标，使用默认位置
        if spiral_coords is None:
            # 默认: 所有位置在核心层 (n=0)
            spiral_coords = torch.zeros(batch_size, seq_len, 3, device=hidden_states.device)

        # 提取螺旋坐标
        n = spiral_coords[:, :, 0].long().clamp(0, self.max_layers - 1)  # [batch, seq]
        m = spiral_coords[:, :, 1].long().clamp(0, 511)  # 角度位置
        k = spiral_coords[:, :, 2].long().clamp(0, 9)  # 深度

        # 获取径向编码
        radial_encoding = self.radial_freq[0, 0, n]  # [batch, seq, head_dim]

        # 获取角度编码
        angular_encoding = self.angular_freq[0, 0, m]  # [batch, seq, head_dim]

        # 合并径向和角度编码
        spiral_encoding = radial_encoding + angular_encoding

        # 扩展到所有注意力头
        head_dim = self.head_dim
        spiral_encoding = spiral_encoding.unsqueeze(2).expand(-1, -1, self.config.num_attention_heads, -1)
        spiral_encoding = spiral_encoding.reshape(batch_size, seq_len, self.config.num_attention_heads * head_dim)

        return hidden_states + spiral_encoding


# ========== 重要性权重计算 ==========

class ImportanceCalculator(nn.Module):
    """
    重要性权重计算器

    计算信息单元的重要性势能:
    E_total = α·E_time + β·E_freq + γ·E_semantic + δ·E_recency
    """

    def __init__(self, config: WormholeConfig):
        super().__init__()
        self.config = config

        # 势能权重
        self.alpha = config.alpha_time
        self.beta = config.beta_freq
        self.gamma = config.gamma_semantic
        self.delta = config.delta_recency

        # 可学习的语义投影
        self.semantic_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        access_counts: Optional[torch.Tensor] = None,
        last_accessed: Optional[torch.Tensor] = None,
        core_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算重要性权重

        参数:
            hidden_states: [batch, seq_len, hidden]
            timestamps: 创建时间戳
            access_counts: 访问次数
            last_accessed: 最后访问时间
            core_embedding: 核心主题向量

        返回:
            importance_weights: [batch, seq_len]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        current_time = torch.tensor([torch.cuda.Event(enable_timing=False).record() if torch.cuda.is_available() else 0],
                                     device=device).float()

        # 时间势能: 越新越高
        if timestamps is not None:
            time_elapsed = current_time.unsqueeze(-1) - timestamps
            E_time = 1.0 / (1.0 + self.config.importance_decay * time_elapsed)
        else:
            E_time = torch.ones(batch_size, seq_len, device=device) * 0.5

        # 访问势能: 访问越频繁越高
        if access_counts is not None:
            max_access = access_counts.max().clamp(min=1)
            E_freq = access_counts.float() / max_access
        else:
            E_freq = torch.ones(batch_size, seq_len, device=device) * 0.5

        # 语义势能: 与核心主题越接近越高
        if core_embedding is not None:
            projected = self.semantic_proj(hidden_states)
            normalized_proj = F.normalize(projected, p=2, dim=-1)
            normalized_core = F.normalize(core_embedding, p=2, dim=-1)
            E_semantic = (normalized_proj @ normalized_core.unsqueeze(-1)).squeeze(-1)
        else:
            E_semantic = torch.ones(batch_size, seq_len, device=device) * 0.5

        # 近因势能: 最近访问的更高
        if last_accessed is not None:
            recency_elapsed = current_time.unsqueeze(-1) - last_accessed
            E_recency = torch.exp(-self.config.importance_decay * recency_elapsed)
        else:
            E_recency = torch.ones(batch_size, seq_len, device=device) * 0.5

        # 综合势能
        importance = (
            self.alpha * E_time +
            self.beta * E_freq +
            self.gamma * E_semantic +
            self.delta * E_recency
        )

        return importance.clamp(0, 1)


# ========== 相对论注意力 ==========

class RelativisticAttention(nn.Module):
    """
    相对论注意力机制

    核心思想: 重要信息应该传播得更快、更远

    通过重要性权重调制注意力分数:
    Attention_relativistic(Q, K, V, ω) = softmax(QK^T / √d_k ⊙ Ω)V
    """

    def __init__(self, config: WormholeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # 标准投影
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # 传播速度参数
        self.max_speed = config.max_propagation_speed
        self.min_speed = config.min_propagation_speed

        # 重要性到速度的映射
        self.speed_weight = nn.Parameter(torch.ones(1, 1, self.num_heads, 1) * 0.5)
        self.speed_bias = nn.Parameter(torch.zeros(1, 1, self.num_heads, 1))

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def compute_speed(self, importance: torch.Tensor) -> torch.Tensor:
        """
        将重要性权重转换为传播速度

        参数:
            importance: [batch, seq_len]

        返回:
            speed: [batch, 1, num_heads, 1]
        """
        # 扩展到头维度
        importance = importance.unsqueeze(2)  # [batch, seq, 1]

        # 使用sigmoid映射到速度范围
        speed = self.min_speed + (self.max_speed - self.min_speed) * (
            torch.sigmoid((importance - 0.5) * 10)
        )

        return speed

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        importance_weights: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        相对论注意力前向传播

        参数:
            x: 输入 [batch, seq_len, hidden]
            position_embeddings: (cos, sin)
            importance_weights: 重要性权重 [batch, seq_len]
            past_key_value: KV缓存
            use_cache: 是否缓存
            attention_mask: 注意力掩码

        返回:
            (output, past_kv)
        """
        bsz, seq_len, _ = x.shape

        # 如果没有重要性权重，使用均匀权重
        if importance_weights is None:
            importance_weights = torch.ones(bsz, seq_len, device=x.device) / seq_len

        # 投影 Q, K, V
        xq = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)

        # 应用RoPE
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # KV缓存
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 调整维度用于注意力计算
        xq = xq.transpose(1, 2)  # [batch, heads, seq, head]
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 计算相对论注意力
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # 使用Flash Attention
            # 计算速度调制
            speed = self.compute_speed(importance_weights)  # [batch, 1, heads, 1]

            # 简化: 在Flash Attention中应用速度的方式受限
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # 标准注意力计算
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 应用相对论速度调制
            speed = self.compute_speed(importance_weights)  # [batch, 1, heads, 1]

            # 调制注意力分数
            scores = scores * speed

            # 因果掩码
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            )

            # 外部掩码
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # Softmax
            attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
            attn_weights = self.attn_dropout(attn_weights)

            # 应用注意力
            output = attn_weights @ xv

        # 恢复维度
        output = output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv


# ========== 标准注意力 (用于对比) ==========

class StandardAttention(nn.Module):
    """
    标准注意力机制 (用于非相对论模式或对比基线)
    """

    def __init__(self, args: WormholeConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


# ========== 前馈网络 ==========

class FeedForward(nn.Module):
    """标准前馈网络"""
    def __init__(self, config: WormholeConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


# ========== MoE门控 ==========

class MoEGate(nn.Module):
    """MoE门控机制"""
    def __init__(self, config: WormholeConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'unsupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """MoE前馈网络"""
    def __init__(self, config: WormholeConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
        return expert_cache


# ========== Wormhole Block ==========

class WormholeBlock(nn.Module):
    """
    Wormhole Transformer Block

    包含:
    - 螺旋位置编码
    - (相对论)注意力机制
    - MoE/标准前馈网络
    """

    def __init__(self, layer_id: int, config: WormholeConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        # 螺旋位置编码
        self.spiral_pos_encoding = SpiralPositionalEncoding(config)

        # 选择注意力类型
        if config.relativistic_attention:
            self.self_attn = RelativisticAttention(config)
        else:
            self.self_attn = StandardAttention(config)

        # Layer Norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        spiral_coords: Optional[torch.Tensor] = None,
        importance_weights: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # 应用螺旋位置编码
        hidden_states = self.spiral_pos_encoding(hidden_states, spiral_coords)

        # 注意力
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            importance_weights,
            past_key_value,
            use_cache,
            attention_mask
        )
        hidden_states += residual

        # 前馈
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value


# ========== Wormhole Model ==========

class WormholeModel(nn.Module):
    """
    Wormhole-LM 基础模型

    支持:
    - 螺旋位置编码
    - 相对论注意力
    - 虫洞检索上下文
    """

    def __init__(self, config: WormholeConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # 词嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer层
        self.layers = nn.ModuleList([
            WormholeBlock(l, config)
            for l in range(self.num_hidden_layers)
        ])

        # 输出层Norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 重要性计算器
        self.importance_calculator = ImportanceCalculator(config)

        # 预计算RoPE频率
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        spiral_coords: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs
    ):
        """
        前向传播

        参数:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            spiral_coords: [batch, seq_len, 3] - (n, m, k) 螺旋坐标
            past_key_values: KV缓存
            use_cache: 是否使用缓存
        """
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 词嵌入
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 计算重要性权重
        importance_weights = self.importance_calculator(hidden_states)

        # RoPE位置编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 逐层处理
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                spiral_coords,
                importance_weights,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 最终Layer Norm
        hidden_states = self.norm(hidden_states)

        # 计算MoE辅助损失
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze()
        )

        return hidden_states, presents, aux_loss


# ========== Wormhole LM Head ==========

class WormholeForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Wormhole-LM 因果语言模型

    兼容HuggingFace训练框架
    """

    config_class = WormholeConfig
    base_model_prefix = "wormhole"

    def __init__(self, config: WormholeConfig = None):
        self.config = config or WormholeConfig()
        super().__init__(self.config)
        self.model = WormholeModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        spiral_coords: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args
    ):
        """
        前向传播

        参数:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            labels: 标签 (用于训练)
            spiral_coords: 螺旋坐标
            past_key_values: KV缓存
            use_cache: 是否缓存
            logits_to_keep: 保留的logits数量
        """
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            spiral_coords=spiral_coords,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )

        # 提取logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 计算损失
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )
        output.aux_loss = aux_loss
        return output

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        **kwargs
    ):
        """更新生成时的模型参数"""
        past_key_values = outputs.past_key_values
        model_kwargs["past_key_values"] = past_key_values
        return model_kwargs
