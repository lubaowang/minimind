"""
Wormhole-LM LoRA 模块

基于LoRA (Low-Rank Adaptation) 技术，为Wormhole-LM模型提供高效微调能力

作者: OpenAgent Team
版本: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class LoRA(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 模块

    通过低秩矩阵分解实现高效参数微调:
    W' = W + BA

    其中:
    - W: 原始预训练权重
    - B, A: 可学习的低秩矩阵
    - rank: 秩，控制低秩矩阵的维度
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        bias: bool = False
    ):
        """
        初始化LoRA模块

        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            rank: LoRA秩
            lora_alpha: 缩放因子
            lora_dropout: dropout概率
            bias: 是否使用偏置
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # 初始化低秩矩阵
        # A: 高斯初始化
        self.lora_A = nn.Parameter(nn.init.normal_(torch.empty(rank, in_features), mean=0.0, std=0.02))
        # B: 零初始化
        self.lora_B = nn.Parameter(nn.init.zeros_(torch.empty(out_features, rank)))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # 缩放因子
        self.scaling = lora_alpha / rank if rank > 0 else 1.0

        # 标志：是否启用LoRA
        self.enable_lora = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入 [batch, seq, in_features]

        返回:
            输出 [batch, seq, out_features]
        """
        if not self.enable_lora:
            return x

        # LoRA: BA * x
        # x: [..., in_features] -> [..., rank] -> [..., out_features]
        lora_output = F.linear(
            F.linear(x, self.lora_A.T),
            self.lora_B.T
        )

        # 缩放
        lora_output = lora_output * self.scaling

        if self.bias is not None:
            lora_output = lora_output + self.bias

        return lora_output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}'


class WormholeLoRAConfig:
    """
    Wormhole-LM LoRA配置
    """

    def __init__(
        self,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
        bias: str = "none",  # "none", "all", "lora_only"
        task_type: str = "CAUSAL_LM",
    ):
        """
        LoRA配置

        参数:
            rank: LoRA秩
            lora_alpha: 缩放因子
            lora_dropout: dropout
            target_modules: 目标模块列表
            bias: 偏置类型
            task_type: 任务类型
        """
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.task_type = task_type

        # 默认目标模块 (针对Wormhole-LM优化)
        if target_modules is None:
            target_modules = [
                # 注意力模块
                "q_proj", "k_proj", "v_proj", "o_proj",
                # 前馈网络
                "gate_proj", "up_proj", "down_proj",
            ]
        self.target_modules = target_modules


def apply_lora_to_model(
    model: nn.Module,
    config: Optional[WormholeLoRAConfig] = None,
    rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None
) -> nn.Module:
    """
    为模型应用LoRA

    参数:
        model: 原始模型 (WormholeForCausalLM)
        config: LoRA配置
        rank: LoRA秩
        lora_alpha: 缩放因子
        lora_dropout: dropout
        target_modules: 目标模块名称列表

    返回:
        添加LoRA后的模型
    """
    if config is not None:
        rank = config.rank
        lora_alpha = config.lora_alpha
        lora_dropout = config.lora_dropout
        target_modules = config.target_modules

    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 检查是否是需要添加LoRA的模块
        module_name = name.split('.')[-1]  # 获取最后一级模块名

        if module_name in target_modules:
            if isinstance(module, nn.Linear):
                # 创建LoRA模块
                lora = LoRA(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=module.bias is not None
                ).to(module.weight.device)

                # 保存原始前向传播
                original_forward = module.forward

                # 包装新的前向传播
                def make_forward(orig_fwd, lora_module):
                    def forward(x):
                        # 原始输出 + LoRA输出
                        return orig_fwd(x) + lora_module(x)
                    return forward

                module.forward = make_forward(original_forward, lora)
                module.lora = lora
                module.lora_enabled = True

                print(f"[LoRA] Applied to {name}: {module.in_features} -> {module.out_features}, rank={rank}")

    # 标记模型已应用LoRA
    model.lora_applied = True
    model.lora_config = config

    return model


def load_lora_weights(
    model: nn.Module,
    path: str,
    device: str = "cpu"
) -> nn.Module:
    """
    加载LoRA权重

    参数:
        model: 带有LoRA的模型
        path: 权重文件路径
        device: 设备

    返回:
        加载权重后的模型
    """
    state_dict = torch.load(path, map_location=device)

    # 移除前缀
    state_dict = {
        k.replace("module.", "").replace("base_model.", ""): v
        for k, v in state_dict.items()
    }

    # 加载权重
    loaded_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 构建LoRA参数名
            lora_a_name = f"{name}.lora.lora_A"
            lora_b_name = f"{name}.lora.lora_B"

            if lora_a_name in state_dict and lora_b_name in state_dict:
                module.lora.lora_A.data = state_dict[lora_a_name].to(device)
                module.lora.lora_B.data = state_dict[lora_b_name].to(device)
                loaded_count += 1

    print(f"[LoRA] Loaded {loaded_count} LoRA modules from {path}")
    return model


def save_lora_weights(
    model: nn.Module,
    path: str
):
    """
    保存LoRA权重

    参数:
        model: 带有LoRA的模型
        path: 保存路径
    """
    state_dict = {}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 构建保存的键名
            lora_a_name = f"{name}.lora.lora_A"
            lora_b_name = f"{name}.lora.lora_B"

            state_dict[lora_a_name] = module.lora.lora_A.data
            state_dict[lora_b_name] = module.lora.lora_B.data

            # 如果有偏置
            if module.lora.bias is not None:
                bias_name = f"{name}.lora.bias"
                state_dict[bias_name] = module.lora.bias.data

    torch.save(state_dict, path)
    print(f"[LoRA] Saved {len(state_dict) // 2} LoRA modules to {path}")


def get_lora_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    获取LoRA参数信息

    参数:
        model: 带有LoRA的模型

    返回:
        参数字典
    """
    lora_params = 0
    base_params = 0

    for name, param in model.named_parameters():
        if 'lora' in name:
            lora_params += param.numel()
        else:
            base_params += param.numel()

    return {
        "lora_parameters": lora_params,
        "base_parameters": base_params,
        "total_parameters": lora_params + base_params,
        "trainable_ratio": lora_params / (base_params + lora_params) * 100
    }


def freeze_base_model(model: nn.Module) -> nn.Module:
    """
    冻结基础模型参数，只训练LoRA参数

    参数:
        model: 模型

    返回:
        冻结后的模型
    """
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False

    print("[LoRA] Base model parameters frozen")
    return model


def unfreeze_lora(model: nn.Module) -> nn.Module:
    """
    解冻LoRA参数

    参数:
        model: 模型

    返回:
        解冻后的模型
    """
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    print("[LoRA] LoRA parameters unfrozen")
    return model


# ========== Wormhole特定的LoRA工具 ==========

def apply_wormhole_lora(
    model: nn.Module,
    rank: int = 8,
    attention_lora_alpha: float = 16.0,
    mlp_lora_alpha: float = 8.0
) -> nn.Module:
    """
    为Wormhole-LM应用专门的LoRA配置

    为注意力层和MLP层使用不同的alpha值

    参数:
        model: WormholeForCausalLM模型
        rank: 通用rank
        attention_lora_alpha: 注意力层alpha
        mlp_lora_alpha: MLP层alpha

    返回:
        应用LoRA后的模型
    """
    for name, module in model.named_modules():
        module_name = name.split('.')[-1]

        # 确定alpha值
        if module_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            alpha = attention_lora_alpha
        elif module_name in ["gate_proj", "up_proj", "down_proj"]:
            alpha = mlp_lora_alpha
        else:
            alpha = rank

        if isinstance(module, nn.Linear):
            lora = LoRA(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                lora_alpha=alpha,
                lora_dropout=0.05,
                bias=module.bias is not None
            ).to(module.weight.device)

            original_forward = module.forward

            def make_forward(orig_fwd, lora_module):
                def forward(x):
                    return orig_fwd(x) + lora_module(x)
                return forward

            module.forward = make_forward(original_forward, lora)
            module.lora = lora

    model.lora_applied = True
    return model


# ========== 演示 ==========

def demo():
    """演示LoRA模块"""
    print("=" * 60)
    print("Wormhole-LM LoRA 模块演示")
    print("=" * 60)

    # 导入Wormhole模型
    from model_wormhole import WormholeForCausalLM, WormholeConfig

    # 创建模型
    print("\n[1] 创建Wormhole-LM模型...")
    config = WormholeConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        vocab_size=6400
    )
    model = WormholeForCausalLM(config)
    print(f"    模型创建完成: {sum(p.numel() for p in model.parameters())} 参数")

    # 应用LoRA
    print("\n[2] 应用LoRA...")
    model = apply_lora_to_model(model, rank=8, lora_alpha=16.0)
    lora_info = get_lora_parameters(model)
    print(f"    LoRA参数: {lora_info['lora_parameters']:,}")
    print(f"    基础参数: {lora_info['base_parameters']:,}")
    print(f"    可训练比例: {lora_info['trainable_ratio']:.2f}%")

    # 测试前向传播
    print("\n[3] 测试前向传播...")
    test_input = torch.randint(0, 6400, (2, 10))
    with torch.no_grad():
        output = model(test_input)
    print(f"    输入: {test_input.shape}")
    print(f"    输出logits: {output.logits.shape}")

    # 冻结基础模型
    print("\n[4] 冻结基础模型...")
    model = freeze_base_model(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    可训练参数: {trainable_params:,}")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
