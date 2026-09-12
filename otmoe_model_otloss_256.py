import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class PrototypeCrossAttention(nn.Module):
    def __init__(self, prototype_bank: torch.Tensor, embed_dim: int = 256,
                 num_heads: int = 8, token_dim: int = 64):
        super().__init__()
        self.register_buffer('prototype_bank', prototype_bank)  # [N, 256]
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.token_dim = token_dim

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(embed_dim, token_dim)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        B = query.size(0)
        keys = self.prototype_bank.unsqueeze(0).expand(B, -1, -1)
        values = keys
        attn_out, _ = self.cross_attn(query.unsqueeze(1), keys, values)
        token = attn_out.squeeze(1)
        token = self.proj(token)
        return token


class Enhanced_MOE_XLSR(nn.Module):
    """
    XLSR版双重域先验引导 MoE (类条件 6D 物理指纹匹配版)
    门控输入 (454维) = 共享特征 (256维) + 微观域Token (192维) + 类条件域匹配度 (6维)
    """
    def __init__(self, 
                 experts, 
                 emb_dim=256, 
                 hidden_dim=64, 
                 num_classes=2, 
                 freezing=True,
                 use_ot_gate: bool = True,
                 pcadt_token_dim: int = 64,
                 num_domains: int = 3):
        super(Enhanced_MOE_XLSR, self).__init__()
        
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_ot_gate = use_ot_gate
        self.pcadt_token_dim = pcadt_token_dim
        self.num_domains = num_domains

        # 初始化为 0，让反向传播自动学习压制跨域专家的 Logit 偏置
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts, num_classes))
        
        
        # 【修改 1】：门控输入维度确认为 256 + 6 + 192 = 454 维
        if use_ot_gate:
            gating_input_dim = emb_dim + (num_domains * 2) + (pcadt_token_dim * num_domains)
        else:
            gating_input_dim = emb_dim
        
        print(f"[Gate] 输入维度精确确认为: {gating_input_dim} 维 "
              f"(共享特征:{emb_dim}, 6D类条件匹配度:{num_domains*2 if use_ot_gate else 0}, PCADT Token:{pcadt_token_dim * num_domains if use_ot_gate else 0})")
        
        self.gating_network = nn.Sequential(
            nn.Linear(gating_input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),           
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),          
            nn.Linear(64, self.num_experts),
        )
        
        self.domain_centers: Optional[List[torch.Tensor]] = None
        self.domain_samples: Optional[List[torch.Tensor]] = None
        self.domain_labels: Optional[List[torch.Tensor]] = None
        self.real_normed_samples: Optional[List[torch.Tensor]] = None
        self.fake_normed_samples: Optional[List[torch.Tensor]] = None
        self.pcadt_modules: Optional[nn.ModuleList] = None
        
        if freezing:
            self.freeze_experts()
            
    def freeze_experts(self):
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False

    def unfreeze_experts_partial(self, unfreeze_mode='deep', ssl_frozen=True):
        for expert in self.experts:
            for name, param in expert.named_parameters():
                if ssl_frozen and 'ssl_model' in name:
                    param.requires_grad = False
                    continue
                
                if unfreeze_mode == 'none':
                    param.requires_grad = False
                elif unfreeze_mode == 'output':
                    if any(k in name for k in ['out_layer', 'drop']):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                elif unfreeze_mode == 'deep':
                    unlock_keywords = ['out_layer', 'HtrgGAT_layer_ST', 'GAT_layer', 'master', 'pos_S', 'attention', 'pooling']
                    if any(k in name for k in unlock_keywords):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                elif unfreeze_mode == 'half':
                    unlock_keywords = ['out_layer', 'HtrgGAT_layer_ST', 'GAT_layer', 'master', 'pos_S', 'attention', 'pooling', 'LL', 'encoder.4', 'encoder.5']
                    if any(k in name for k in unlock_keywords):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                elif unfreeze_mode == 'encoder_all':
                    unlock_keywords = ['out_layer', 'HtrgGAT_layer_ST', 'GAT_layer', 'master', 'pos_S', 'attention', 'pooling', 'LL', 'encoder', 'first_bn', 'first_bn1']
                    if any(k in name for k in unlock_keywords):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                elif unfreeze_mode == 'all_but_ssl':
                    if 'ssl_model' in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
        
        for idx, expert in enumerate(self.experts):
            total = sum(p.numel() for p in expert.parameters())
            trainable = sum(p.numel() for p in expert.parameters() if p.requires_grad)
            print(f"[Expert {idx}] 可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def set_domain_prototypes(self, centers: List[torch.Tensor], samples: List[torch.Tensor], labels: List[torch.Tensor]):
        """【修改 2】：拆分 Real 和 Fake 原型点云，加速伪造痕迹精确匹配"""
        device = next(self.parameters()).device
        self.domain_centers = centers
        self.domain_samples = [s.to(device) for s in samples]
        self.domain_labels = [l.to(device) for l in labels]
        
        self.real_normed_samples = []
        self.fake_normed_samples = []
        
        for s, l in zip(samples, labels):
            s_dev, l_dev = s.to(device), l.to(device)
            real_mask = (l_dev == 0)
            fake_mask = (l_dev == 1)
            self.real_normed_samples.append(F.normalize(s_dev[real_mask], p=2, dim=1))
            self.fake_normed_samples.append(F.normalize(s_dev[fake_mask], p=2, dim=1))

        if self.use_ot_gate and samples is not None:
            modules = []
            for domain_idx in range(self.num_domains):
                proto_bank = self.domain_samples[domain_idx]
                module = PrototypeCrossAttention(
                    proto_bank.clone().detach(),
                    embed_dim=self.emb_dim,
                    num_heads=8,
                    token_dim=self.pcadt_token_dim
                )
                modules.append(module)
            self.pcadt_modules = nn.ModuleList(modules)
            print(f"[PCADT] 已构建 {len(modules)} 个域交叉注意力模块，Token维度={self.pcadt_token_dim}")

    def forward(self, x, shared_emb: Optional[torch.Tensor] = None, return_embs=False):
        logits_list = []
        emb_list = []
        
        # 1. 专家独立计算真假 Logits
        for expert in self.experts:
            logits, emb = expert(x)
            logits_list.append(logits)
            emb_list.append(emb)
        
        # 2. 构造 454 维双重域先验门控特征
        if shared_emb is not None:
            gate_in_list = [shared_emb]  # [B, 256]
        else:
            gate_in_list = [emb_list[0]] 

        soft_target_tuple = None
        if self.use_ot_gate and shared_emb is not None:
            # (a) PCADT 域 Token [B, 192]
            if self.pcadt_modules is not None:
                domain_tokens = [module(shared_emb) for module in self.pcadt_modules]
                all_tokens = torch.cat(domain_tokens, dim=1)
                gate_in_list.append(all_tokens)
            else:
                gate_in_list.append(torch.zeros(shared_emb.size(0), 192, device=shared_emb.device))

            # (b) 【修改 3】：分别计算到 3 个域 Real 原型和 Fake 原型的 Top-100 相似度
            shared_emb_norm = F.normalize(shared_emb, p=2, dim=1)
            
            real_sims, fake_sims = [], []
            for r_norm, f_norm in zip(self.real_normed_samples, self.fake_normed_samples):
                # Real 匹配
                sim_r = torch.matmul(shared_emb_norm, r_norm.T)
                k_r = min(100, r_norm.size(0))
                real_sims.append(torch.topk(sim_r, k=k_r, dim=1)[0].mean(dim=1))
                
                # Fake 匹配 (关键解相干指纹！)
                sim_f = torch.matmul(shared_emb_norm, f_norm.T)
                k_f = min(100, f_norm.size(0))
                fake_sims.append(torch.topk(sim_f, k=k_f, dim=1)[0].mean(dim=1))
            
            s_real = torch.stack(real_sims, dim=1)  # [B, 3]
            s_fake = torch.stack(fake_sims, dim=1)  # [B, 3]
            
            # 6 维特征拼入门控网络 [B, 6]
            gate_feature_6d = torch.cat([s_real, s_fake], dim=1)
            gate_in_list.append(gate_feature_6d)
            
            # 依据物理幅值生成类条件软目标
            soft_target_real = F.softmax(s_real / 0.15, dim=-1).detach()
            soft_target_fake = F.softmax(s_fake / 0.15, dim=-1).detach()
            soft_target_tuple = (soft_target_real, soft_target_fake)

        # 拼接 256 + 192 + 6 = 454 维门控总输入
        gate_in = torch.cat(gate_in_list, dim=1)
        
        # 3. 门控网络计算路由权重
        gating_logits = self.gating_network(gate_in)
        gating_weights = F.softmax(gating_logits, dim=-1)

        # # 4. 概率空间加权融合专家输出
        # expert_probs = [F.softmax(lg, dim=-1) for lg in logits_list] 
        # stacked_probs = torch.stack(expert_probs, dim=-1)           # [B, 2, 3]
        
        # weighted_probs = torch.einsum('bn,bcn->bc', gating_weights, stacked_probs) # [B, 2]
        # weighted_logits = torch.log(weighted_probs + 1e-10)         # 取对数还原为 Logit 供 CE Loss 计算
        calibrated_logits = []
        for i in range(self.num_experts):
            calibrated_logits.append((logits_list[i] - self.expert_bias[i]))
        
        stacked_calibrated_logits = torch.stack(calibrated_logits, dim=-1) # [B, 2, 3]
        weighted_logits = torch.einsum('bn,bcn->bc', gating_weights, stacked_calibrated_logits) # [B, 2]

        if return_embs:
            return weighted_logits, gating_weights, [emb_list[0], emb_list[1], emb_list[2]]
        else:
            return weighted_logits, gating_weights, gating_logits, soft_target_tuple