import random
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional

# from OT_match_xlsr import build_ot_gate_input_xlsr, sinkhorn_matching_1d_xlsr
from OT_match_xlsr_batch_clust import build_ot_gate_input_xlsr, sinkhorn_matching_1d_xlsr

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def infer_domain_from_path(path: str) -> str:
    """从文件路径推断所属域（用于监控日志）"""
    if 'ASVspoof2019' in path:
        return 'ASV'
    elif 'CFAD' in path:
        return 'CFAD'
    elif 'Codecfake' in path:
        return 'Codec'
    else:
        return 'Unknown'

# def compute_adaptive_domain_soft_labels(shared_emb, domain_samples_list, 
#                                         fixed_tau=0.25, max_proto=5000, top_k=100):
#     """
#     利用 Top-100 最相似子类簇余弦距离生成精准的软域目标分布 q
#     【修改点】：fixed_tau = 0.25 配合 256 维强反差特征，主域目标概率锁定在 0.85 ~ 0.95
#     """
#     device = shared_emb.device
#     shared_emb_norm = F.normalize(shared_emb, p=2, dim=1) # [B, 256]
    
#     processed_sims = []
#     for samples in domain_samples_list:
#         samples = samples.to(device)
#         if samples.size(0) > max_proto:
#             idx = torch.randperm(samples.size(0), device=device)[:max_proto]
#             samples_sub = samples[idx]
#         else:
#             samples_sub = samples
            
#         samples_sub_norm = F.normalize(samples_sub, p=2, dim=1) # [N, 256]
        
#         # 1. 余弦相似度
#         cosine_sim = torch.matmul(shared_emb_norm, samples_sub_norm.T) # [B, N]
        
#         # 2. 提取 Top-100 最匹配子类簇相似度均值
#         k_val = min(top_k, samples_sub.size(0))
#         topk_sim = torch.topk(cosine_sim, k=k_val, dim=1, largest=True)[0]
#         sim = topk_sim.mean(dim=1) # [B]
#         processed_sims.append(sim)
    
#     s_raw = torch.stack(processed_sims, dim=1) # [B, 3]
    
#     # 3. 规范化与 Tanh 有界映射
#     s_centered = s_raw - s_raw.mean(dim=1, keepdim=True)
#     s_std = s_raw.std(dim=1, keepdim=True) + 1e-6
#     s_bounded = torch.tanh(s_centered / s_std) # [-1.0, 1.0]
    
#     # 4. Softmax 映射
#     q = F.softmax(s_bounded / fixed_tau, dim=1)
#     return q.detach()

def compute_adaptive_domain_soft_labels(shared_emb, domain_samples_list, 
                                        fixed_tau=0.30, max_proto=5000, top_k=100):
    """
    【修正版】：取消 Z-score 导致的跨域伪目标放大，直接使用物理余弦相似度幅值做 Softmax
    """
    device = shared_emb.device
    shared_emb_norm = F.normalize(shared_emb, p=2, dim=1) # [B, 256]
    
    processed_sims = []
    for samples in domain_samples_list:
        samples = samples.to(device)
        if samples.size(0) > max_proto:
            idx = torch.randperm(samples.size(0), device=device)[:max_proto]
            samples_sub = samples[idx]
        else:
            samples_sub = samples
            
        samples_sub_norm = F.normalize(samples_sub, p=2, dim=1) # [N, 256]
        
        # 1. 余弦相似度
        cosine_sim = torch.matmul(shared_emb_norm, samples_sub_norm.T) # [B, N]
        
        # 2. 提取 Top-100 最匹配子类簇相似度均值
        k_val = min(top_k, samples_sub.size(0))
        topk_sim = torch.topk(cosine_sim, k=k_val, dim=1, largest=True)[0]
        sim = topk_sim.mean(dim=1) # [B]
        processed_sims.append(sim)
    
    s_raw = torch.stack(processed_sims, dim=1) # [B, 3] 物理余弦幅值 [-1.0, 1.0]
    
    q = F.softmax(s_raw / fixed_tau, dim=1)
    return q.detach()

# 然后在 `valid_epoch_ot` 之前定义 compute_eer 函数
def compute_eer(labels, scores):
    """
    labels: 真实标签（0=real, 1=fake）
    scores: 预测为 fake 的概率（即 softmax 输出的第1列）
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100   # 返回百分比

# ==================== SDS分层动态采样器（新增） ====================

from collections import defaultdict
from typing import Dict, Tuple

class StratifiedDomainSampler:
    """
    分层域采样器：实现DOSS-Select思想，按域-子类二维结构截断采样
    适配MoE训练：确保3个数据集（域）的专家都能被充分激活
    """
    
    def __init__(self, 
                 domain_files: Dict[str, Dict[str, List[str]]],
                 domain_labels: Dict[str, Dict[str, int]],
                 domain_subclass_map: Dict[str, Dict[str, int]],
                 Nc_subclass: int = 2500,
                 rho: float = 0.25,
                 seed: int = 42):
        self.domain_files = domain_files
        self.domain_labels = domain_labels
        self.domain_subclass_map = domain_subclass_map
        self.Nc_subclass = Nc_subclass
        self.rho = rho
        self.seed = seed
        self.epoch = 0
        
        self._compute_domain_stats()
        
    def _compute_domain_stats(self):
        """计算每个域的子类分布"""
        self.domain_stats = {}
        for domain_name, subclass_dict in self.domain_files.items():
            stats = {
                'subclasses': {},
                'total_fake': 0,
                'total_real': 0,
            }
            for subclass_name, files in subclass_dict.items():
                subclass_id = self.domain_subclass_map[domain_name].get(subclass_name, 0)
                sample_label = self.domain_labels[domain_name].get(files[0], 1)
                is_real = (sample_label == 0)
                
                stats['subclasses'][subclass_name] = {
                    'id': subclass_id,
                    'count': len(files),
                    'is_real': is_real,
                    'files': files,
                }
                if is_real:
                    stats['total_real'] += len(files)
                else:
                    stats['total_fake'] += len(files)
            
            self.domain_stats[domain_name] = stats
            print(f"[SDS] Domain {domain_name}: real={stats['total_real']}, "
                  f"fake={stats['total_fake']}, subclasses={len(stats['subclasses'])}")
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def build_epoch_subset(self) -> List[str]:
        """构建本轮训练子集（DOSS-Select核心算法）"""
        np.random.seed(self.seed + self.epoch)
        
        selected_files = []
        
        for domain_name, stats in self.domain_stats.items():
            domain_selected = []
            fake_selected_count = 0
            
            # Step 1: 伪造子类截断采样
            for subclass_name, info in stats['subclasses'].items():
                if info['is_real']:
                    continue
                
                files = info['files']
                n_select = min(len(files), self.Nc_subclass)
                
                indices = np.random.permutation(len(files))[:n_select]
                selected = [files[i] for i in indices]
                domain_selected.extend(selected)
                fake_selected_count += n_select
            
            # Step 2: 真实子类动态配比
            target_real = int(fake_selected_count * self.rho)
            
            real_subclasses = [name for name, info in stats['subclasses'].items() 
                              if info['is_real']]
            
            if len(real_subclasses) == 0:
                continue
                
            real_files_all = []
            for subclass_name in real_subclasses:
                real_files_all.extend(stats['subclasses'][subclass_name]['files'])
            
            if len(real_files_all) > target_real:
                indices = np.random.permutation(len(real_files_all))[:target_real]  # 修复：统一用numpy
                real_selected = [real_files_all[i] for i in indices]
            else:
                if len(real_files_all) > 0:
                    repeats = (target_real // len(real_files_all)) + 1
                    extended = (real_files_all * repeats)[:target_real]
                    real_selected = extended
                else:
                    real_selected = []
            
            domain_selected.extend(real_selected)
            
            # 域内打乱
            perm = np.random.permutation(len(domain_selected))  # 修复：统一用numpy
            domain_shuffled = [domain_selected[i] for i in perm]
            selected_files.extend(domain_shuffled)
            
            print(f"[SDS] {domain_name}: fake={fake_selected_count}, "
                  f"real={len(real_selected)}, total={len(domain_selected)}")
        
        # 全局打乱
        global_perm = np.random.permutation(len(selected_files))  # 修复：统一用numpy
        final_subset = [selected_files[i] for i in global_perm]
        
        print(f"[SDS] Epoch {self.epoch} total subset: {len(final_subset)} samples")
        return final_subset




def safe_collate_fn(batch):
    """过滤None值，支持返回路径"""
    filtered = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if len(filtered) == 0:
        return None, None, None
    # 解包成三个列表并分别 collate
    xs, ys, paths = zip(*filtered)
    return torch.utils.data.dataloader.default_collate(xs), \
           torch.utils.data.dataloader.default_collate(ys), \
           list(paths)   # 路径保留为列表


def safe_collate_fn_eval(batch):
    """验证集collate - 支持返回路径，用于样本级监控"""
    filtered = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if len(filtered) == 0:
        return None, None, None
    # 解包成三个列表
    xs, ys, paths = zip(*filtered)
    return torch.utils.data.dataloader.default_collate(xs), \
           torch.utils.data.dataloader.default_collate(ys), \
           list(paths)   # 返回路径列表


# ==================== 【保留但不再调用】OT Matching Loss 定义 ====================
# 该函数保留在文件中，以便后续对比实验或消融研究使用，但当前训练循环不再调用。
# 原因：OT loss通过KL散度对gate weights施加强约束，会限制Transport Embedding的跨域泛化能力。

def ot_matching_loss_xlsr(gate_weights: torch.Tensor,
                          shared_emb: torch.Tensor,
                          domain_centers: List[torch.Tensor],
                          domain_samples: List[torch.Tensor],
                          domain_labels: List[torch.Tensor],
                          temperature: float = 0.2,
                          eps: float = 0.1,
                          max_samp: int = 5000):
    B, device = shared_emb.size(0), shared_emb.device
    
    ot_scores = []
    for ds, dl in zip(domain_samples, domain_labels):
        ds, dl = ds.to(device), dl.to(device)
        
        real_mask = (dl == 0)
        fake_mask = (dl == 1)
        real_s = ds[real_mask][:max_samp]
        fake_s = ds[fake_mask][:max_samp]
        
        if real_s.size(0) == 0 or fake_s.size(0) == 0:
            print(f"[OT-XLSR] Skip domain due to empty real/fake samples")
            continue
        
        def _wass_single(x: torch.Tensor, samp: torch.Tensor):
            if samp.size(0) == 0:
                return torch.tensor(0., device=device)
            cost = torch.cdist(x.unsqueeze(0), samp, p=2).squeeze(0)
            cost = cost / (cost.max().clamp_min(1e-12))
            a = torch.ones(1, device=device)
            b = torch.ones(samp.size(0), device=device) / samp.size(0)
            
            from ot import sinkhorn
            T = sinkhorn(a, b, cost.unsqueeze(0), reg=0.1, numItermax=100)
            return -(T * cost.unsqueeze(0)).sum()
        
        r_score = torch.stack([_wass_single(shared_emb[b], real_s) for b in range(B)])
        f_score = torch.stack([_wass_single(shared_emb[b], fake_s) for b in range(B)])
        
        total = real_s.size(0) + fake_s.size(0)
        w_real = real_s.size(0) / total
        w_fake = fake_s.size(0) / total
        ot_scores.append(w_real * r_score + w_fake * f_score)
    
    if len(ot_scores) == 0:
        return torch.tensor(0., device=device)
    
    ot_scores = torch.stack(ot_scores, dim=1) / temperature
    ot_target = F.softmax(ot_scores, dim=1)
    
    return F.kl_div(gate_weights.log(), ot_target, reduction='batchmean')


def pad(x, max_len=64600):
    x_len = x.shape[0] if isinstance(x, torch.Tensor) else len(x)
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    if isinstance(x, torch.Tensor):
        padded = x.repeat(num_repeats)[:max_len]
    else:
        padded = np.tile(x, num_repeats)[:max_len]
    return padded

class LoadTrainData_XLSR(Dataset):
    """
    XLSR版训练数据集 - 支持SDS分层动态采样
    """
    def __init__(self, list_IDs, labels, win_len=4.0375, fs=16000, 
                 sampler: Optional[StratifiedDomainSampler] = None):
        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = win_len
        self.fs = fs
        self.win_len_samples = int(win_len * fs)
        self.sampler = sampler  # SDS采样器
        
        # 如果有采样器，构建初始子集
        self.current_subset = list_IDs
        if self.sampler is not None:
            self.current_subset = self.sampler.build_epoch_subset()
        
        df = pd.DataFrame(labels.items(), columns=['path', 'label'])
        real_count = (df['label'] == 0).sum()
        fake_count = (df['label'] == 1).sum()
        print(f"[TrainData] Total pool: {len(labels)}, Real: {real_count}, Fake: {fake_count}")
        if self.sampler:
            print(f"[TrainData] SDS mode: Nc={sampler.Nc_subclass}, rho={sampler.rho}")

    def set_epoch(self, epoch: int):
        """每轮重建子集（SDS核心）"""
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
            self.current_subset = self.sampler.build_epoch_subset()
            print(f"[TrainData] Epoch {epoch} subset rebuilt: {len(self.current_subset)} samples")

    def __len__(self):
        return len(self.current_subset)
    
    def __getitem__(self, index):
        # SDS模式下使用current_subset
        track = self.current_subset[index]
        # ================================================
        try:
            suffix = Path(track).suffix.lower()
            if suffix == '.pt':
                x = torch.load(track, weights_only=True).float()
            else:
                import soundfile as sf
                import librosa
                x, sr = sf.read(track)
                if sr != self.fs:
                    x = librosa.resample(x, orig_sr=sr, target_sr=self.fs)
                x = torch.tensor(x, dtype=torch.float32)
        except Exception as e:
            print(f'[TrainData] Error loading {track}: {e}')
            return None, None, None

        x = torch.tensor(pad(x, self.win_len_samples), dtype=torch.float32)
        y = self.labels[track]
        return x, y, track 


class LoadEvalData_XLSR(Dataset):
    """
    XLSR版验证数据集 - 顺序遍历
    """
    def __init__(self, list_IDs, labels, win_len=4.0375, fs=16000):
        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = win_len
        self.fs = fs
        self.win_len_samples = int(win_len * fs)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        track = self.list_IDs[index]
        try:
            import soundfile as sf
            import librosa
            x, sr = sf.read(track)
            if sr != self.fs:
                x = librosa.resample(x, orig_sr=sr, target_sr=self.fs)
            x = torch.tensor(x, dtype=torch.float32)
        except Exception as e:
            print(f"[EvalData] Error loading {track}: {e}")
            return None, None, None

        if len(x) < self.win_len_samples:
            x = torch.nn.functional.pad(x, (0, self.win_len_samples - len(x)))
        
        start = (len(x) - self.win_len_samples) // 2
        x_win = x[start:start + self.win_len_samples]
        
        y = self.labels[track]
        return x_win, y, track


# ==================== 【核心修改】与ATADD对齐的训练函数：移除OT loss ====================

def train_epoch_ot(train_loader, 
                   model, 
                   optimizer, 
                   criterion, 
                   device,
                   shared_encoder,
                   domain_centers=None,
                   domain_samples=None,
                   domain_labels=None,
                   epoch=0,
                   total_epochs=100,
                   log_interval=50,
                   gate_log_interval=200,
                   gate_stats_history=None,
                   grad_clip_norm=0.5,
                   sample_monitor_interval=200):   # 新增参数
    """
    XLSR版OT-MoE训练epoch（与ATADD对齐，Transport Embedding版）
    【修改】完全移除OT辅助损失，仅使用CE loss驱动训练。
    Transport Embedding作为纯特征输入gate，不施加额外分布约束。
    """
    running_loss = 0
    num_correct = 0
    num_total = 0
    model.train()
    shared_encoder.eval()

    epoch_gate_sum = None
    epoch_gate_sq_sum = None
    gate_count = 0
    
    pbar = tqdm(train_loader, desc='[Train OT]')
    for batch_idx, batch_data in enumerate(pbar):
        # 由于 collate_fn 现在返回三个值
        if len(batch_data) == 3:
            batch_x, batch_y, batch_paths = batch_data
        else:
            # 兼容旧格式，但建议统一
            batch_x, batch_y = batch_data
            batch_paths = None
        
        if batch_x is None or batch_y is None:
            continue
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        with torch.no_grad():
            shared_emb = shared_encoder(batch_x)
        
        # logits, gate_weights = model(batch_x, shared_emb=shared_emb)
        logits, gate_weights, gate_logits, soft_targets = model(batch_x, shared_emb=shared_emb)


        # ========== 样本级监控（新增） ==========
        if batch_paths is not None and batch_idx % sample_monitor_interval == 0:
                batch_size_act = batch_x.size(0)
                num_samples = min(5, batch_size_act)
                rand_indices = torch.randperm(batch_size_act)[:num_samples].tolist()
                
                for idx in rand_indices:
                    path = batch_paths[idx]
                    domain_name = infer_domain_from_path(path)
                    gw = gate_weights[idx].cpu().tolist()
                    gw_str = ', '.join([f'{v:.3f}' for v in gw])
                    label_val = batch_y[idx].item()
                    label_str = "Fake" if label_val == 1 else "Real"
                    
                    print(f"[SampleMonitor] E:{epoch:02d} B:{batch_idx:04d} | "   # 标明 SampleMonitor 和 Epoch
                          f"Domain:{domain_name:<5} | Label:{label_str:<4}({label_val}) | "
                          f"Gate:[{gw_str}]")
        # ====================================

        # 门控权重统计记录（仅监控，无梯度回传）
        if gate_stats_history is not None and batch_idx % gate_log_interval == 0:
            with torch.no_grad():
                gate_mean = gate_weights.mean(0).cpu()
                gate_std = gate_weights.std(0).cpu()
                gate_max = gate_weights.max(0)[0].cpu()
                
                dominant_expert = gate_weights.argmax(dim=1)
                activation_ratio = torch.tensor([
                    (dominant_expert == i).float().mean().item() 
                    for i in range(gate_weights.size(1))
                ])
                
                stats = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'gate_mean': gate_mean.tolist(),
                    'gate_std': gate_std.tolist(),
                    'gate_max': gate_max.tolist(),
                    'activation_ratio': activation_ratio.tolist(),
                }
                
                # 仍计算OT score用于监控gate-OT相关性（不用于loss）
                if model.use_ot_gate and hasattr(model, 'domain_samples'):
                    from OT_match_xlsr_batch_clust import build_ot_gate_input_xlsr
                    ot_scores = build_ot_gate_input_xlsr(
                        shared_emb, model.domain_centers,
                        model.domain_samples, model.domain_labels,
                        max_samp=5000
                    )
                    ot_mean = ot_scores.mean(0).cpu()
                    ot_std = ot_scores.std(0).cpu()
                    stats['ot_mean'] = ot_mean.tolist()
                    stats['ot_std'] = ot_std.tolist()
                    
                    correlation = torch.nn.functional.cosine_similarity(
                        gate_weights, ot_scores, dim=1
                    ).mean().item()
                    stats['gate_ot_correlation'] = correlation
                
                gate_stats_history.append(stats)
                
                if batch_idx % (gate_log_interval) == 0:
                    print(f"[GateStats] Epoch:{epoch} Batch:{batch_idx} "
                          f"Gate:[{', '.join([f'{x:.2f}' for x in gate_mean.tolist()])}] "
                          f"Corr:{stats.get('gate_ot_correlation', 0):.3f}")
        
        # 累加epoch级统计
        if epoch_gate_sum is None:
            epoch_gate_sum = gate_weights.sum(0).detach()
            epoch_gate_sq_sum = (gate_weights ** 2).sum(0).detach()
        else:
            epoch_gate_sum += gate_weights.sum(0).detach()
            epoch_gate_sq_sum += (gate_weights ** 2).sum(0).detach()
        gate_count += batch_size
        

        # total_loss = criterion(logits, batch_y)
        # 2. 计算分类主损失
        ce_loss = criterion(logits, batch_y)

        # 【修改 4】：动态根据 batch_y 标签选择对应的类条件软目标！
        if soft_targets is not None:
            soft_target_real, soft_target_fake = soft_targets
            
            # 如果 batch_y == 0 选 soft_target_real，如果 batch_y == 1 选 soft_target_fake
            soft_target_q = torch.where(
                (batch_y == 0).unsqueeze(1),
                soft_target_real,
                soft_target_fake
            )
            domain_loss = F.cross_entropy(gate_logits, soft_target_q, reduction='mean')
        else:
            domain_loss = torch.tensor(0.0, device=device)

        # 4. 组合总损失（λ=0.1 是黄金比例，可根据监控微调）
        lambda_domain = 0.1
        total_loss = ce_loss + lambda_domain * domain_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        # 与ATADD对齐：梯度裁剪0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        
        _, pred = logits.max(1)
        num_correct += (pred == batch_y).sum().item()
        running_loss += total_loss.item() * batch_size
        
        pbar.set_postfix({
            'loss': f'{total_loss.item():.3f}',
            'acc': f'{(num_correct/num_total)*100:.1f}%'
        })
    
    if gate_count > 0 and gate_stats_history is not None:
        epoch_gate_mean = epoch_gate_sum / gate_count
        epoch_gate_var = (epoch_gate_sq_sum / gate_count) - (epoch_gate_mean ** 2)
        epoch_stats = {
            'epoch': epoch,
            'type': 'epoch_summary',
            'gate_mean': epoch_gate_mean.cpu().tolist(),
            'gate_var': epoch_gate_var.cpu().tolist(),
            'total_samples': gate_count
        }
        gate_stats_history.append(epoch_stats)
    
    avg_loss = running_loss / num_total if num_total > 0 else 0
    accuracy = (num_correct / num_total) * 100 if num_total > 0 else 0
    return avg_loss, accuracy


def valid_epoch_ot(dev_loader, model, criterion, device, shared_encoder,
                   sample_monitor_interval=400): 
    """
    验证集评估，支持样本级门控监控
    """
    running_loss = 0
    num_correct = 0
    num_total = 0
    model.eval()
    shared_encoder.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dev_loader, desc='[Valid OT]')
        for batch_idx, batch_data in enumerate(pbar):
            # ---------- 解包数据（支持路径） ----------
            if len(batch_data) == 3:
                batch_x, batch_y, batch_paths = batch_data
            else:
                # 兼容旧格式
                batch_x, batch_y = batch_data
                batch_paths = None
            
            if batch_x is None or batch_y is None:
                continue
            
            batch_size = batch_x.size(0)
            num_total += batch_size
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).long().to(device)
            
            shared_emb = shared_encoder(batch_x)
            
            # 模型前向（返回 logits, gate_weights, gate_logits）
            batch_out, gate_weights, _, _ = model(batch_x, shared_emb=shared_emb)
            if isinstance(batch_out, tuple):
                batch_out = batch_out[0]
            
            # ========== 验证集样本级监控（新增） ==========
            if batch_paths is not None and batch_idx % sample_monitor_interval == 0:
                batch_size_act = batch_x.size(0)
                num_samples = min(3, batch_size_act)
                rand_indices = torch.randperm(batch_size_act)[:num_samples].tolist()
                
                for idx in rand_indices:
                    path = batch_paths[idx]
                    domain_name = infer_domain_from_path(path)
                    gw = gate_weights[idx].cpu().tolist()
                    gw_str = ', '.join([f'{v:.3f}' for v in gw])
                    label_val = batch_y[idx].item()
                    label_str = "Fake" if label_val == 1 else "Real"
                    
                    print(f"[ValSampleMonitor] B:{batch_idx:04d} | "
                          f"Domain:{domain_name:<5} | Label:{label_str:<4}({label_val}) | "
                          f"Gate:[{gw_str}]")
            # ===========================================
            
            # 计算损失和准确率
            batch_loss = criterion(batch_out, batch_y)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum().item()
            running_loss += batch_loss.item() * batch_size
            
            # 收集分数
            probs = F.softmax(batch_out, dim=1)[:, 1].cpu().numpy()
            all_scores.extend(probs)
            all_labels.extend(batch_y.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{batch_loss.item():.3f}'})
    
    avg_loss = running_loss / num_total if num_total > 0 else 0
    accuracy = (num_correct / num_total) * 100 if num_total > 0 else 0
    eer = compute_eer(all_labels, all_scores) if len(all_labels) > 0 else 100.0
    
    return avg_loss, accuracy, eer


def eval_model_ot(model, data_loader, save_path, device, shared_encoder):
    model.eval()
    shared_encoder.eval()
    
    total_gating_weights = None
    batch_count = 0
    fname_list, pred_list, label_list = [], [], []

    with torch.no_grad():
        for batch_x, batch_y, utt_id in tqdm(data_loader, total=len(data_loader)):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            shared_emb = shared_encoder(batch_x)
            
            batch_out, gate_weights, _, _ = model(batch_x, shared_emb=shared_emb)
            batch_out = F.softmax(batch_out, dim=1)
            batch_score = batch_out[:, 0].cpu().numpy().ravel()

            if total_gating_weights is None:
                total_gating_weights = gating_weights.sum(0).cpu()
            else:
                total_gating_weights += gating_weights.sum(0).cpu()
            batch_count += gating_weights.size(0)

            fname_list.extend(utt_id)
            pred_list.extend(batch_score.tolist())
            label_list.extend(batch_y.cpu().numpy().tolist())

    with open(save_path, 'a+') as fh:
        for f, s, l in zip(fname_list, pred_list, label_list):
            fh.write(f'{f} {s} {l}\n')

    if batch_count:
        avg_weights = total_gating_weights / batch_count
        np.save(save_path.replace('.txt', '_gate.npy'), avg_weights)

    print(f'Scores saved to {save_path}')