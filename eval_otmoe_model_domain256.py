#!/usr/bin/env python3
"""
XLSR-OT-MoE 模型测试代码
基于 SSL-AASIST 的 OT-MoE 实现，适配 160 维特征
支持 ASVspoof2019、CFAD、Codecfake、ITW 数据集

与单专家测试的关键区别：
1. 需要同时加载：共享编码器 + 3个预训练专家 + MoE门控 + 域原型
2. 前向传播：shared_encoder提取嵌入 → MoE融合专家输出（OT门控加权）
3. 分数约定：使用 real 概率 (probs[:, 0])，高分=更可能是真人
   这与 tDCF 的 "higher score = stronger bonafide support" 约定一致
4. 额外输出每个测试集的平均门控权重，用于分析专家路由行为

"""

import os
import sys
sys.path.insert(0, '/data1/tjj/XLSR-MOE/FM')
sys.path.insert(0, '/data1/tjj/XLSR-MOE')

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==================== 修改点 1：导入正确的模型文件与类 ====================
from model_SSL_Enhanced import SSLAASISTExpert
from otmoe_model_otloss_256 import Enhanced_MOE_XLSR
from utils import read_yaml, seed_everything
import eval_metric_LA as em
import argparse
# 保持与训练 100% 完全一致的 256D 特征提取器（正交矩阵种子42，微调XLSR Layer-12）
class NativeXLSR256DExtractor(nn.Module):
    """提取微调后 XLSR-300M 第 12 层中间声学/信道域特征 + 256维固定正交投影"""
    def __init__(self, 
                 ckpt_path='/data1/tjj/XLSR-MOE/xlsr2_300m.pt', 
                 mixed_aasist_ckpt='/data1/tjj/XLSR-MOE/models_ssl_youhua/SSLAASIST_Mixed_best.pth',
                 device='cuda'):
        super().__init__()
        import fairseq
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.ssl_model = model[0].to(device)
        self.device = device
        
        if mixed_aasist_ckpt and os.path.exists(mixed_aasist_ckpt):
            ckpt = torch.load(mixed_aasist_ckpt, map_location='cpu')
            state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
            ssl_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('ssl_model.model.'):
                    ssl_state_dict[k.replace('ssl_model.model.', '')] = v
                elif k.startswith('ssl_model.'):
                    ssl_state_dict[k.replace('ssl_model.', '')] = v
            self.ssl_model.load_state_dict(ssl_state_dict, strict=False)

        self.ssl_model.eval()
        for param in self.ssl_model.parameters():
            param.requires_grad = False
            
        # 种子 42 + 正交矩阵保距投影，确保与训练特征空间 100% 对齐
        torch.manual_seed(42)
        linear_layer = nn.Linear(1024, 256)
        nn.init.orthogonal_(linear_layer.weight)
        nn.init.zeros_(linear_layer.bias)

        self.proj = nn.Sequential(
            linear_layer,
            nn.LayerNorm(256)
        ).to(device).eval()
        
        for param in self.proj.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        if x.ndim == 3:
            x = x.squeeze(-1)
        x = x.to(self.device)
        res = self.ssl_model(x, mask=False, features_only=True, layer=12)
        ssl_out = res['x'] # [B, Frame, 1024]
        feat_1024 = ssl_out.mean(dim=1)
        feat_256 = self.proj(feat_1024)
        return F.normalize(feat_256, p=2, dim=1)
# ====================================================================

# ==================== 基础配置 ====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 使用与训练相同的配置文件（从中读取模型路径、域缓存路径等）
cfg_path = Path('/data1/tjj/XLSR-MOE/OTMOE/OTM_MOE/train_config_otmoe_doss.yaml')
cfg = read_yaml(str(cfg_path))
seed_everything(cfg.get('seed', 42))

# 结果保存目录
# save_root = Path('/data1/tjj/XLSR-MOE/OTMOE/ot_results/domainmixxlsr0.2+otlossnoz256_12_freeze')
save_root = Path('/data1/tjj/XLSR-MOE/OTMOE/ot_results/domainmixxlsr0.2+otlossnoz256_12')
save_root.mkdir(parents=True, exist_ok=True)
log_file = save_root / 'otmoe_eval.log'

# ASV评分文件路径（仅 ASVspoof2019 LA eval 计算 tDCF 时需要）
asv_score_path = '/data1/tjj/LCNN_moe/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt'

# 测试时建议的batch_size（OT-MoE有4个SSL模型，内存占用约为单专家的4倍）
TEST_BATCH_SIZE = 16


def log(msg):
    """写入日志并打印"""
    t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'{t}  {msg}'
    print(line)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ==================== 评估数据集 ====================
class OTMoEEvalDataset(Dataset):
    """
    OT-MoE 评估数据集
    与训练时 valid_epoch_ot 使用的 LoadEvalData_XLSR 保持一致：
    - 中心裁剪（非重复填充）
    - 输入：完整路径列表
    """
    def __init__(self, paths, labels, win_len=4.0375, fs=16000):
        self.paths = paths
        self.labels = labels
        self.win_len_samples = int(win_len * fs)
        self.fs = fs

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            import soundfile as sf
            import librosa
            x, sr = sf.read(path)
            if sr != self.fs:
                x = librosa.resample(x, orig_sr=sr, target_sr=self.fs)
            x = torch.tensor(x, dtype=torch.float32)
        except Exception as e:
            log(f'[ERROR] 加载失败 {path}: {e}')
            return None, None, None

        # 中心裁剪（与训练验证阶段一致）
        if len(x) < self.win_len_samples:
            x = F.pad(x, (0, self.win_len_samples - len(x)))
        start = (len(x) - self.win_len_samples) // 2
        x = x[start:start + self.win_len_samples]

        fname = Path(path).stem
        y = self.labels[path]
        return x, y, fname


def safe_collate_eval(batch):
    """过滤加载失败的样本"""
    filtered = [b for b in batch if b[0] is not None]
    if len(filtered) == 0:
        return None, None, None
    return torch.utils.data.default_collate(filtered)


# ==================== 模型加载 ====================
def load_otmoe_model(cfg):
    log('=' * 60)
    log('加载 OT-MoE 模型组件...')

    ssl_path = cfg.get('ssl_path', '/data1/tjj/XLSR-MOE/xlsr2_300m.pt')
    mixed_aasist_ckpt = cfg.get('mixed_aasist_ckpt', '/data1/tjj/XLSR-MOE/models_ssl_youhua/SSLAASIST_Mixed_best.pth')

    # ---------- 组件1: 256D 共享编码器（完全与训练对齐） ----------
    shared_encoder = NativeXLSR256DExtractor(
        ckpt_path=ssl_path,
        mixed_aasist_ckpt=mixed_aasist_ckpt,
        device=device
    ).to(device).eval()

    log(f'[1/4] 共享编码器: NativeXLSR256DExtractor 已加载 (256维微调空间, FROZEN)')

    # ---------- 组件2: 3个预训练专家（冻结） ----------
    expert_paths = [
        cfg['expert_1_path'],
        cfg['expert_2_path'],
        cfg['expert_3_path']
    ]

    experts = []
    for i, path in enumerate(expert_paths, 1):
        expert = SSLAASISTExpert(
            device=device,
            ssl_path=ssl_path,
            return_emb=True
        )
        ckpt = torch.load(path, map_location=device)
        if 'model_state' in ckpt:
            expert.load_state_dict(ckpt['model_state'])
        else:
            expert.load_state_dict(ckpt)
        expert = expert.to(device).eval()
        for param in expert.parameters():
            param.requires_grad = False
        experts.append(expert)
        n_p = sum(p.numel() for p in expert.parameters())
        log(f'[2/4] 专家 {i}: {Path(path).name} ({n_p:,} params, FROZEN)')

    # ---------- 组件3: MoE 门控模型 (emb_dim=256) ----------
    moe_model = Enhanced_MOE_XLSR(
        experts=experts,
        emb_dim=256,
        hidden_dim=64,
        num_classes=2,
        freezing=True,
        use_ot_gate=True,
        pcadt_token_dim=64,
        num_domains=3
    ).to(device)

    # ---------- 组件4: 域原型注入 ----------
    cache_dir = cfg.get('domain_cache_dir', '/data1/tjj/XLSR-MOE/FM/domainmixxlsr_cache')
    n_samples = cfg.get('n_prototype_samples', 5000)

    domain_names = ["ASVspoof2019_generated", "CFAD_generated", "Codecfake_generated"]
    centers, samples_proto, labels_proto = [], [], []

    for dname in domain_names:
        ck_path = os.path.join(cache_dir, f'{dname}_n{n_samples}.pth')
        if not os.path.exists(ck_path):
            raise FileNotFoundError(f'域原型缓存不存在: {ck_path}\n请先确认路径!')
        cache = torch.load(ck_path, map_location='cpu')
        centers.append(cache['center'].to(device))
        samples_proto.append(cache['samples'].to(device))
        labels_proto.append(cache['labels'].to(device))

    # 注入原型，创建 PCADT 模块与解相干点云
    moe_model.set_domain_prototypes(
        centers=centers, samples=samples_proto, labels=labels_proto
    )
    moe_model.to(device)
    log(f'[3/4] 域原型已注入，PCADT模块与点云拆分就绪')

    # ==================== 目标测试模型权重路径 ====================
    # moe_best_path = '/data1/tjj/XLSR-MOE/OTMOE/ot_models_doss/deep/batch/domainmixxlsr_freeze+otlossnoz/XLSR_OT_MOE_best.pth'
    moe_best_path = '/data1/tjj/XLSR-MOE/OTMOE/ot_models_doss/deep/batch/domainmixxlsr0.2+otlossnoz/XLSR_OT_MOE_best.pth'
    if not os.path.exists(moe_best_path):
        raise FileNotFoundError(f'MoE最佳模型不存在: {moe_best_path}')

    moe_model.load_state_dict(torch.load(moe_best_path, map_location=device))
    moe_model.eval()

    moe_total = sum(p.numel() for p in moe_model.parameters())
    log(f'[4/4] MoE门控与专家权重已成功加载: {moe_best_path}')
    log(f'      模型总参数量: {moe_total:,}')

    return shared_encoder, moe_model


# ==================== 通用协议/目录读取函数 ====================
def load_proto(proto, root, suffix):
    """
    读取协议文件或根据目录结构加载样本
    """
    # ---------------- 模式 1: CD-ADD 零样本 TTS 目录递归扫描 ----------------
    if proto == 'DIR_CD_ADD':
        file_list, label_dict = [], {}
        for p in sorted(Path(root).rglob('*.wav')):
            file_list.append(str(p))
            label_dict[str(p)] = 0 if p.name.lower() == 'real.wav' else 1
        log(f'[CD-ADD Scan] 从 {root} 成功加载: Real={sum(1 for v in label_dict.values() if v==0)}, Fake={sum(1 for v in label_dict.values() if v==1)}')
        return file_list, label_dict

    # ---------------- 模式 2: AI4T 目录结构扫描 ----------------
    if proto == 'DIR_REAL_FAKE':
        real_dir = Path(root) / 'real_wav'
        fake_dir = Path(root) / 'fake_wav'
        file_list, label_dict = [], {}
        if real_dir.exists():
            for p in sorted(real_dir.glob(f'*{suffix}')):
                file_list.append(str(p))
                label_dict[str(p)] = 0
        if fake_dir.exists():
            for p in sorted(fake_dir.glob(f'*{suffix}')):
                file_list.append(str(p))
                label_dict[str(p)] = 1
        log(f'[DIR Scan] 从 {root} 成功加载: Real={sum(1 for v in label_dict.values() if v==0)}, Fake={sum(1 for v in label_dict.values() if v==1)}')
        return file_list, label_dict

    # ---------------- 模式 3: SONAR 多模型基准扫描 ----------------
    if proto == 'DIR_SONAR':
        real_dir = Path(root) / 'real_samples'
        file_list, label_dict = [], {}
        if real_dir.exists():
            for p in sorted(real_dir.glob(f'*{suffix}')):
                file_list.append(str(p))
                label_dict[str(p)] = 0
        for sub_dir in sorted(Path(root).iterdir()):
            if sub_dir.is_dir() and sub_dir.name != 'real_samples' and not sub_dir.name.startswith('.'):
                for p in sorted(sub_dir.glob(f'*{suffix}')):
                    file_list.append(str(p))
                    label_dict[str(p)] = 1
        log(f'[SONAR Scan] 从 {root} 成功加载: Real={sum(1 for v in label_dict.values() if v==0)}, Fake={sum(1 for v in label_dict.values() if v==1)}')
        return file_list, label_dict

    # ---------------- 模式 4: SpeechFake CSV 协议 ----------------
    if proto.endswith('.csv'):
        df = pd.read_csv(proto)
        if 'file' in df.columns and 'label' in df.columns:
            label_map = {'bonafide': 0, 'real': 0, 'bona-fide': 0, 'genuine': 0, 'spoof': 1, 'fake': 1}
            df['label'] = df['label'].map(label_map)

            def find_real_path(rel_file):
                p1 = Path(root) / rel_file
                if p1.exists():
                    return str(p1)
                p2 = Path(root) / 'BD' / rel_file
                if p2.exists():
                    return str(p2)
                p3 = Path(root) / 'Real' / rel_file
                if p3.exists():
                    return str(p3)
                return str(p1)

            df['path'] = df['file'].apply(find_real_path)
            exists = df['path'].apply(os.path.exists)
            if not exists.all():
                log(f'[WARN] {(~exists).sum()} 个文件不存在，已跳过 (成功加载: {exists.sum()} 个)')
                df = df[exists]
            else:
                log(f'[OK] 全部 {len(df)} 个样本路径成功匹配！')
            return df['path'].tolist(), dict(zip(df['path'], df['label']))

    # ---------------- 模式 5: 标准 TXT 协议 ----------------
    df = pd.read_csv(proto, sep=r'\s+', header=None, engine='python')
    if df.shape[1] == 2:  # 2列 (ADD2023)
        df.columns = ['fname', 'label']
        def parse_col2(val):
            v_str = str(val).strip()
            if v_str.isdigit():
                return 0 if v_str == '0' else 1
            return v_str
        df['label'] = df['label'].apply(parse_col2)
    elif df.shape[1] == 3:  # 3列 (CFAD / ADD2022)
        df.columns = ['fname', 'label', 'attack']
    elif df.shape[1] == 5:  # 5列 (ASV2019)
        df.columns = ['spk', 'fname', '_', 'attack', 'key']
        df['label'] = df['key']
    elif df.shape[1] == 6:  # 6列 (Codecfake)
        df.columns = ['vcoder', '_', 'fname', '_', '_', 'key']
        df['label'] = df['key']
    elif df.shape[1] == 7:  # 7列
        df.columns = ['spk', 'fname', '_', '_', '_', '_', 'key']
        df['label'] = df['key']
    elif df.shape[1] == 8:  # 8列 (精准适配 ASVspoof2021 trial_metadata: - LA_E_xxx - - - spoof - eval)
        df['fname'] = df.iloc[:, 1]
        df['label'] = df.iloc[:, 5]
    elif df.shape[1] >= 9:  # ASVspoof 5 格式
        df['fname'] = df.iloc[:, 1]
        df['label'] = df.iloc[:, 8]
    else:
        raise ValueError(f'不支持的协议列数: {df.shape[1]}，文件: {proto}')

    # 统一标签映射
    label_map = {
        'bonafide': 0, 'real': 0, 'bona-fide': 0, 'genuine': 0,
        'spoof': 1, 'fake': 1,
        0: 0, 1: 1, '0': 0, '1': 1
    }
    df['label'] = df['label'].map(label_map)

    invalid = df['label'].isna()
    if invalid.any():
        df = df[~invalid]

    # 智能后缀与路径拼装
    def build_path(x):
        fname_str = str(x)
        if fname_str.endswith(('.wav', '.flac', '.pt')):
            return str(Path(root) / fname_str)
        # 自动探测 .wav 或 .flac 是否存在
        p_wav = Path(root) / (fname_str + '.wav')
        if p_wav.exists():
            return str(p_wav)
        p_flac = Path(root) / (fname_str + '.flac')
        if p_flac.exists():
            return str(p_flac)
        return str(Path(root) / (fname_str + suffix))

    df['path'] = df['fname'].apply(build_path)
    exists = df['path'].apply(os.path.exists)
    if not exists.all():
        log(f'[WARN] {(~exists).sum()} 个文件不存在，已跳过')
        df = df[exists]

    return df['path'].tolist(), dict(zip(df['path'], df['label']))

# ==================== 单子集测试 ====================
def run_subset(name, list_IDs, labels,
              shared_encoder, moe_model,
              trial_proto=None, need_tdcf=False):
    """
    测试单个子集

    分数约定说明（重要）：
    - 本代码输出 real 概率 probs[:, 0]，高分 = 更可能是真人
    - 这与 eval_metric_LA.compute_tDCF 的约定一致：
      "higher detection scores values are assumed to indicate
       stronger support for the bona fide hypothesis"
    - 单专家测试代码使用 fake 概率 probs[:, 1]，EER值相同，
      但 tDCF 结果会不正确
    """
    total = len(list_IDs)
    log(f'>>>> 开始测试 {name}  (N={total})')

    # 创建结果目录
    score_file = save_root / name / f'{name}_otmoe.txt'
    score_file.parent.mkdir(exist_ok=True)

    # 创建数据加载器
    dataset = OTMoEEvalDataset(
        paths=list_IDs,
        labels=labels,
        win_len=cfg.get('win_len', 4.0375),
        fs=cfg.get('fs', 16000)
    )

    loader = DataLoader(
        dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.get('num_workers', 8),
        pin_memory=True,
        collate_fn=safe_collate_eval
    )

    # ---------- 推理 ----------
    fout = open(score_file, 'w', encoding='utf-8')
    t0 = time.time()
    total_batches = len(loader)

    # 累计门控权重（用于分析专家路由行为）
    gate_weights_sum = None
    gate_count = 0

    with torch.no_grad():
        for idx, (batch_x, batch_y, utt_ids) in enumerate(
            tqdm(loader, desc=f'[{name}]', ncols=100), 1
        ):
            if batch_x is None:
                continue

            batch_x = batch_x.to(device)
            batch_y_cpu = batch_y  # 保留CPU副本用于统计
            batch_size_actual = batch_x.size(0)

            # Step 1: 共享编码器提取 160 维嵌入（用于 OT 门控）
            shared_emb = shared_encoder(batch_x)

            # Step 2: MoE 前向传播（注意训练模型返回4个值）
            # ==================== 修改点 3：解包4个返回值 ====================
            logits, gate_weights, _, _ = moe_model(batch_x, shared_emb=shared_emb)
            # =================================================================

            # Step 3: Softmax → 取 real 概率作为分数
            probs = F.softmax(logits, dim=1)
            scores = probs[:, 0].cpu().numpy()  # real概率，高分=真人

            # 累计门控权重
            if gate_weights_sum is None:
                gate_weights_sum = gate_weights.sum(0).cpu()
            else:
                gate_weights_sum += gate_weights.sum(0).cpu()
            gate_count += gate_weights.size(0)

            # 写入分数文件：fname real_prob bonafide/spoof
            for uid, sc, ky in zip(utt_ids, scores, batch_y_cpu):
                key_str = 'bonafide' if ky == 0 else 'spoof'
                fout.write(f'{uid} {sc:.6f} {key_str}\n')

            # 进度日志（每10%记录一次）
            if idx % max(1, total_batches // 10) == 0 or idx == total_batches:
                elapsed = time.time() - t0
                progress = idx / total_batches
                eta = elapsed / progress * (1 - progress) if progress > 0 else 0
                processed = min(idx * TEST_BATCH_SIZE, total)
                log(f'{name}  {processed}/{total}  {progress*100:3.0f}%  '
                    f'elapsed={elapsed // 60:.0f}m{elapsed % 60:.0f}s  '
                    f'ETA={eta // 60:.0f}m{eta % 60:.0f}s')

    fout.close()
    log(f'[OK] 分数文件 -> {score_file}')

    # ---------- 保存门控权重统计 ----------
    if gate_count > 0:
        avg_gate = (gate_weights_sum / gate_count).numpy()
        gate_path = save_root / name / f'{name}_otmoe_gate.npy'
        np.save(gate_path, avg_gate)

        # 解读门控分布（假设专家顺序: ASVspoof / CFAD / Codecfake）
        domain_names_short = ['ASV', 'CFAD', 'Codec']
        gate_info = '  '.join([
            f'{domain_names_short[i]}={avg_gate[i]:.3f}' for i in range(3)
        ])
        dominant = domain_names_short[np.argmax(avg_gate)]
        entropy = -np.sum(avg_gate * np.log(avg_gate + 1e-10))
        log(f'[GATE] {name} 平均门控: {gate_info}  主导={dominant}  熵={entropy:.3f}')

    # ---------- 计算 EER / min-tDCF ----------
    eer_txt = save_root / name / f'{name}_otmoe_eer.txt'

    df = pd.read_csv(score_file, sep=' ', header=None, names=['fname', 'score', 'key'])
    bona = df[df.key == 'bonafide']['score'].values.astype(np.float64)
    spoof = df[df.key == 'spoof']['score'].values.astype(np.float64)

    if len(bona) == 0 or len(spoof) == 0:
        log(f'[ERROR] {name} 缺少 bonafide 或 spoof 样本，无法计算EER')
        return None

    # EER计算（bona=高real概率, spoof=低real概率 → 符合target/nontarget约定）
    eer, th = em.compute_eer(bona, spoof)

    # tDCF计算（仅 ASVspoof2019 LA eval）
    if need_tdcf and trial_proto and os.path.exists(asv_score_path):
        try:
            asv_key = pd.read_csv(trial_proto, sep=' ', header=None)
            asv_scr = pd.read_csv(asv_score_path, sep=' ', header=None)

            asv_key.columns = ['spk', 'utt', '_', 'attack', 'key']
            asv_scr.columns = ['key', 'target', 'score']

            tar = asv_scr[asv_scr['target'] == 'target']['score'].astype(float)
            non = asv_scr[asv_scr['target'] == 'nontarget']['score'].astype(float)
            spf = asv_scr[asv_scr['key'] == 'spoof']['score'].astype(float)

            # 用CM的EER阈值作为ASV联合阈值
            ret = em.obtain_asv_error_rates(tar, non, spf, th)
            Pfa, Pmiss, Pmiss_sp, _ = ret

            # ASVspoof 2019 官方 tDCF 成本模型
            cost = {
                'Pspoof': 0.05,
                'Ptar': 0.95 * 0.99,
                'Pnon': 0.95 * 0.01,
                'Cmiss': 1,
                'Cfa': 10,
                'Cfa_spoof': 10
            }

            tdcf_curve, _ = em.compute_tDCF(
                bona, spoof, Pfa, Pmiss, Pmiss_sp, cost, False
            )
            min_tdcf = np.min(tdcf_curve)

            with open(eer_txt, 'w') as f:
                f.write(f'min_tDCF: {min_tdcf:.4f}\neer: {eer * 100:.3f}\n')
            log(f'{name}  EER={eer * 100:.3f}%  min-tDCF={min_tdcf:.4f}')
            return {'eer': eer * 100, 'min_tdcf': min_tdcf}

        except Exception as e:
            log(f'[ERROR] {name} tDCF计算失败: {e}')
            with open(eer_txt, 'w') as f:
                f.write(f'eer: {eer * 100:.3f}\n')
            log(f'{name}  EER={eer * 100:.3f}%  (tDCF失败)')
            return {'eer': eer * 100, 'min_tdcf': None}
    else:
        with open(eer_txt, 'w') as f:
            f.write(f'eer: {eer * 100:.3f}\n')
        log(f'{name}  EER={eer * 100:.3f}%')
        return {'eer': eer * 100, 'min_tdcf': None}


# ==================== 主流程 ====================
def main():
    # 1. 添加命令行参数解析
    parser = argparse.ArgumentParser(description='XLSR-OT-MoE 模型测试')
    parser.add_argument('--dataset', nargs='+', default=None,
                        help='指定测试的数据集名称（支持模糊匹配，如: SpeechFake, ASV, cfad）')
    args = parser.parse_args()

    # 加载完整模型
    shared_encoder, moe_model = load_otmoe_model(cfg)

    # 定义所有测试集
    all_sets = [
        # ---- ASVspoof2019 LA eval ----
        ('asv19_eval',
         '/data0/yyj/dataset/ASVspoof2019_LA/ASVspoof2019_LA/ASVspoof2019.LA.cm.eval.trl.txt',
         '/data0/yyj/dataset/ASVspoof2019_LA/ASVspoof2019_LA/ASVspoof2019_LA_eval/wav/',
         '.wav',
         False),

        # ---- CFAD 6个子集 ----
        ('cfad_clean_test_see', '/data0/cff/CFAD/clean/test_see.txt', '/data0/cff/CFAD/clean/test_see/', '.wav', False),
        ('cfad_clean_test_unsee', '/data0/cff/CFAD/clean/test_unsee.txt', '/data0/cff/CFAD/clean/test_unsee/', '.wav', False),
        ('cfad_noisy_test_see', '/data0/cff/CFAD/noisy/test_see.txt', '/data0/cff/CFAD/noisy/test_see/', '.wav', False),
        ('cfad_noisy_test_unsee', '/data0/cff/CFAD/noisy/test_unsee.txt', '/data0/cff/CFAD/noisy/test_unsee/', '.wav', False),
        ('cfad_codec_test_see', '/data0/cff/CFAD/codec/test_see.txt', '/data0/cff/CFAD/codec/test_see/', '.wav', False),
        ('cfad_codec_test_unsee', '/data0/cff/CFAD/codec/test_unsee.txt', '/data0/cff/CFAD/codec/test_unsee/', '.wav', False),

        # ---- Codecfake C1-C7 ----
    ] + [
        (f'codecfake_C{c}', f'/data0/yyj/dataset/Codecfake/label/C{c}.txt', f'/data0/yyj/dataset/Codecfake/test/C{c}/', '.wav', False)
        for c in range(1, 8)
    ] + [
        # ---- ITW (In The Wild) ----
        ('ITW', '/data0/cff/wild/labels.txt', '/data0/cff/wild/wav/', '.wav', False),

        # ==================== SpeechFake 核心基准 ====================
        ('SpeechFake_BD_All',
         '/data0/speechfake/metadata/metadata/experiments/baseline/test_all.csv',
         '/data0/speechfake/',
         '.wav', False),
        # ==================== 新增：ASVspoof 5 Track 1 Eval 核心测试集 ====================
        ('asv5_track1_eval',
         '/data0/ASVspoof5/ASVspoof5_protocols/ASVspoof5.eval.track_1.txt',
         '/data0/ASVspoof5/ASVspoof5_eval/',
         '.flac', False),
         # ==================== 新增：ADD 2023 Track 3 核心评测集 ====================
        ('ADD2023_T3_test',
         '/data0/cff/ADD2023/test/label.txt',
         '/data0/cff/ADD2023/test/wav/',
         '.wav', False),
         # 1. ADD2022 Track 3.2 R2 (经典首届攻防博弈最终轮)
        ('ADD2022_T3.2_R2_eval',
         '/data0/yyj/dataset/ADD2022track3.2_eval_R2/track3_R2_label.txt',
         '/data0/yyj/dataset/ADD2022track3.2_eval_R2/wav/',
         '.wav', False),

        # 2. ADD2023 Track 1.2 R2 (低质量信道与局部篡改最终轮)
        ('ADD2023_T1.2_R2_eval',
         '/data0/yyj/dataset/ADD2023track1.2_eval_R2/label.txt',
         '/data0/yyj/dataset/ADD2023track1.2_eval_R2/wav/',
         '.wav', False),

        # 3. AI4T (欧洲 AI4Trust 多语言可信检测基准 - 自动扫描 real/fake)
        ('AI4T',
         'DIR_REAL_FAKE',
         '/data0/yyj/dataset/AI4T/',
         '.wav', False),

        # 4. SONAR (前沿大模型零样本生成基准 - 自动扫描 VoiceBox/VALL-E/xTTS 等)
        ('SONAR_All',
         'DIR_SONAR',
         '/data0/yyj/dataset/SONAR/',
         '.wav', False),
         # ==================== 新增：CD-ADD, ASV2021 LA, ASV2021 DF ====================
        # 1. CD-ADD (零样本跨域 TTS 测试集 - LibriTTS test-clean)
        ('CD_ADD_test_clean',
         'DIR_CD_ADD',
         '/data1/tjj/CD-ADD/lyaleo___CD-ADD/dataset_LibriTTS/test-clean/',
         '.wav', False),

        # 2. ASVspoof 2021 LA eval (信道/编码有损传输评测集)
        ('asv21_la_eval',
         '/data1/tjj/CFF/SSL_Anti-spoofing-main/LA-keys-stage-1/keys/CM/trial_metadata.txt',
         '/data0/yyj/dataset/ASVspoof2019_LA/ASVspoof2019_LA/ASVspoof2021_LA_eval/wav/',
         '.wav', False),

        # 3. ASVspoof 2021 DF eval (100+ 算法超大规模深度伪造基准)
        ('asv21_df_eval',
         '/data1/tjj/CFF/SSL_Anti-spoofing-main/DF-keys-stage-1/keys/CM/trial_metadata.txt',
         '/data0/yyj/dataset/ASVspoof2019_LA/ASVspoof2019_LA/ASVspoof2021_DF_eval/wav/',
         '.wav', False),
        # =================================================================
    ]

    # ==================== 2. 根据命令行参数过滤测试集 ====================
    if args.dataset:
        selected_sets = []
        for item in all_sets:
            # 支持模糊匹配（如输入 SpeechFake 会自动匹配 SpeechFake_BD_All 和 SpeechFake_BD_UT）
            if any(target.lower() in item[0].lower() for target in args.dataset):
                selected_sets.append(item)
        if len(selected_sets) == 0:
            log(f'[ERROR] 未匹配到任何包含 {args.dataset} 的测试集！可用名称: {[s[0] for s in all_sets]}')
            return
        all_sets = selected_sets
        log(f'[CLI Filter] 仅测试指定的数据集: {[s[0] for s in all_sets]}')
    # ====================================================================

    log('=' * 60)
    log('XLSR-OT-MoE 模型测试开始')
    log(f'配置文件: {cfg_path}')
    log(f'设备: {device}')
    log(f'测试batch_size: {TEST_BATCH_SIZE} (原始训练batch_size: {cfg.get("batch_size", "?")})')
    log(f'总测试集数量: {len(all_sets)}')
    log(f'分数约定: real概率 (probs[:,0])，高分=真人')
    log('=' * 60)

    # 汇总结果
    results_summary = []

    # 逐个测试
    for name, proto, root, ext, need_tdcf in all_sets:
        try:
            # 排除所有目录扫描模式
            if proto not in ['DIR_REAL_FAKE', 'DIR_SONAR', 'DIR_CD_ADD'] and not os.path.exists(proto):
                log(f'[SKIP] {name}: 协议文件不存在 {proto}')
                continue
            if not os.path.exists(root):
                log(f'[SKIP] {name}: 数据目录不存在 {root}')
                continue

            files, labels = load_proto(proto, root, ext)
            if len(files) == 0:
                log(f'[SKIP] {name}: 无有效样本')
                continue

            result = run_subset(
                name, files, labels,
                shared_encoder, moe_model,
                trial_proto=proto if need_tdcf else None,
                need_tdcf=need_tdcf
            )

            if result:
                results_summary.append((name, result))

            # 子集之间清理显存碎片
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            log(f'[ERROR] {name} 测试失败: {str(e)}')
            import traceback
            log(traceback.format_exc())

    # ---------- 汇总表格 ----------
    log('\n' + '=' * 60)
    log('测试结果汇总')
    log('=' * 60)
    log(f'{"数据集":<25} {"EER(%)":>10} {"min-tDCF":>10}')
    log('-' * 47)

    valid_results = []   # 用于存储有效的字典结果，用于计算平均EER

    for name, res in results_summary:
        # 检查 res 是否为字典且包含 'eer' 键
        if isinstance(res, dict) and 'eer' in res:
            tdcf_str = f'{res["min_tdcf"]:.4f}' if res.get('min_tdcf') is not None else 'N/A'
            log(f'{name:<25} {res["eer"]:>10.3f} {tdcf_str:>10}')
            valid_results.append(res)
        else:
            # 如果是其他类型（如元组），记录错误并跳过
            log(f'{name:<25} {"??":>10}  (返回值格式异常: {type(res)})')
            # 可选：打印 res 内容以辅助调试
            log(f'  [DEBUG] {name} res = {res}')

    # 计算平均 EER（仅基于有效的字典结果）
    if valid_results:
        avg_eer = np.mean([r['eer'] for r in valid_results])
        log('-' * 47)
        log(f'{"平均EER":<25} {avg_eer:>10.3f}')
    else:
        log('-' * 47)
        log(f'{"平均EER":<25} {"无有效数据":>10}')

    log('=' * 60)
    log(f'[ALL DONE] 全部结果已保存至 {save_root}')
    log('=' * 60)


if __name__ == '__main__':
    main()