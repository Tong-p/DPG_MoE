#!/usr/bin/env python3
"""
XLSR-OTMoE训练脚本（Transport Embedding版，无OT loss）
基于SSL-AASIST的OT-MoE实现，适配160维特征
【修改】完全移除OT辅助损失，仅依赖Transport Embedding作为gate特征输入
"""

import os
import sys
sys.path.insert(0, '/data1/tjj/XLSR-MOE/FM')
sys.path.insert(0, '/data1/tjj/XLSR-MOE')
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import yaml
import logging
import random
from datetime import datetime
import json
from collections import defaultdict
import torch.nn.functional as F
from model_SSL_Enhanced import SSLAASISTExpert

from otmoe_model_otloss_256 import Enhanced_MOE_XLSR

from ot_training_otloss_untils256 import (
    train_epoch_ot, valid_epoch_ot, eval_model_ot,
    LoadTrainData_XLSR, LoadEvalData_XLSR,
    safe_collate_fn, safe_collate_fn_eval,
    StratifiedDomainSampler
)
from utils import seed_everything 
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

class NativeXLSR256DExtractor(nn.Module):
    """提取微调后 XLSR-300M 第 12 层中间声学/信道域特征 + 256维固定正交投影 (100% 对齐 CFM 空间)"""
    def __init__(self, 
                 ckpt_path='/data1/tjj/XLSR-MOE/xlsr2_300m.pt', 
                 mixed_aasist_ckpt='/data1/tjj/XLSR-MOE/models_ssl_youhua/SSLAASIST_Mixed_best.pth', # 【关键修改】：增加微调权重路径
                 device='cuda'):
        super().__init__()
        import fairseq
        print(f"[XLSR-256D] 正在加载原生 XLSR 基座: {ckpt_path}")
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.ssl_model = model[0].to(device)
        self.device = device
        
        # 【关键修改】：植入微调后的 XLSR 权重，确保与 CFM 训练时提取的特征空间绝对一致
        if mixed_aasist_ckpt and os.path.exists(mixed_aasist_ckpt):
            print(f"[XLSR-256D] 正在植入微调权重: {mixed_aasist_ckpt}")
            ckpt = torch.load(mixed_aasist_ckpt, map_location='cpu')
            state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
            
            ssl_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('ssl_model.model.'):
                    ssl_state_dict[k.replace('ssl_model.model.', '')] = v
                elif k.startswith('ssl_model.'):
                    ssl_state_dict[k.replace('ssl_model.', '')] = v
                    
            missing, unexpected = self.ssl_model.load_state_dict(ssl_state_dict, strict=False)
            print(f"  -> 微调 XLSR 权重成功植入! (未匹配键: {len(unexpected)})")
        else:
            print(f"[WARNING] 未找到微调权重，将使用原生 XLSR!")

        self.ssl_model.eval()
        for param in self.ssl_model.parameters():
            param.requires_grad = False
            
        # 【种子 42 + 正交矩阵保距投影】：保证投影空间与 CFM 生成域原型 100% 绝对重合
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
        # 提取微调后 XLSR 的 Layer-12 中间层特征
        res = self.ssl_model(x, mask=False, features_only=True, layer=12)
        ssl_out = res['x'] # [B, Frame, 1024]
        feat_1024 = ssl_out.mean(dim=1)
        feat_256 = self.proj(feat_1024)
        return F.normalize(feat_256, p=2, dim=1)

def compute_eer(labels, scores):
    """
    labels: 真实标签（0=real, 1=fake）
    scores: 预测为 fake 的概率（即 softmax 输出的第1列）
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100   # 返回百分比

# ==================== 协议加载函数（保持不变）====================

def load_asvspoof2019_protocol(protocol_path, root_dir, suffix='.wav'):
    labels = {}
    file_list = []
    with open(protocol_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            speaker_id, fname, _, attack_type, label = parts[:5]
            file_list.append(fname)
            labels[fname] = 0 if label == 'bonafide' else 1
    
    def safe_add_suffix(fname):
        if fname.endswith('.pt') or fname.endswith('.wav') or fname.endswith('.flac'):
            return fname
        return fname + suffix
    
    df = pd.DataFrame({
        'fname': file_list,
        'label': [labels[f] for f in file_list]
    })
    df['path'] = df['fname'].apply(lambda x: os.path.join(root_dir, safe_add_suffix(x)))
    return df[['path', 'label']]


def load_cfad_protocol(protocol_path, root_dir, suffix='.wav'):
    df = pd.read_csv(protocol_path, sep=r'\s+', header=None, engine='python')
    df.columns = ['fname', 'label', 'attack_type']
    label_map = {'real': 0, 'genuine': 0, 'fake': 1, 'spoof': 1}
    df['label'] = df['label'].map(label_map).astype(int)
    df = df.dropna(subset=['label'])
    
    def safe_add_suffix(fname):
        if fname.endswith('.pt') or fname.endswith('.wav') or fname.endswith('.flac'):
            return fname
        return fname + suffix
    
    df['path'] = df['fname'].apply(lambda x: os.path.join(root_dir, safe_add_suffix(x)))
    return df[['path', 'label']]


def load_codecfake_protocol(protocol_path, root_dir, suffix='.wav'):
    df = pd.read_csv(protocol_path, sep=r'\s+', header=None, engine='python')
    df.columns = ['fname', 'label_str', 'code']
    df['label'] = df['label_str'].map({'real': 0, 'fake': 1}).astype(int)
    df = df.dropna(subset=['label'])
    
    def safe_add_suffix(fname):
        if fname.endswith('.pt') or fname.endswith('.wav') or fname.endswith('.flac'):
            return fname
        return fname + suffix
    
    df['path'] = df['fname'].apply(lambda x: os.path.join(root_dir, safe_add_suffix(x)))
    return df[['path', 'label']]


# ==================== 域原型加载类（保持不变）====================

class DomainPrototypeXLSR:
    def __init__(self, domain_name, cache_dir, n_samples=5000, device='cpu'):
        self.domain_name = domain_name
        self.n_samples = n_samples
        self.device = device
        
        cache_file = os.path.join(cache_dir, f"{domain_name}_n{n_samples}.pth")
        
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"域原型缓存不存在: {cache_file}\n请先运行ge.py生成!")
        
        print(f"[DomainPrototypeXLSR] Loading {cache_file}")
        cache = torch.load(cache_file, map_location=device)
        
        self.samples = cache["samples"]
        self.labels = cache["labels"]
        self.subclasses = cache.get("subclasses", cache["labels"])
        self.center = cache["center"]
        self.subclass_centers = cache.get("subclass_centers", {})
        self.attack_to_id = cache.get("attack_to_id", {})
        
        print(f"[DomainPrototypeXLSR] Loaded {self.samples.shape[0]} samples, "
              f"{self.labels.unique().numel()} classes, "
              f"{self.subclasses.unique().numel()} subclasses")
    
    def get_center(self):
        return self.center.to(self.device)
    
    def get_samples(self):
        return self.samples.to(self.device)
    
    def get_labels(self):
        return self.labels.to(self.device)
    
    def get_subclasses(self):
        return self.subclasses.to(self.device)


# ==================== 与ATADD对齐：分层学习率优化器 ====================

def get_optimizer_with_lr_groups(model, config, args):
    gate_params = []
    expert_unfrozen_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'gating_network' in name:
            gate_params.append(param)
        elif 'experts' in name:
            expert_unfrozen_params.append(param)
        else:
            other_params.append(param)
    
    base_lr = config['lr']
    gate_lr = 5e-5
    expert_lr = base_lr * args.expert_lr_ratio
    wd = config.get('weight_decay', 0.1)
    
    param_groups = []
    
    if gate_params:
        param_groups.append({
            'params': gate_params,
            'lr': gate_lr,               
            'weight_decay': wd,          
            'name': 'gating'
        })
    
    if expert_unfrozen_params:
        param_groups.append({
            'params': expert_unfrozen_params,
            'lr': expert_lr,
            'weight_decay': wd * 1.0,
            'name': 'expert_finetune'
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr * 0.5,
            'weight_decay': wd * 1.0,
            'name': 'other'
        })
    
    optimizer = torch.optim.AdamW(param_groups)
    
    print(f"\n[Optimizer Groups]")
    for g in param_groups:
        print(f"  {g['name']}: lr={g['lr']:.2e}, wd={g['weight_decay']:.4f}, "
              f"params={sum(p.numel() for p in g['params']):,}")
    
    return optimizer

# ==================== SDS域结构构建函数 ====================

def build_domain_structure(df_asv, df_cfad, df_codec,
                           asv_proto_path=None, cfad_proto_path=None, codec_proto_path=None):
    domain_files = {}
    domain_labels = {}
    domain_subclass_map = {}
    
    # ===== ASVspoof2019 =====
    asv_files = defaultdict(list)
    asv_labels = {}
    
    asv_attack_map = {}
    
    if asv_proto_path and os.path.exists(asv_proto_path):
        with open(asv_proto_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    speaker_id, utt_id, _, attack_type, label = parts[:5]
                    fname = utt_id
                    if label == 'bonafide':
                        attack_type = 'bonafide'
                    asv_attack_map[fname] = attack_type
    
    for _, row in df_asv.iterrows():
        path = row['path']
        label = int(row['label'])
        asv_labels[path] = label
        
        fname = os.path.basename(path).replace('.flac', '').replace('.wav', '').replace('.pt', '')
        attack_type = asv_attack_map.get(fname, 'bonafide' if label == 0 else 'A01')
        subclass_name = f"asv_{attack_type}"
        asv_files[subclass_name].append(path)
    
    domain_files['ASVspoof2019'] = dict(asv_files)
    domain_labels['ASVspoof2019'] = asv_labels
    domain_subclass_map['ASVspoof2019'] = {
        name: i for i, name in enumerate(sorted(asv_files.keys()))
    }
    
    # ===== CFAD =====
    cfad_files = defaultdict(list)
    cfad_labels = {}
    
    cfad_subclass_map = {}
    
    if cfad_proto_path and os.path.exists(cfad_proto_path):
        with open(cfad_proto_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    fname = parts[0].replace('.wav', '').replace('.flac', '').replace('.pt', '')
                    label = parts[1]
                    attack_type = parts[2]
                    cfad_subclass_map[fname] = attack_type
    
    for _, row in df_cfad.iterrows():
        path = row['path']
        label = int(row['label'])
        cfad_labels[path] = label
        
        fname = os.path.basename(path).replace('.wav', '').replace('.flac', '').replace('.pt', '')
        subclass_type = cfad_subclass_map.get(fname, None)
        
        if subclass_type is None:
            if label == 0:
                subclass_type = 'real_unknown'
            else:
                subclass_type = 'fake_unknown'
        
        prefix = 'cfad_real' if label == 0 else 'cfad_fake'
        subclass_name = f"{prefix}_{subclass_type}"
        
        cfad_files[subclass_name].append(path)
    
    domain_files['CFAD'] = dict(cfad_files)
    domain_labels['CFAD'] = cfad_labels
    domain_subclass_map['CFAD'] = {
        name: i for i, name in enumerate(sorted(cfad_files.keys()))
    }
    
    # ===== Codecfake =====
    codec_files = defaultdict(list)
    codec_labels = {}
    
    codec_code_map = {}
    
    if codec_proto_path and os.path.exists(codec_proto_path):
        with open(codec_proto_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    fname = parts[0].replace('.wav', '').replace('.flac', '').replace('.pt', '')
                    code = parts[2]
                    codec_code_map[fname] = code
    
    for _, row in df_codec.iterrows():
        path = row['path']
        label = int(row['label'])
        codec_labels[path] = label
        
        fname = os.path.basename(path).replace('.wav', '').replace('.flac', '').replace('.pt', '')
        code = codec_code_map.get(fname, '0' if label == 0 else '1')
        
        if label == 0:
            subclass_name = "codec_real"
        else:
            subclass_name = f"codec_fake_{code}"
        
        codec_files[subclass_name].append(path)
    
    domain_files['Codecfake'] = dict(codec_files)
    domain_labels['Codecfake'] = codec_labels
    domain_subclass_map['Codecfake'] = {
        name: i for i, name in enumerate(sorted(codec_files.keys()))
    }
    
    print("\n[SDS] Domain structure built:")
    for domain in ['ASVspoof2019', 'CFAD', 'Codecfake']:
        n_subclasses = len(domain_files[domain])
        n_samples = sum(len(v) for v in domain_files[domain].values())
        print(f"  {domain}: {n_subclasses} subclasses, {n_samples} samples")
        for sub_name, files in sorted(domain_files[domain].items()):
            print(f"    {sub_name}: {len(files)}")
    
    return domain_files, domain_labels, domain_subclass_map

# ==================== 主函数（移除OT loss相关配置）====================

def main():
    parser = argparse.ArgumentParser(description='XLSR-OTMoE训练（Transport Embedding版，无OT loss）')
    parser.add_argument('--config', type=str, default='/data1/tjj/XLSR-MOE/OTMOE/OTM_MOE/train_config_otmoe_doss.yaml')
    parser.add_argument('--resume', type=str, default='',
                       help='从检查点恢复训练')
    parser.add_argument('--unfreeze_experts', action='store_true',
                       help='解冻专家参数')
    parser.add_argument('--unfreeze_mode', type=str, default='deep',
                       choices=['none', 'output', 'deep', 'half', 'encoder_all', 'all_but_ssl'],
                       help='专家解冻模式')
    parser.add_argument('--expert_lr_ratio', type=float, default=0.2,
                       help='专家微调学习率相对于主学习率的比例')
    # SDS分层采样参数
    parser.add_argument('--use_sds', action='store_true', default=False,
                       help='启用分层动态采样(SDS)')
    parser.add_argument('--Nc_subclass', type=int, default=2500,
                       help='每子类截断上限（DOSS-Select Nc）')
    parser.add_argument('--rho', type=float, default=0.25,
                       help='真实/伪造采样比例')
    parser.add_argument('--sds_quick', action='store_true', default=False,
                       help='SDS快速模式(Nc=1000)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 设置日志
    log_dir = config.get('log_dir', '/data1/tjj/XLSR-MOE/OTMOE/logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_transport_emb_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    seed_everything(config.get('seed', 42))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'使用设备: {device}')
    
    os.makedirs(config['save_model_folder'], exist_ok=True)
    
    # ==================== 加载共享编码器（256 维微调物理域空间） ====================
    logger.info("="*60)
    logger.info("加载 256维 共享编码器 (Fine-tuned XLSR-300M Layer-12)...")
    
    # 【修改】：传入 mixed_aasist_ckpt 路径
    shared_encoder = NativeXLSR256DExtractor(
        ckpt_path=config.get('ssl_path', '/data1/tjj/XLSR-MOE/xlsr2_300m.pt'),
        mixed_aasist_ckpt=config.get('mixed_aasist_ckpt', '/data1/tjj/XLSR-MOE/models_ssl_youhua/SSLAASIST_Mixed_best.pth'),
        device=device
    ).to(device).eval()

    for param in shared_encoder.parameters():
        param.requires_grad = False
    
    logger.info(">>> 共享 256维 XLSR Layer-12 特征提取器已加载并严格冻结 (完全对齐 CFM 域原型空间)")

    # ==================== 加载预训练专家 ====================
    logger.info("="*60)
    logger.info("加载预训练专家...")
    
    expert_paths = [
        config['expert_1_path'],
        config['expert_2_path'],
        config['expert_3_path']
    ]
    
    experts = []
    for i, path in enumerate(expert_paths, 1):
        logger.info(f"加载专家 {i}: {path}")
        expert = SSLAASISTExpert(
            device=device,
            ssl_path=config.get('ssl_path', '/data1/tjj/XLSR-MOE/xlsr2_300m.pt'),
            return_emb=True
        )
        
        ckpt = torch.load(path, map_location=device)
        if 'model_state' in ckpt:
            expert.load_state_dict(ckpt['model_state'])
        else:
            expert.load_state_dict(ckpt)
        
        expert = expert.to(device)
        experts.append(expert)
        logger.info(f"专家 {i} 加载完成")
    
    # ==================== 构建 OT-MoE 模型 ====================
    logger.info("="*60)
    logger.info("构建 XLSR-OT-MoE 模型 (256 维物理域先验版)...")
    
    moe_model = Enhanced_MOE_XLSR(
        experts=experts,
        emb_dim=256,         # 【修改 3】：特征维度传入 256
        hidden_dim=64,
        num_classes=2,
        freezing=True,
        use_ot_gate=True,
        num_domains=3
    ).to(device)
    
    total_params = sum(p.numel() for p in moe_model.parameters())
    trainable_params = sum(p.numel() for p in moe_model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # ==================== 加载域原型 ====================
    logger.info("="*60)
    logger.info("加载域原型（Transport Embedding计算用）...")
    
    cache_dir = config.get('domain_cache_dir', "/data1/tjj/XLSR-MOE/FM/domain_cache")
    n_samples = config.get('n_prototype_samples', 5000)
    
    domain_names = ["ASVspoof2019_generated", "CFAD_generated", "Codecfake_generated"]
    for name in domain_names:
        ck = os.path.join(cache_dir, f"{name}_n{n_samples}.pth")
        if not os.path.isfile(ck):
            raise FileNotFoundError(f"域原型缓存缺失: {ck}\n请先运行ge.py生成!")
    
    proto_asv = DomainPrototypeXLSR("ASVspoof2019_generated", cache_dir, n_samples, device='cpu')
    proto_cfad = DomainPrototypeXLSR("CFAD_generated", cache_dir, n_samples, device='cpu')
    proto_codec = DomainPrototypeXLSR("Codecfake_generated", cache_dir, n_samples, device='cpu')

    centers = [c.to(device) for c in [proto_asv.get_center(), proto_cfad.get_center(), proto_codec.get_center()]]
    samples = [s.to(device) for s in [proto_asv.get_samples(), proto_cfad.get_samples(), proto_codec.get_samples()]]
    labels = [l.to(device) for l in [proto_asv.get_labels(), proto_cfad.get_labels(), proto_codec.get_labels()]]
    
    moe_model.set_domain_prototypes(centers=centers, samples=samples, labels=labels)
    logger.info(">>> 域原型已注入MoE（GPU）")
    moe_model.to(device)  # 确保 PCADT 模块也在 GPU
    
    # ==================== 与ATADD对齐：优化器设置 ====================
    if args.unfreeze_experts:
        moe_model.unfreeze_experts_partial(unfreeze_mode=args.unfreeze_mode)
        logger.info(f">>> 专家已解冻 (mode={args.unfreeze_mode})")
    
    for param in moe_model.gating_network.parameters():
        param.requires_grad = True
    
    if config.get('use_layered_optimizer', True):
        optimizer = get_optimizer_with_lr_groups(moe_model, config, args)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, moe_model.parameters()),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.1)
        )
    
    scheduler_factor = float(config.get('scheduler_factor', 0.5))
    scheduler_patience = int(config.get('scheduler_patience', 3))
    scheduler_threshold = float(config.get('scheduler_threshold', 1e-4))
    scheduler_min_lr = float(config.get('scheduler_min_lr', 1e-7))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        min_lr=scheduler_min_lr,
        verbose=False
    )
    
    # ==================== 与ATADD对齐：带标签平滑和类别权重的损失函数 ====================
    logger.info("="*60)
    logger.info("加载数据集...")
    
    df_asv = load_asvspoof2019_protocol(config['train_asv_proto'], config['train_asv_root'])
    df_asv_dev = load_asvspoof2019_protocol(config['dev_asv_proto'], config['dev_asv_root'])
    df_cfad = load_cfad_protocol(config['train_cfad_proto'], config['train_cfad_root'])
    df_cfad_dev = load_cfad_protocol(config['dev_cfad_proto'], config['dev_cfad_root'])
    df_codec = load_codecfake_protocol(config['train_BCodec_proto'], config['train_BCodec_root'])
    df_codec_dev = load_codecfake_protocol(config['dev_BCodec_proto'], config['dev_BCodec_root'])
    
    df_train = pd.concat([df_asv, df_cfad, df_codec], ignore_index=True)
    df_dev = pd.concat([df_asv_dev, df_cfad_dev, df_codec_dev], ignore_index=True)
    
    logger.info(f"训练集样本数: {len(df_train)}")
    logger.info(f"验证集样本数: {len(df_dev)}")
    
    # ===== SDS：构建域结构并创建分层采样器 =====
    sampler = None
    if args.use_sds:
        logger.info("="*60)
        logger.info("启用SDS分层动态采样...")
        
        Nc = 1000 if args.sds_quick else args.Nc_subclass
        rho = args.rho
        
        logger.info(f"SDS参数: Nc_subclass={Nc}, rho={rho}")
        
        domain_files, domain_labels, domain_subclass_map = build_domain_structure(
            df_asv, df_cfad, df_codec,
            asv_proto_path=config['train_asv_proto'],
            cfad_proto_path=config['train_cfad_proto'],
            codec_proto_path=config['train_BCodec_proto']
        )
        
        sampler = StratifiedDomainSampler(
            domain_files=domain_files,
            domain_labels=domain_labels,
            domain_subclass_map=domain_subclass_map,
            Nc_subclass=Nc,
            rho=rho,
            seed=config.get('seed', 42)
        )
        
        logger.info(">>> SDS采样器已创建")
    else:
        logger.info("使用标准全量采样模式")

    # ===== 与ATADD对齐：计算类别权重（SDS适配版）=====
    real_count = (df_train['label'] == 0).sum()
    fake_count = (df_train['label'] == 1).sum()
    logger.info(f"训练集分布: Real={real_count}, Fake={fake_count}")

    if args.use_sds:
        weight_real = 4.0
        weight_fake = 1.0
        logger.info(f"SDS模式: 使用适配权重 [{weight_real:.2f}, {weight_fake:.2f}]")
    else:
        max_count = max(real_count, fake_count)
        weight_real = max_count / real_count if real_count > 0 else 1.0
        weight_fake = max_count / fake_count if fake_count > 0 else 1.0
        MAX_WEIGHT = 5.0
        weight_real = min(weight_real, MAX_WEIGHT)
        weight_fake = min(weight_fake, MAX_WEIGHT)

    class_weights = torch.FloatTensor([weight_real, weight_fake]).to(device)
    logger.info(f"类别权重: Real={weight_real:.4f}, Fake={weight_fake:.4f}")
    
    label_smoothing = config.get('label_smoothing', 0.1)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    ).to(device)
    logger.info(f"使用加权CrossEntropyLoss + 标签平滑: {label_smoothing}")
    
    # ==================== 创建数据加载器（支持SDS）====================
    d_label_trn = dict(zip(df_train['path'], df_train['label']))
    train_set = LoadTrainData_XLSR(
        list_IDs=df_train['path'].tolist(),
        labels=d_label_trn,
        win_len=config.get('win_len', 4.0375),
        sampler=sampler
    )
    
    train_shuffle = (sampler is None)
    
    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=train_shuffle,          
        drop_last=True,
        num_workers=config.get('num_workers', 8),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=safe_collate_fn
    )
    
    d_label_dev = dict(zip(df_dev['path'], df_dev['label']))
    dev_set = LoadEvalData_XLSR(
        list_IDs=df_dev['path'].tolist(),
        labels=d_label_dev,
        win_len=config.get('win_len', 4.0375)
    )
    
    dev_loader = DataLoader(
        dev_set,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 8),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=safe_collate_fn_eval
    )
    
    # ==================== 恢复训练 ====================
    start_epoch = 0
    best_eer = 100.0
    best_loss = float('inf')
    best_acc = 0.0
    early_stop_counter = 0
    
    ckpt_path = os.path.join(config['save_model_folder'], 'ot_checkpoint_last.pth')
    
    if args.resume and os.path.exists(args.resume):
        logger.info(f"从检查点恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        
        missing_keys, unexpected_keys = moe_model.load_state_dict(ckpt['model'], strict=False)
        if missing_keys:
            logger.info(f"[Resume] 新层已初始化 (missing {len(missing_keys)} keys): {missing_keys[:5]}")
        if unexpected_keys:
            logger.info(f"[Resume] 忽略旧层 (unexpected {len(unexpected_keys)} keys): {unexpected_keys[:5]}")
        
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        
        start_epoch = ckpt.get('epoch', 0) + 1
        best_eer = ckpt.get('best_eer', 100.0)
        best_loss = ckpt.get('best_loss', float('inf'))
        best_acc = ckpt.get('best_acc', 0.0)
        early_stop_counter = ckpt.get('early_stop', 0)
        
        if 'rng' in ckpt:
            try:
                torch.set_rng_state(ckpt['rng'])
            except TypeError:
                print("Warning: RNG state type mismatch, skipping...")
        if 'cuda_rng' in ckpt and ckpt['cuda_rng'] is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state(ckpt['cuda_rng'])
            except TypeError:
                print("Warning: CUDA RNG state type mismatch, skipping...")
        
        logger.info(f"恢复至 epoch {start_epoch}, best_loss={best_loss:.4f}")
    
    # ==================== 训练循环（Transport Embedding版，无OT loss）====================
    num_epochs = config['num_epochs']
    early_stopping_patience = config.get('early_stopping', 8)
    gate_log_interval = config.get('gate_log_interval', 100)
    log_weights = config.get('log_weights', True)
    grad_clip_norm = config.get('grad_clip_norm', 0.5)
    
    gate_stats_history = [] if log_weights else None
    
    logger.info(f"\n开始训练，总epoch: {num_epochs}")
    logger.info(f"【配置】OT辅助损失: 已移除（仅Transport Embedding作为特征输入）")
    logger.info(f"【配置】早停耐心: {early_stopping_patience}")
    logger.info(f"【配置】最佳模型标准: EER优先")
    logger.info(f"【配置】梯度裁剪: {grad_clip_norm}")
    logger.info(f"【配置】门控记录间隔: 每{gate_log_interval}个batch")
    
    for epoch in range(start_epoch, num_epochs):
        if early_stop_counter >= early_stopping_patience:
            logger.info(f'早停触发 @ epoch {epoch}')
            break
        
        logger.info(f"\n{'='*60}")
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        
        # ===== SDS：每轮重建子集 =====
        if args.use_sds and hasattr(train_set, 'set_epoch'):
            train_set.set_epoch(epoch)
            logger.info(f'[SDS] Epoch {epoch} subset rebuilt: {len(train_set)} samples')
        
        # ===== 打印所有参数组的学习率 =====
        lr_info = []
        for i, group in enumerate(optimizer.param_groups):
            group_name = group.get('name', f'group_{i}')
            lr_info.append(f"{group_name}={group['lr']:.2e}")
        lr_str = ", ".join(lr_info)
        
        logger.info(f'Epoch:{epoch+1:03d}  lr:[{lr_str}]')
        
        # 训练阶段（【修改】移除alpha_min/max/use_ot_loss参数）
        train_loss, train_acc = train_epoch_ot(
            train_loader, moe_model, optimizer, criterion, device,
            shared_encoder=shared_encoder,
            domain_centers=centers,          # 仍传入用于监控统计
            domain_samples=samples,
            domain_labels=labels,
            epoch=epoch,
            total_epochs=num_epochs,
            gate_log_interval=gate_log_interval,
            gate_stats_history=gate_stats_history,
            grad_clip_norm=grad_clip_norm
        )
        
        # 验证阶段
        val_loss, val_acc, val_eer = valid_epoch_ot(
            dev_loader, moe_model, criterion, device,
            shared_encoder=shared_encoder
        )
        
        # ReduceLROnPlateau
        scheduler.step(val_loss)
        
        logger.info(f'Epoch:{epoch+1:03d}')
        logger.info(f'Train Loss/Acc: {train_loss:.5f}/{train_acc:.2f}%')
        logger.info(f'Valid Loss/Acc/EER: {val_loss:.5f}/{val_acc:.2f}% / {val_eer:.2f}%')
        
        # 保存检查点
        ckpt = {
            'epoch': epoch,
            'model': moe_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_eer': best_eer,
            'best_loss': best_loss,
            'best_acc': best_acc,
            'early_stop': early_stop_counter,
            'rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'gate_stats': gate_stats_history if gate_stats_history else None
        }
        torch.save(ckpt, ckpt_path)
        
        # 定期保存门控统计
        if log_weights and (epoch + 1) % 10 == 0:
            gate_stats_path = os.path.join(config['save_model_folder'], f'gate_stats_epoch{epoch+1}.json')
            with open(gate_stats_path, 'w') as f:
                json.dump(gate_stats_history, f, indent=2)
            logger.info(f'门控统计已保存: {gate_stats_path}')
        
        # ===== 以 EER 为主，Loss 为辅的最佳模型保存策略 =====
        is_better = False
        if val_eer < best_eer - 0.01:
            is_better = True
            logger.info(f'*** EER 显著下降: {val_eer:.2f}% < {best_eer:.2f}% ***')
        elif abs(val_eer - best_eer) <= 0.05 and val_loss < best_loss:
            is_better = True
            logger.info(f'*** EER 持平，Loss 下降: {val_loss:.4f} < {best_loss:.4f} ***')

        if is_better:
            # 使用 min/max 确保记录的永远是历史最优值，防止 best_eer 爬升
            best_eer = min(val_eer, best_eer)
            best_loss = min(val_loss, best_loss)
            best_acc = max(val_acc, best_acc)
            early_stop_counter = 0
            best_path = os.path.join(config['save_model_folder'], 'XLSR_OT_MOE_best.pth')
            torch.save(moe_model.state_dict(), best_path)
            if gate_stats_history:
                best_stats_path = os.path.join(config['save_model_folder'], 'gate_stats_best.json')
                with open(best_stats_path, 'w') as f:
                    json.dump(gate_stats_history, f, indent=2)
        else:
            early_stop_counter += 1
            logger.info(f'早停计数器: {early_stop_counter}/{early_stopping_patience}')
    
    # 训练结束
    if log_weights and gate_stats_history:
        final_stats_path = os.path.join(config['save_model_folder'], 'gate_stats_final.json')
        with open(final_stats_path, 'w') as f:
            json.dump(gate_stats_history, f, indent=2)
        logger.info(f'最终门控统计已保存: {final_stats_path}')
    
    logger.info(f"\n{'='*60}")
    logger.info(f'训练完成! 最佳Val EER: {best_eer:.4f}%, Loss: {best_loss:.4f}, Acc: {best_acc:.2f}%')
    logger.info(f"{'='*60}")

if __name__ == '__main__':
    main()