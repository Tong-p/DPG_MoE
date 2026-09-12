#!/usr/bin/env python3
"""
SSL-AASIST 域原型生成脚本
使用训练好的Flow Matching模型生成160维特征
支持三种数据集，按子类分布生成，包含可视化功能
新增：支持生成特征与原始特征的分布对比
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import logging
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datetime import datetime
from collections import defaultdict, Counter
from typing import Optional, Tuple, Dict
from scipy.stats import wasserstein_distance

# 导入训练脚本中的模型定义
from trainge_main import (
    SSLAASISTFeatureGenerator, 
    load_dataset_protocol,
    SSLAASISTFeatureExtractor,
    load_audio_fixed_length
)

# 设置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_real_subclass(subclass_id, attack_to_id, real_keywords=['aishell', 'magic', 'thchs']):
    """
    判断给定的 subclass_id 是否代表 Real 数据

    """
    id_to_attack = {v: k for k, v in attack_to_id.items()}
    if subclass_id in id_to_attack:
        name = id_to_attack[subclass_id].lower()
        # 如果名称包含常见的 real source 关键词，或者你可以在训练时保存一个 "real_id_set"
        return name in ['aishell1', 'aishell3', 'magicread', 'thchs30'] # 硬编码或动态传入
    return False

def compute_real_subclass_distribution(proto_txt, root_dir, suffix, dataset_name):
    """
    计算真实数据中的子类分布比例，并返回real子类ID列表
    Returns:
        subclass_ratios: dict {subclass_id: ratio}
        subclass_counts: dict {subclass_id: count}
        attack_to_id: dict
        real_subclass_ids: list  # 新增：所有real子类ID
    """
    # 原有加载协议代码不变
    _, attack_types, file_list, paths, attack_to_id = \
        load_dataset_protocol(dataset_name, proto_txt, root_dir, suffix)
    
    subclass_counts = Counter(attack_types.values())
    total = len(file_list)
    subclass_ratios = {subclass_id: count / total 
                      for subclass_id, count in subclass_counts.items()}
    
    id_to_attack = {v: k for k, v in attack_to_id.items()}
    
    # 确定real子类ID
    real_subclass_ids = []
    if dataset_name in ['ASVspoof2019', 'Codecfake']:
        # 这两个数据集中real固定为子类0
        real_subclass_ids = [0] if 0 in subclass_ratios else []
    elif dataset_name == 'CFAD':
        # CFAD中real source名称列表
        real_sources = {'aishell1', 'aishell3', 'magicread', 'thchs30'}
        for sid, name in id_to_attack.items():
            if name.lower() in real_sources:
                real_subclass_ids.append(sid)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"真实数据子类分布统计 ({dataset_name})")
    logger.info(f"总样本数: {total}")
    logger.info(f"Real子类ID: {real_subclass_ids}")
    logger.info(f"子类分布:")
    for subclass_id in sorted(subclass_ratios.keys()):
        ratio = subclass_ratios[subclass_id]
        count = subclass_counts[subclass_id]
        name = id_to_attack.get(subclass_id, f"subclass_{subclass_id}")
        label_type = "real" if subclass_id in real_subclass_ids else "fake"
        logger.info(f"  Subclass {subclass_id} ({name}, {label_type}): "
                   f"{count} ({ratio*100:.2f}%)")
    logger.info(f"{'='*60}")
    
    return subclass_ratios, subclass_counts, attack_to_id, real_subclass_ids


def compute_binary_ratio(subclass_ratios):
    """从子类比例计算二分类real/fake比例"""
    real_ratio = subclass_ratios.get(0, 0.0)
    fake_ratio = 1.0 - real_ratio
    
    logger.info(f"二分类比例: Real={real_ratio*100:.2f}%, Fake={fake_ratio*100:.2f}%")
    
    return real_ratio, fake_ratio


class DomainPrototypeGenerator:
    """
    域原型生成器
    按子类分布生成160维SSL-AASIST特征
    """
    
    def __init__(
        self,
        checkpoint_path,
        dataset_name,
        num_subclasses,
        attack_to_id,
        feature_dim=160,
        device='cuda',
    ):
        self.device = device
        self.dataset_name = dataset_name
        self.feature_dim = feature_dim
        self.num_subclasses = num_subclasses
        self.attack_to_id = attack_to_id
        
        # 加载生成模型
        self.model = SSLAASISTFeatureGenerator(
            feature_dim=feature_dim,
            num_subclasses=num_subclasses,
            cond_dim=256,
        ).to(device)
        
        # 加载权重（与LCNN一致，只加载模型权重）
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
            logger.info(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded model weights from {checkpoint_path}")
        
        self.model.eval()
        
        # 统计参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model total parameters: {total_params:,}")
    
    @torch.no_grad()
    def generate_by_subclass_distribution(
        self,
        subclass_distribution,
        n_timesteps=10,
        temperature=1.0,
        batch_size=256,
        real_subclass_ids=None,  # 新增参数
    ):
        """
        按子类分布生成特征
        Args:
            real_subclass_ids: list of subclass IDs that are considered real (bonafide)
        """
        total_samples = sum(subclass_distribution.values())
        logger.info(f"Generating {total_samples} samples for {self.dataset_name}...")
        logger.info("Subclass distribution:")
        for subclass_id, count in sorted(subclass_distribution.items()):
            logger.info(f"  Subclass {subclass_id}: {count} ({count/total_samples*100:.1f}%)")
        
        all_features = []
        all_subclasses = []
        
        for subclass_id, count in sorted(subclass_distribution.items()):
            generated = 0
            pbar = tqdm(total=count, desc=f"Subclass {subclass_id}")
            
            while generated < count:
                current_batch = min(batch_size, count - generated)
                subclasses = torch.full((current_batch,), subclass_id, 
                                    dtype=torch.long, device=self.device)
                
                shape = (current_batch, self.feature_dim, 1)
                mask = torch.ones(current_batch, 1, 1, device=self.device)
                
                features = self.model.generate(
                    shape=shape,
                    mask=mask,
                    n_timesteps=n_timesteps,
                    temperature=temperature,
                    subclasses=subclasses,
                )
                
                if features.dim() == 3:
                    features = features.squeeze(-1)
                if features.shape[1] != self.feature_dim:
                    features = features.transpose(1, 2)
                
                all_features.append(features.cpu())
                all_subclasses.append(torch.full((current_batch,), subclass_id, dtype=torch.long))
                
                generated += current_batch
                pbar.update(current_batch)
            pbar.close()
        
        features = torch.cat(all_features, dim=0)
        subclasses = torch.cat(all_subclasses, dim=0)
        
        # 打乱顺序
        perm = torch.randperm(total_samples)
        features = features[perm]
        subclasses = subclasses[perm]
        
        # 计算二分类标签：如果提供了real_subclass_ids，则根据是否在其中判断
        if real_subclass_ids is not None:
            real_set = set(real_subclass_ids)
            labels = torch.tensor([0 if sid.item() in real_set else 1 for sid in subclasses], dtype=torch.long)
        else:
            # 向后兼容：假设real就是子类0
            labels = (subclasses > 0).long()
        
        # 计算域中心
        center = features.mean(dim=0)
        
        # 计算各子类中心
        subclass_centers = {}
        for subclass_id in range(self.num_subclasses):
            mask = subclasses == subclass_id
            if mask.sum() > 0:
                subclass_centers[int(subclass_id)] = features[mask].mean(dim=0)
        
        logger.info(f"Generation completed: {features.shape}")
        logger.info(f"Binary labels: real={int((labels==0).sum())}, fake={int((labels==1).sum())}")
        
        return {
            'features': features,
            'subclasses': subclasses,
            'labels': labels,
            'center': center,
            'subclass_centers': subclass_centers,
        }
    
    def generate_ot_moe_format(
        self,
        total_samples,
        subclass_ratios,
        real_subclass_ids,   # 新增参数，必须提供
        real_ratio=None,
        n_timesteps=10,
        temperature=1.0,
        batch_size=256,
    ):
        """
        生成OT-MOE格式数据
        Args:
            real_subclass_ids: list of subclass IDs that are real (bonafide)
        """
        # 计算real样本总数
        real_set = set(real_subclass_ids)
        real_ratio = sum(subclass_ratios.get(sid, 0) for sid in real_set)
        
        n_real = int(total_samples * real_ratio)
        n_fake = total_samples - n_real
        
        logger.info(f"生成 {total_samples} 个样本:")
        logger.info(f"  Real (0): {n_real} ({real_ratio*100:.2f}%)")
        logger.info(f"  Fake total: {n_fake} ({(1-real_ratio)*100:.2f}%)")
        
        # 构建子类分布字典
        subclass_dist = {}
        
        # 分配real子类（按原始比例）
        real_subclass_ratios = {sid: subclass_ratios[sid] for sid in real_set if sid in subclass_ratios}
        real_total = sum(real_subclass_ratios.values())
        if real_total > 0:
            remaining_real = n_real
            sorted_real = sorted(real_subclass_ratios.keys())
            for i, sid in enumerate(sorted_real):
                ratio = real_subclass_ratios[sid] / real_total
                if i == len(sorted_real) - 1:
                    count = remaining_real
                else:
                    count = int(n_real * ratio)
                    remaining_real -= count
                subclass_dist[sid] = count
                source_name = [k for k, v in self.attack_to_id.items() if v == sid][0] if hasattr(self, 'attack_to_id') else f"source_{sid}"
                logger.info(f"  Real Subclass {sid} ({source_name}): {count}")
        
        # 分配fake子类
        fake_subclass_ratios = {k: v for k, v in subclass_ratios.items() if k not in real_set}
        fake_total = sum(fake_subclass_ratios.values())
        if fake_total > 0:
            normalized_fake_ratios = {k: v / fake_total for k, v in fake_subclass_ratios.items()}
            remaining_fake = n_fake
            sorted_fake = sorted(normalized_fake_ratios.keys())
            for i, sid in enumerate(sorted_fake):
                ratio = normalized_fake_ratios[sid]
                if i == len(sorted_fake) - 1:
                    count = remaining_fake
                else:
                    count = int(n_fake * ratio)
                    remaining_fake -= count
                subclass_dist[sid] = count
                attack_name = [k for k, v in self.attack_to_id.items() if v == sid][0] if hasattr(self, 'attack_to_id') else f"attack_{sid}"
                logger.info(f"  Fake Subclass {sid} ({attack_name}): {count}")
        
        return self.generate_by_subclass_distribution(
            subclass_dist,
            n_timesteps=n_timesteps,
            temperature=temperature,
            batch_size=batch_size,
            real_subclass_ids=real_subclass_ids,  # 传递real子类ID
        )

class FeatureAnalyzer:
    """
    特征分析器：可视化生成特征与原始特征的分布对比
    保持与LCNN代码一致的可视化功能
    """
    
    def __init__(
        self,
        device='cuda',
        viz_dir='./visualizations',
    ):
        self.device = device
        self.viz_dir = Path(viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_features = {}  # 缓存加载的特征
        self.dataset_name = "Unknown"  # 默认数据集名称
    
    def load_original_features(
        self,
        proto_txt: str,
        root_dir: str,
        ssl_aasist_checkpoint: str,
        num_samples: Optional[int] = None,
        batch_size: int = 32,
        use_all_data: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从原始音频提取SSL-AASIST 160维特征
        优化：减少磁盘IO，恢复原始高效路径解析逻辑
        """
        # 检查缓存
        cache_key = f"{proto_txt}_{root_dir}_{use_all_data}_{num_samples}"
        if cache_key in self.loaded_features:
            logger.info("使用缓存的原始特征...")
            return self.loaded_features[cache_key]

        # 加载SSL-AASIST特征提取器
        logger.info("Loading SSL-AASIST feature extractor...")
        ssl_model = SSLAASISTFeatureExtractor(
            device=self.device,
            ssl_path='/data1/tjj/XLSR-MOE/xlsr2_300m.pt',
            pretrained_path=ssl_aasist_checkpoint,
            freeze=True
        ).to(self.device).eval()

        # 冻结参数
        for param in ssl_model.parameters():
            param.requires_grad = False

        # ========== 优化：使用高效路径解析（恢复原始逻辑） ==========
        dataset_suffix_map = {
            'ASVspoof2019': '.wav',
            'CFAD': '.wav',
            'Codecfake': '.wav'
        }
        correct_suffix = dataset_suffix_map.get(self.dataset_name, '.wav')
        
        # 加载协议
        labels_dict, attack_types, file_list, paths, attack_to_id = \
            load_dataset_protocol(self.dataset_name, proto_txt, root_dir, correct_suffix)

        # ========== 关键优化：批量构建路径，减少os.path.exists调用 ==========
        # 先尝试使用协议返回的paths（最高效）
        valid_paths = []
        valid_labels = []
        valid_subclasses = []
        failed_files = []

        # 第一轮：直接检查协议返回的路径（大多数情况）
        for fname, path in zip(file_list, paths):
            if os.path.exists(path):
                valid_paths.append(path)
                valid_labels.append(labels_dict[fname])
                valid_subclasses.append(attack_types[fname])
            else:
                failed_files.append((fname, path))

        # 第二轮：仅对失败的文件尝试其他扩展名（降级逻辑）
        if failed_files:
            logger.warning(f"第一轮匹配失败 {len(failed_files)} 个文件，尝试扩展名回退...")
            extensions_order = ['.flac', '.wav', '.pt'] if self.dataset_name == 'ASVspoof2019' else ['.wav', '.flac', '.pt']
            
            for fname, orig_path in failed_files:
                base_dir = os.path.dirname(orig_path)
                fname_base = os.path.splitext(fname)[0]
                
                found = False
                for ext in extensions_order:
                    test_path = os.path.join(base_dir, fname_base + ext)
                    if os.path.exists(test_path):
                        valid_paths.append(test_path)
                        valid_labels.append(labels_dict[fname])
                        valid_subclasses.append(attack_types[fname])
                        found = True
                        break
                
                if not found:
                    logger.warning(f"文件不存在: {fname} (尝试路径: {orig_path})")

        total_available = len(valid_paths)
        logger.info(f"找到 {total_available} 个有效音频文件 (从 {len(file_list)} 个协议文件中)")

        # 采样或全量（原有逻辑不变）
        if use_all_data:
            logger.info(f"使用全部原始数据: {total_available} 条样本")
            selected_indices = list(range(total_available))
        else:
            if num_samples and num_samples < total_available:
                selected_indices = np.random.choice(total_available, num_samples, replace=False).tolist()
                logger.info(f"从 {total_available} 条原始数据中随机采样 {num_samples} 条")
            else:
                selected_indices = list(range(total_available))
                logger.info(f"使用全部原始数据: {total_available} 条样本")

        valid_paths = [valid_paths[i] for i in selected_indices]
        valid_labels = [valid_labels[i] for i in selected_indices]
        valid_subclasses = [valid_subclasses[i] for i in selected_indices]

        # 批量提取特征（原有逻辑不变，确保没有额外IO）
        logger.info(f"从 {len(valid_paths)} 个原始音频提取SSL-AASIST特征...")
        all_features = []
        all_labels = []
        all_subclasses = []
        failed_count = 0

        for i in tqdm(range(0, len(valid_paths), batch_size), desc="Extracting original features"):
            batch_paths = valid_paths[i:i+batch_size]
            batch_labels = valid_labels[i:i+batch_size]
            batch_subclasses = valid_subclasses[i:i+batch_size]

            batch_audio = []
            valid_indices = []

            for idx, path in enumerate(batch_paths):
                try:
                    # 使用统一的音频加载函数（这里可能有IO，但不可避免）
                    audio = load_audio_fixed_length(
                        path,
                        target_length=4.0375,
                        target_sr=16000
                    )
                    batch_audio.append(audio)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.warning(f"加载失败 {path}: {e}")
                    failed_count += 1
                    continue

            if batch_audio:
                # 堆叠并转移到GPU
                audio_tensor = torch.stack(batch_audio).to(self.device)

                # 批量提取特征
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        feats = ssl_model(audio_tensor)  # [B, 160]

                # 关键：保持GPU张量直到最后，避免频繁同步
                all_features.append(feats.cpu().numpy())

                # 只添加成功加载的标签
                for idx in valid_indices:
                    all_labels.append(batch_labels[idx])
                    all_subclasses.append(batch_subclasses[idx])

        if not all_features:
            raise RuntimeError(f"没有成功提取任何特征，请检查数据路径")

        features = np.concatenate(all_features, axis=0)
        labels = np.array(all_labels)
        subclasses = np.array(all_subclasses)

        # 统计分布
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"原始特征提取完成: {features.shape}")
        logger.info("原始数据类别分布:")
        for label, count in zip(unique, counts):
            label_name = "real" if label == 0 else "fake"
            logger.info(f"  Class {label} ({label_name}): {count} 条")

        if failed_count > 0:
            logger.warning(f"加载失败: {failed_count} 条")

        # 缓存结果
        self.loaded_features[cache_key] = (features, labels, subclasses)

        return features, labels, subclasses
    
    def visualize_distribution(
        self,
        generated_features,
        generated_subclasses,
        generated_labels,
        original_features=None,
        original_labels=None,
        method='tsne',
        save_name='distribution.png',
        subclass_names=None,
    ):
        """
        可视化特征分布
        支持两种模式：
        1. 仅生成特征（向后兼容）
        2. 原始vs生成对比（当提供original_*参数时，与LCNN一致）
        """
        import umap
        
        logger.info(f"Visualizing with {method.upper()}...")
        
        # 确保所有输入都是numpy数组
        gen_features = np.asarray(generated_features)
        gen_subclasses = np.asarray(generated_subclasses)
        gen_labels = np.asarray(generated_labels)
        
        # 检查是否需要对比模式
        has_original = original_features is not None and original_labels is not None
        
        if has_original:
            orig_features = np.asarray(original_features)
            orig_labels = np.asarray(original_labels)
            
            # 确保维度一致
            if orig_features.ndim != 2:
                raise ValueError(f"Original features must be 2D, got shape {orig_features.shape}")
            if gen_features.ndim != 2:
                raise ValueError(f"Generated features must be 2D, got shape {gen_features.shape}")
            
            # 合并所有特征进行对比
            all_features = np.concatenate([orig_features, gen_features], axis=0)
            all_sources = np.array(['original'] * len(orig_features) + ['generated'] * len(gen_features))
            all_labels = np.concatenate([orig_labels, gen_labels])
            
            # 降维
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)-1))
            elif method == 'umap':
                reducer = umap.UMAP(n_components=2, random_state=42)
            else:
                reducer = PCA(n_components=2)
            
            embedded = reducer.fit_transform(all_features)
            
            # 创建对比可视化
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            # 左图：按数据来源着色（original vs generated）
            ax1 = axes[0]
            colors_source = {'original': '#1f77b4', 'generated': '#ff7f0e'}
            for source in ['original', 'generated']:
                mask = all_sources == source
                if mask.sum() > 0:
                    ax1.scatter(
                        embedded[mask, 0], embedded[mask, 1],
                        c=colors_source[source],
                        alpha=0.5,
                        s=20,
                        label=f'{source} (n={mask.sum()})'
                    )
            ax1.legend()
            ax1.set_title(f'{self.dataset_name} - Original vs Generated')
            ax1.grid(True, alpha=0.3)
            
            # 右图：按类别着色（0=real, 1=fake）
            ax2 = axes[1]
            colors_binary = {0: '#2ca02c', 1: '#d62728'}  # green, red
            labels_binary = {0: 'real (bonafide)', 1: 'fake (spoof)'}
            
            for label in [0, 1]:
                mask = all_labels == label
                if mask.sum() > 0:
                    ax2.scatter(
                        embedded[mask, 0], embedded[mask, 1],
                        c=colors_binary[label],
                        alpha=0.5,
                        s=20,
                        label=f"{labels_binary[label]} (n={mask.sum()})"
                    )
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.set_title(f'{self.dataset_name} - By Binary Class')
            ax2.grid(True, alpha=0.3)
            
        else:
            # 仅生成特征模式（原有功能）
            if gen_features.ndim != 2:
                raise ValueError(f"Features must be 2D, got shape {gen_features.shape}")
            
            # 降维
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(gen_features)-1))
            elif method == 'umap':
                reducer = umap.UMAP(n_components=2, random_state=42)
            else:
                reducer = PCA(n_components=2)
            
            embedded = reducer.fit_transform(gen_features)
            
            # 创建图形
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            # 左图：按二分类着色（real/fake）
            ax1 = axes[0]
            colors_binary = {0: '#1f77b4', 1: '#ff7f0e'}
            labels_binary = {0: 'real (bonafide)', 1: 'fake (spoof)'}
            
            for label in [0, 1]:
                mask = gen_labels == label
                if mask.sum() > 0:
                    ax1.scatter(
                        embedded[mask, 0], embedded[mask, 1],
                        c=colors_binary[label],
                        alpha=0.5,
                        s=20,
                        label=f"{labels_binary[label]} (n={mask.sum()})"
                    )
            
            ax1.legend()
            ax1.set_title(f'{self.dataset_name} - Binary Classification (Real vs Fake)')
            ax1.grid(True, alpha=0.3)
            
            # 右图：按子类着色
            ax2 = axes[1]
            colors_subclass = plt.cm.tab10(np.linspace(0, 1, len(np.unique(gen_subclasses))))
            
            for idx, subclass_id in enumerate(sorted(np.unique(gen_subclasses))):
                mask = gen_subclasses == subclass_id
                if mask.sum() > 0:
                    name = subclass_names.get(subclass_id, f"subclass_{subclass_id}") if subclass_names else f"subclass_{subclass_id}"
                    ax2.scatter(
                        embedded[mask, 0], embedded[mask, 1],
                        c=[colors_subclass[idx]],
                        alpha=0.5,
                        s=20,
                        label=f'{name} (n={mask.sum()})'
                    )
            
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.set_title(f'{self.dataset_name} - By Subclass (Attack Type)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.viz_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved: {save_path}")
    
    def compute_statistics(self, results, original_features=None, original_labels=None, subclass_names=None):
        gen_features = results['features'].numpy()
        gen_subclasses = results['subclasses'].numpy()
        gen_labels = results['labels'].numpy()
        
        has_original = original_features is not None and original_labels is not None
        
        if has_original:
            orig_features = np.asarray(original_features)
            orig_labels = np.asarray(original_labels)
            
            # 【关键修复】提前清洗，确保无NaN/Inf
            orig_features = np.nan_to_num(orig_features, nan=0.0, posinf=1e5, neginf=-1e5)
            
            # 再次检查，确保清洗生效
            if not np.isfinite(orig_features).all():
                logger.warning("仍然存在非有限值，进行二次清洗")
                orig_features = np.where(np.isfinite(orig_features), orig_features, 0.0)
            
            # 现在可以安全计算统计量
            stats = {
                'sample_counts': {
                    'original_total': int(len(orig_features)),
                    'generated_total': int(len(gen_features)),
                    'by_class': {}
                },
                'overall_stats': {
                    'original_mean': float(np.mean(orig_features)),
                    'original_std': float(np.std(orig_features)),  # 现在应该正常
                    'generated_mean': float(np.mean(gen_features)),
                    'generated_std': float(np.std(gen_features)),
                }
            }
            
            # 按类别统计和计算Wasserstein距离
            for label in [0, 1]:
                orig_mask = orig_labels == label
                gen_mask = gen_labels == label
                
                label_name = "real" if label == 0 else "fake"
                
                stats['sample_counts']['by_class'][label_name] = {
                    'original': int(orig_mask.sum()),
                    'generated': int(gen_mask.sum()),
                }
                
                # 计算Wasserstein距离
                if orig_mask.sum() > 0 and gen_mask.sum() > 0:
                    w_distances = []
                    for dim in range(min(160, gen_features.shape[1])):
                        # 注意：Wasserstein 距离计算同样需要清洗数据，否则也会报错
                        orig_dim = orig_features[orig_mask, dim]
                        gen_dim = gen_features[gen_mask, dim]
                        # 再次清洗（防御性编程）
                        orig_dim = np.nan_to_num(orig_dim, nan=0.0)
                        gen_dim = np.nan_to_num(gen_dim, nan=0.0)
                        w_dist = wasserstein_distance(orig_dim, gen_dim)
                        w_distances.append(w_dist)
                    
                    stats[f'class_{label}_wasserstein_mean'] = float(np.mean(w_distances))
                    stats[f'class_{label}_wasserstein_std'] = float(np.std(w_distances))
        else:
            # 仅生成特征统计（原有功能）
            stats = {
                'total_samples': len(gen_features),
                'feature_dim': gen_features.shape[1],
                'by_subclass': {},
                'by_binary_class': {
                    'real': int((gen_labels == 0).sum()),
                    'fake': int((gen_labels == 1).sum()),
                },
                'feature_stats': {
                    'mean': float(gen_features.mean()),
                    'std': float(gen_features.std()),
                    'min': float(gen_features.min()),
                    'max': float(gen_features.max()),
                }
            }
            
            # 按子类统计
            for subclass_id in sorted(np.unique(gen_subclasses)):
                mask = gen_subclasses == subclass_id
                subclass_features = gen_features[mask]
                name = subclass_names.get(subclass_id, f"subclass_{subclass_id}") if subclass_names else f"subclass_{subclass_id}"
                
                stats['by_subclass'][name] = {
                    'count': int(mask.sum()),
                    'mean': float(subclass_features.mean()),
                    'std': float(subclass_features.std()),
                }
        
        # 保存统计信息
        stats_path = self.viz_dir / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved: {stats_path}")
        
        # 打印关键指标
        if has_original:
            logger.info("分布对比统计:")
            for label in [0, 1]:
                label_name = "real" if label == 0 else "fake"
                w_mean = stats.get(f'class_{label}_wasserstein_mean', 'N/A')
                logger.info(f"  {label_name}: Wasserstein距离 = {w_mean:.4f}" if w_mean != 'N/A' else f"  {label_name}: Wasserstein距离 = N/A")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Generate Domain Prototypes for OT-MOE')
    
    # 模型和数据集
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained generator checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ASVspoof2019', 'CFAD', 'Codecfake'])
    
    # 数据路径（用于获取原始分布统计）
    parser.add_argument('--proto_txt', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--suffix', type=str, default=None)
    
    # 新增：SSL-AASIST预训练模型路径（用于提取原始特征）
    parser.add_argument('--ssl_aasist_checkpoint', type=str,
                       default='/data1/tjj/XLSR-MOE/models_ssl/shared_aasist.pth',
                       help='SSL-AASIST模型路径（用于提取原始特征进行对比） ，3专家融合AASIST权重路径，必须与训练时使用的shared_aasist.pth一致，'
         '格式: XLS-R官方权重 + 3专家AASIST平均融合')
    
    # 新增：是否进行对比可视化
    parser.add_argument('--compare_with_original', action='store_true', default=False,
                       help='是否提取原始特征并进行对比可视化')
    parser.add_argument('--num_original_compare', type=int, default=5000,
                       help='用于对比的原始特征数量')
    parser.add_argument('--use_all_original', action='store_true', default=False,
                       help='使用全部原始数据进行对比，而不是采样')
    
    # 生成参数
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--real_ratio', type=float, default=None,
                       help='Ratio of real samples in generated data')
    parser.add_argument('--n_timesteps', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=256)
    
    # 输出路径
    parser.add_argument('--output_dir', type=str, 
                       default='/data1/tjj/XLSR-MOE/FM/domain_cache')
    parser.add_argument('--viz_dir', type=str, default=None,
                       help='Visualization directory (default: output_dir/visualizations)')
    parser.add_argument('--viz_methods', type=str, default='tsne,umap,pca')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--use_real_distribution', action='store_true', default=True,
                       help='使用真实子类分布生成（默认True）')
    
    args = parser.parse_args()
    
    # 硬编码数据集路径
    dataset_configs = {
        'ASVspoof2019': {
            'proto_txt': '/data1/tjj/XLSR-MOE/DQ/dq/asv19_dq60.txt',
            'root_dir': '/data0/yyj/dataset/ASVspoof2019_LA/ASVspoof2019_LA/ASVspoof2019_LA_train/wav/',
            'suffix': '.wav',
        },
        'CFAD': {
            'proto_txt': '/data1/tjj/XLSR-MOE/DQ/dq/cfad_dq60.txt',
            'root_dir': '/data0/cff/CFAD/clean/train/',
            'suffix': '.wav',
        },
        'Codecfake': {
            'proto_txt': '/data1/tjj/XLSR-MOE/DQ/dq/codecfake_dq60.txt',
            'root_dir': '/data0/yyj/dataset/Codecfake/train/',
            'suffix': '.wav',
        },
    }
    
    # 使用硬编码或命令行参数
    config = dataset_configs[args.dataset]
    proto_txt = args.proto_txt or config['proto_txt']
    root_dir = args.root_dir or config['root_dir']
    suffix = args.suffix or config['suffix']
    
    # 设置可视化目录
    viz_dir = args.viz_dir or os.path.join(args.output_dir, 'visualizations', args.dataset)
    
    logger.info(f"{'='*60}")
    logger.info(f"Generating domain prototype for {args.dataset}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Num samples: {args.num_samples}")
    if args.compare_with_original:
        logger.info(f"对比模式: 将提取原始特征进行对比")
    logger.info(f"{'='*60}")
    
    # 加载协议获取子类信息
    _, _, _, _, attack_to_id = load_dataset_protocol(args.dataset, proto_txt, root_dir, suffix)
    num_subclasses = len(attack_to_id)
    
    logger.info(f"Number of subclasses: {num_subclasses}")
    logger.info(f"Attack type mapping: {attack_to_id}")
    
    # 计算真实子类分布
    subclass_ratios, subclass_counts, attack_to_id, real_subclass_ids = compute_real_subclass_distribution(
        proto_txt=proto_txt,
        root_dir=root_dir,
        suffix=suffix,
        dataset_name=args.dataset,
    )
    
    # 创建生成器
    generator = DomainPrototypeGenerator(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        num_subclasses=num_subclasses,
        attack_to_id=attack_to_id, 
        feature_dim=160,
        device=args.device,
    )
    
    # 生成OT-MOE格式数据
    results = generator.generate_ot_moe_format(
        total_samples=args.num_samples,
        subclass_ratios=subclass_ratios,
        real_subclass_ids=real_subclass_ids,
        real_ratio=None,
        n_timesteps=args.n_timesteps,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )
    
    # --- 关键修复：处理特征维度 ---
    # 将 features 从 [N, 160, 1] 转换为 [N, 160]
    gen_features = results['features']
    if gen_features.dim() == 3 and gen_features.shape[2] == 1:
        gen_features = gen_features.squeeze(-1) # 去除最后一个维度
    elif gen_features.dim() == 3 and gen_features.shape[1] == 1:
        gen_features = gen_features.squeeze(1)  # 如果是中间维度为1，则去除中间维度

    # 同步更新 results 字典中的 features
    results['features'] = gen_features
    # --- 关键修复结束 ---

    # 保存为OT-MOE格式
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    domain_name = f"{args.dataset}_generated"
    cache_file = output_dir / f"{domain_name}_n{args.num_samples}.pth"
    
    torch.save({
        "samples": results['features'],      # [N, 160]
        "labels": results['labels'],          # [N] 0=real, 1=fake
        "subclasses": results['subclasses'],  # [N] 子类ID
        "center": results['center'],          # [160]
        "subclass_centers": results['subclass_centers'],
        "attack_to_id": attack_to_id,
    }, cache_file)
    
    logger.info(f"Domain prototype saved: {cache_file}")
    
    # 保存统计信息
    strata_stats = {
        'bonafide': int((results['labels'] == 0).sum().item()),
        'fake': int((results['labels'] == 1).sum().item()),
    }
    stats_file = output_dir / f"{domain_name}_n{args.num_samples}_stats.pth"
    torch.save(strata_stats, stats_file)
    
    # 保存numpy格式
    np.savez(
        output_dir / f"{domain_name}_n{args.num_samples}.npz",
        features=results['features'].numpy(),
        labels=results['labels'].numpy(),
        subclasses=results['subclasses'].numpy(),
        center=results['center'].numpy(),
    )
    
    # ==================== 可视化部分（与LCNN一致） ====================
    analyzer = FeatureAnalyzer(device=args.device, viz_dir=viz_dir)
    analyzer.dataset_name = args.dataset
    
    # 反转映射获取子类名称
    id_to_attack = {v: k for k, v in attack_to_id.items()}
    
    # 检查是否需要对比模式
    if args.compare_with_original:
        logger.info(f"{'='*60}")
        logger.info("步骤: 提取原始特征进行对比...")
        
        # 加载原始特征
        orig_features, orig_labels, orig_subclasses = analyzer.load_original_features(
            proto_txt=proto_txt,
            root_dir=root_dir,
            ssl_aasist_checkpoint=args.ssl_aasist_checkpoint,
            num_samples=args.num_original_compare,
            batch_size=32,
            use_all_data=args.use_all_original,
        )
        
        # 对比模式可视化
        viz_methods = [m.strip() for m in args.viz_methods.split(',')]
        for method in viz_methods:
            analyzer.visualize_distribution(
                generated_features=results['features'].numpy(),
                generated_subclasses=results['subclasses'].numpy(),
                generated_labels=results['labels'].numpy(),
                original_features=orig_features,      # 新增：原始特征
                original_labels=orig_labels,          # 新增：原始标签
                method=method,
                save_name=f'distribution_comparison_{method}.png',  # 对比模式命名
                subclass_names=id_to_attack,
            )
        
        # 对比模式统计（包含Wasserstein距离）
        stats = analyzer.compute_statistics(
            results,
            original_features=orig_features,
            original_labels=orig_labels,
            subclass_names=id_to_attack,
        )
        
    else:
        # 仅生成特征可视化（原有功能）
        viz_methods = [m.strip() for m in args.viz_methods.split(',')]
        for method in viz_methods:
            analyzer.visualize_distribution(
                generated_features=results['features'].numpy(),
                generated_subclasses=results['subclasses'].numpy(),
                generated_labels=results['labels'].numpy(),
                method=method,
                save_name=f'distribution_{method}.png',
                subclass_names=id_to_attack,
            )
        
        # 仅生成特征统计
        stats = analyzer.compute_statistics(results, subclass_names=id_to_attack)
    
    logger.info(f"{'='*60}")
    logger.info("Generation completed!")
    logger.info(f"Total samples: {stats.get('total_samples', stats.get('generated_total', args.num_samples))}")
    if 'by_binary_class' in stats:
        logger.info(f"Real: {stats['by_binary_class'].get('real', stats['by_binary_class'].get('bonafide', 'N/A'))}, "
                   f"Fake: {stats['by_binary_class'].get('fake', 'N/A')}")
    logger.info(f"Output: {cache_file}")
    logger.info(f"Visualizations: {viz_dir}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()