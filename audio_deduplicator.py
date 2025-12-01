#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频级别去重模块
在音频进入 Whisper 识别之前，检测并跳过重复或高度相似的音频块
这样可以减少模型的计算量，提高处理速度

技术方案：
1. 音频特征提取：RMS能量、频谱质心、过零率
2. 特征向量比较：使用余弦相似度快速比较
3. 滑动窗口缓存：维护最近处理的音频特征历史
"""

import numpy as np
import time
from typing import Optional, Tuple, List, Dict
from collections import deque


class AudioDeduplicator:
    """
    音频级别去重器
    
    在音频进入 ASR 模型之前，检测重复或高度相似的音频块
    避免对相同或相似的音频进行重复识别，提高处理速度
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.95,
                 time_window: float = 3.0,
                 min_audio_length: float = 0.1,
                 enable: bool = True):
        """
        Args:
            similarity_threshold: 音频相似度阈值（0-1），超过此值认为是重复
                                 0.95 = 非常相似（几乎相同）
                                 0.90 = 高度相似
                                 0.85 = 相似
            time_window: 时间窗口（秒），只在此时间窗口内进行去重
            min_audio_length: 最小音频长度（秒），短于此长度的音频不进行去重
            enable: 是否启用音频去重
        """
        self.similarity_threshold = similarity_threshold
        self.time_window = time_window
        self.min_audio_length = min_audio_length
        self.enabled = enable
        
        # 音频特征历史：[(特征向量, 时间戳, 音频长度), ...]
        self.feature_history = deque(maxlen=100)  # 最多保存100个特征
        
        # 统计信息
        self.stats = {
            'total_checked': 0,
            'skipped_duplicate': 0,
            'skipped_similar': 0,
            'passed': 0,
            'total_audio_time_skipped': 0.0  # 跳过的音频总时长（秒）
        }
    
    def _extract_features(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        提取音频特征向量
        
        特征包括：
        1. RMS能量（均方根）
        2. 频谱质心（spectral centroid）
        3. 过零率（zero crossing rate）
        4. 频谱滚降点（spectral rolloff）
        5. 频谱带宽（spectral bandwidth）
        
        Args:
            audio: 音频数据（numpy数组，float32）
            sample_rate: 采样率（Hz）
            
        Returns:
            特征向量（5维）
        """
        if len(audio) == 0:
            return np.zeros(5)
        
        # 初始化变量
        rms = 0.0
        spectral_centroid = 0.0
        zcr = 0.0
        spectral_rolloff = 0.0
        spectral_bandwidth = 0.0
        magnitude = None
        frequencies = None
        
        try:
            # 1. RMS能量
            rms = np.sqrt(np.mean(audio ** 2))
        except:
            rms = 0.0
        
        # 2. 频谱质心（需要FFT）
        try:
            # 使用短时傅里叶变换（STFT）
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            if len(magnitude) > 0:
                # 频率轴
                frequencies = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
                # 频谱质心：加权平均频率
                spectral_centroid = np.sum(frequencies * magnitude) / (np.sum(magnitude) + 1e-10)
                # 归一化到0-1范围（假设最大频率为采样率的一半）
                spectral_centroid = spectral_centroid / (sample_rate / 2)
            else:
                spectral_centroid = 0.0
        except:
            spectral_centroid = 0.0
        
        # 3. 过零率（ZCR）
        try:
            # 计算符号变化的次数
            if len(audio) > 1:
                sign_changes = np.sum(np.diff(np.signbit(audio)))
                zcr = sign_changes / len(audio)
            else:
                zcr = 0.0
        except:
            zcr = 0.0
        
        # 4. 频谱滚降点（spectral rolloff）
        # 累积能量达到85%时的频率
        try:
            if magnitude is not None and frequencies is not None and len(magnitude) > 0:
                cumsum = np.cumsum(magnitude)
                total_energy = cumsum[-1]
                if total_energy > 1e-10:
                    rolloff_threshold = 0.85 * total_energy
                    rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
                    if len(rolloff_idx) > 0:
                        rolloff_freq = frequencies[rolloff_idx[0]]
                        spectral_rolloff = rolloff_freq / (sample_rate / 2)
                    else:
                        spectral_rolloff = 1.0
                else:
                    spectral_rolloff = 0.0
            else:
                spectral_rolloff = 0.0
        except:
            spectral_rolloff = 0.0
        
        # 5. 频谱带宽（spectral bandwidth）
        # 频谱围绕质心的分散程度
        try:
            if magnitude is not None and frequencies is not None and len(magnitude) > 0 and spectral_centroid > 0:
                # 计算标准差
                centroid_freq = spectral_centroid * sample_rate / 2
                spectral_bandwidth = np.sqrt(np.sum(((frequencies - centroid_freq) ** 2) * magnitude) / (np.sum(magnitude) + 1e-10))
                # 归一化
                spectral_bandwidth = spectral_bandwidth / (sample_rate / 2)
            else:
                spectral_bandwidth = 0.0
        except:
            spectral_bandwidth = 0.0
        
        # 组合特征向量
        try:
            features = np.array([
                rms,
                spectral_centroid,
                zcr,
                spectral_rolloff,
                spectral_bandwidth
            ], dtype=np.float32)
            
            # 归一化特征（避免某些特征值过大影响相似度计算）
            # 使用简单的归一化：除以最大值（避免除零）
            max_val = np.max(np.abs(features))
            if max_val > 1e-10:
                features = features / max_val
            else:
                # 如果所有特征都是0，返回零向量
                features = np.zeros(5, dtype=np.float32)
        except:
            # 如果特征提取完全失败，返回零向量
            features = np.zeros(5, dtype=np.float32)
        
        return features
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个特征向量的余弦相似度
        
        Args:
            vec1: 特征向量1
            vec2: 特征向量2
            
        Returns:
            相似度（0-1），1表示完全相同
        """
        # 避免除零
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        # 余弦相似度
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        
        # 归一化到0-1范围（余弦相似度范围是-1到1）
        similarity = (similarity + 1.0) / 2.0
        
        return float(similarity)
    
    def _clean_history(self, current_time: float):
        """清理超出时间窗口的历史记录"""
        cutoff_time = current_time - self.time_window
        self.feature_history = deque(
            [(feat, ts, length) for feat, ts, length in self.feature_history if ts > cutoff_time],
            maxlen=100
        )
    
    def should_skip(self, 
                    audio: np.ndarray, 
                    sample_rate: int = 16000,
                    current_time: Optional[float] = None) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        判断是否应该跳过此音频块（不发送到 ASR 模型）
        
        Args:
            audio: 音频数据（numpy数组，float32）
            sample_rate: 采样率（Hz）
            current_time: 当前时间戳（如果为None则使用当前时间）
            
        Returns:
            (是否跳过, 跳句原因, 详细信息)
        """
        if not self.enabled:
            return False, None, None
        
        if current_time is None:
            current_time = time.time()
        
        self.stats['total_checked'] += 1
        
        # 检查音频长度
        audio_length = len(audio) / sample_rate
        if audio_length < self.min_audio_length:
            # 太短的音频不进行去重，直接通过
            self.stats['passed'] += 1
            return False, None, None
        
        # 清理历史记录
        self._clean_history(current_time)
        
        # 提取当前音频特征
        try:
            current_features = self._extract_features(audio, sample_rate)
        except Exception as e:
            # 特征提取失败，不跳过（保守策略）
            self.stats['passed'] += 1
            return False, None, None
        
        # 与历史特征比较
        max_similarity = 0.0
        most_similar_time = None
        
        for hist_features, hist_time, hist_length in self.feature_history:
            similarity = self._cosine_similarity(current_features, hist_features)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_time = hist_time
        
        # 判断是否跳过
        if max_similarity >= self.similarity_threshold:
            # 相似度超过阈值，跳过
            time_since_similar = current_time - most_similar_time if most_similar_time else None
            
            # 判断是完全重复还是相似
            if max_similarity >= 0.98:
                reason = "duplicate"
                self.stats['skipped_duplicate'] += 1
            else:
                reason = "similar"
                self.stats['skipped_similar'] += 1
            
            self.stats['total_audio_time_skipped'] += audio_length
            
            details = {
                'type': reason,
                'similarity': max_similarity,
                'time_since_similar': time_since_similar,
                'audio_length': audio_length
            }
            
            return True, reason, details
        else:
            # 不跳过，添加到历史记录
            self.feature_history.append((current_features, current_time, audio_length))
            self.stats['passed'] += 1
            return False, None, None
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_checked': 0,
            'skipped_duplicate': 0,
            'skipped_similar': 0,
            'passed': 0,
            'total_audio_time_skipped': 0.0
        }
    
    def reset(self):
        """重置去重器（清空历史记录和统计）"""
        self.feature_history.clear()
        self.reset_stats()

