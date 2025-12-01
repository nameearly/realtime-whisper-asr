#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语速自适应音频处理模块
通过时间拉伸/压缩来归一化语速，提高不同语速下的识别准确率
"""

import numpy as np
from typing import Tuple, Optional
import librosa


class SpeechRateAudioProcessor:
    """语速自适应音频处理器"""
    
    def __init__(self, 
                 target_rate: float = 1.0,
                 min_stretch: float = 0.8,
                 max_stretch: float = 1.2,
                 enable_adaptive: bool = True):
        """
        Args:
            target_rate: 目标语速倍数（1.0 = 正常语速）
            min_stretch: 最小拉伸倍数（0.8 = 放慢到80%，即加快20%）
            max_stretch: 最大拉伸倍数（1.2 = 加快到120%，即放慢20%）
            enable_adaptive: 是否启用自适应（根据检测到的语速自动调整）
        """
        self.target_rate = target_rate
        self.min_stretch = min_stretch
        self.max_stretch = max_stretch
        self.enable_adaptive = enable_adaptive
        
        # 当前拉伸倍数
        self.current_stretch = 1.0
        
        # 语速检测（用于自适应）
        self.detected_rate = 0.0  # 字符/秒
        self.normal_rate = 10.0  # 正常语速（字符/秒），可根据实际情况调整
        
    def detect_speech_rate(self, text: str, audio_duration: float) -> float:
        """
        检测语速
        
        Args:
            text: 识别的文本
            audio_duration: 音频时长（秒）
            
        Returns:
            语速（字符/秒）
        """
        if audio_duration <= 0:
            return 0.0
        
        text_length = len(text.strip())
        rate = text_length / audio_duration
        self.detected_rate = rate
        return rate
    
    def calculate_stretch_factor(self, speech_rate: Optional[float] = None) -> float:
        """
        计算拉伸倍数
        
        Args:
            speech_rate: 语速（字符/秒），如果为None则使用检测到的语速
            
        Returns:
            拉伸倍数（>1.0表示加快，<1.0表示放慢）
        """
        if not self.enable_adaptive:
            return 1.0
        
        if speech_rate is None:
            speech_rate = self.detected_rate
        
        if speech_rate <= 0:
            return 1.0
        
        # 计算相对于正常语速的倍数
        rate_ratio = speech_rate / self.normal_rate
        
        # 如果语速过快（>1.2倍），放慢音频
        # 如果语速过慢（<0.8倍），加快音频
        if rate_ratio > 1.2:
            # 快速：放慢音频（拉伸倍数 < 1.0）
            # 例如：1.5倍语速 -> 0.8倍拉伸（放慢到80%）
            stretch = max(self.min_stretch, 1.0 / rate_ratio)
        elif rate_ratio < 0.8:
            # 慢速：加快音频（拉伸倍数 > 1.0）
            # 例如：0.6倍语速 -> 1.2倍拉伸（加快到120%）
            stretch = min(self.max_stretch, 1.0 / rate_ratio)
        else:
            # 正常语速：不拉伸
            stretch = 1.0
        
        self.current_stretch = stretch
        return stretch
    
    def process_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[np.ndarray, float]:
        """
        处理音频（时间拉伸/压缩）
        
        Args:
            audio: 音频数据（numpy数组）
            sample_rate: 采样率
            
        Returns:
            (处理后的音频, 实际拉伸倍数)
        """
        stretch = self.current_stretch
        
        # 如果不需要拉伸，直接返回
        if abs(stretch - 1.0) < 0.01:
            return audio, 1.0
        
        try:
            # 使用 librosa 进行时间拉伸（保持音调不变）
            # librosa.effects.time_stretch 使用相位声码器，保持音调
            processed_audio = librosa.effects.time_stretch(
                y=audio,
                rate=stretch
            )
            
            return processed_audio, stretch
        except Exception as e:
            # 如果处理失败，返回原音频
            print(f"⚠ 音频拉伸失败: {e}，使用原音频")
            return audio, 1.0
    
    def update_from_recognition(self, text: str, audio_duration: float):
        """
        根据识别结果更新拉伸参数
        
        Args:
            text: 识别的文本
            audio_duration: 音频时长（秒）
        """
        if not self.enable_adaptive:
            return
        
        # 检测语速
        rate = self.detect_speech_rate(text, audio_duration)
        
        # 计算新的拉伸倍数
        self.calculate_stretch_factor(rate)
    
    def set_normal_rate(self, rate: float):
        """
        设置正常语速（用于校准）
        
        Args:
            rate: 正常语速（字符/秒）
        """
        self.normal_rate = rate
    
    def get_status(self) -> dict:
        """获取状态信息"""
        return {
            'current_stretch': self.current_stretch,
            'detected_rate': self.detected_rate,
            'normal_rate': self.normal_rate,
            'enable_adaptive': self.enable_adaptive
        }
    
    def reset(self):
        """重置处理器"""
        self.current_stretch = 1.0
        self.detected_rate = 0.0


class AdaptiveWhisperParams:
    """根据语速自适应调整 Whisper 模型参数"""
    
    def __init__(self,
                 base_beam_size: int = 5,
                 base_temperature: float = 0.0,
                 enable_adaptive: bool = True):
        """
        Args:
            base_beam_size: 基础 beam_size（正常语速时使用）
            base_temperature: 基础 temperature（正常语速时使用）
            enable_adaptive: 是否启用自适应
        """
        self.base_beam_size = base_beam_size
        self.base_temperature = base_temperature
        self.enable_adaptive = enable_adaptive
        
        self.current_beam_size = base_beam_size
        self.current_temperature = base_temperature
        
        # 语速检测
        self.detected_rate = 0.0
        self.normal_rate = 10.0
    
    def update_from_recognition(self, text: str, audio_duration: float):
        """
        根据识别结果更新参数
        
        Args:
            text: 识别的文本
            audio_duration: 音频时长（秒）
        """
        if not self.enable_adaptive:
            return
        
        # 检测语速
        if audio_duration > 0:
            self.detected_rate = len(text.strip()) / audio_duration
        else:
            self.detected_rate = 0.0
        
        # 根据语速调整参数
        rate_ratio = self.detected_rate / self.normal_rate if self.normal_rate > 0 else 1.0
        
        if rate_ratio > 1.3:  # 快速语速
            # 快速语速：增大 beam_size，提高搜索范围
            self.current_beam_size = min(10, self.base_beam_size + 2)
            # 稍微提高 temperature，增加多样性
            self.current_temperature = min(0.2, self.base_temperature + 0.1)
        elif rate_ratio < 0.7:  # 慢速语速
            # 慢速语速：可以稍微减小 beam_size，加快处理
            self.current_beam_size = max(3, self.base_beam_size - 1)
            self.current_temperature = self.base_temperature
        else:  # 正常语速
            self.current_beam_size = self.base_beam_size
            self.current_temperature = self.base_temperature
    
    def get_transcribe_kwargs(self) -> dict:
        """
        获取 transcribe 方法的参数字典
        
        Returns:
            参数字典，可以直接传递给 transcribe 方法
        """
        return {
            'beam_size': self.current_beam_size,
            'temperature': self.current_temperature,
        }
    
    def get_status(self) -> dict:
        """获取状态信息"""
        return {
            'beam_size': self.current_beam_size,
            'temperature': self.current_temperature,
            'detected_rate': self.detected_rate,
            'normal_rate': self.normal_rate
        }
    
    def reset(self):
        """重置参数"""
        self.current_beam_size = self.base_beam_size
        self.current_temperature = self.base_temperature
        self.detected_rate = 0.0

