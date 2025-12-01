#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语速自适应处理模块（仅用于静音检测时间调整）
根据检测到的语速动态调整静音检测时间，优化不同语速下的识别效果
注意：此模块仅调整静音检测时间，不调整处理间隔
"""

import time
from typing import Dict, Optional
from collections import deque


class SpeechRateDetector:
    """语速检测器"""
    
    def __init__(self, window_size: int = 5, min_samples: int = 2):
        """
        Args:
            window_size: 用于计算语速的窗口大小（最近N次识别结果）
            min_samples: 最少需要的样本数才开始计算语速
        """
        self.window_size = window_size
        self.min_samples = min_samples
        
        # 存储历史记录：(时间戳, 文本长度, 音频时长, 语速)
        self.history = deque(maxlen=window_size)
        
        # 语速指标
        self.current_rate = 0.0  # 当前语速（字符/秒）
        self.avg_rate = 0.0  # 平均语速
        self.rate_category = "normal"  # 语速类别：slow, normal, fast
    
    def record_recognition(self, text: str, audio_duration: float, timestamp: Optional[float] = None):
        """
        记录一次识别结果
        
        Args:
            text: 识别的文本
            audio_duration: 音频时长（秒）
            timestamp: 时间戳（如果为None则使用当前时间）
        """
        if timestamp is None:
            timestamp = time.time()
        
        text_length = len(text.strip())
        
        # 计算语速（字符/秒）
        if audio_duration > 0:
            rate = text_length / audio_duration
        else:
            rate = 0.0
        
        self.history.append((timestamp, text_length, audio_duration, rate))
        
        # 更新语速指标
        self._update_rate()
    
    def _update_rate(self):
        """更新语速指标"""
        if len(self.history) < self.min_samples:
            return
        
        # 计算平均语速
        rates = [rate for _, _, _, rate in self.history]
        self.avg_rate = sum(rates) / len(rates)
        
        # 当前语速（最近一次）
        if self.history:
            self.current_rate = self.history[-1][3]
        
        # 分类语速
        if self.avg_rate < 5.0:  # 慢速：< 5 字符/秒
            self.rate_category = "slow"
        elif self.avg_rate > 15.0:  # 快速：> 15 字符/秒
            self.rate_category = "fast"
        else:  # 正常：5-15 字符/秒
            self.rate_category = "normal"
    
    def get_rate(self) -> float:
        """获取当前语速（字符/秒）"""
        return self.current_rate
    
    def get_avg_rate(self) -> float:
        """获取平均语速（字符/秒）"""
        return self.avg_rate
    
    def get_category(self) -> str:
        """获取语速类别"""
        return self.rate_category
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'current_rate': self.current_rate,
            'avg_rate': self.avg_rate,
            'category': self.rate_category,
            'samples': len(self.history)
        }
    
    def reset(self):
        """重置检测器"""
        self.history.clear()
        self.current_rate = 0.0
        self.avg_rate = 0.0
        self.rate_category = "normal"


class AdaptiveSilenceController:
    """自适应静音检测时间控制器（仅调整静音检测时间）"""
    
    def __init__(self, 
                 initial_silence_ms: int = 500,
                 min_silence_ms: int = 200,
                 max_silence_ms: int = 1000):
        """
        Args:
            initial_silence_ms: 初始静音检测时间（毫秒）
            min_silence_ms: 最小静音检测时间（毫秒）
            max_silence_ms: 最大静音检测时间（毫秒）
        """
        self.initial_silence_ms = initial_silence_ms
        self.min_silence_ms = min_silence_ms
        self.max_silence_ms = max_silence_ms
        
        self.current_silence_ms = initial_silence_ms
        
        # 语速检测器
        self.rate_detector = SpeechRateDetector()
        
        # 调整历史
        self.adjustment_history = []
    
    def update_from_recognition(self, text: str, audio_duration: float):
        """
        根据识别结果更新参数
        
        Args:
            text: 识别的文本
            audio_duration: 音频时长（秒）
        """
        # 记录识别结果
        self.rate_detector.record_recognition(text, audio_duration)
        
        # 根据语速调整静音检测时间
        self._adjust_silence()
    
    def _adjust_silence(self):
        """根据语速调整静音检测时间"""
        category = self.rate_detector.get_category()
        
        # 根据语速类别调整静音检测时间
        if category == "slow":
            # 慢速：延长静音检测时间，给更多时间让用户说完
            new_silence_ms = min(
                self.max_silence_ms,
                self.current_silence_ms + 100
            )
        elif category == "fast":
            # 快速：缩短静音检测时间，更快响应
            new_silence_ms = max(
                self.min_silence_ms,
                self.current_silence_ms - 100
            )
        else:  # normal
            # 正常：逐渐回归到初始值
            if self.current_silence_ms > self.initial_silence_ms:
                new_silence_ms = max(
                    self.initial_silence_ms,
                    self.current_silence_ms - 50
                )
            elif self.current_silence_ms < self.initial_silence_ms:
                new_silence_ms = min(
                    self.initial_silence_ms,
                    self.current_silence_ms + 50
                )
            else:
                new_silence_ms = self.current_silence_ms
        
        # 应用调整（只有变化足够大时才更新）
        if abs(new_silence_ms - self.current_silence_ms) >= 50:
            old_value = self.current_silence_ms
            self.current_silence_ms = new_silence_ms
            self.adjustment_history.append({
                'time': time.time(),
                'type': 'silence',
                'old': old_value,
                'new': new_silence_ms,
                'reason': f'语速: {category}'
            })
            return True
        return False
    
    def get_silence_ms(self) -> int:
        """获取当前静音检测时间（毫秒）"""
        return int(self.current_silence_ms)
    
    def get_rate_stats(self) -> Dict:
        """获取语速统计信息"""
        return self.rate_detector.get_stats()
    
    def get_adjustment_history(self, limit: int = 10):
        """获取调整历史"""
        return self.adjustment_history[-limit:]
    
    def reset(self):
        """重置控制器"""
        self.rate_detector.reset()
        self.current_silence_ms = self.initial_silence_ms
        self.adjustment_history.clear()


class SpeechRateAdaptiveProcessor:
    """语速自适应处理器（仅用于静音检测时间调整）"""
    
    def __init__(self, 
                 vad_processor=None,
                 initial_silence_ms: int = 500):
        """
        Args:
            vad_processor: VAD 处理器对象（如果有 set_silence_duration 方法）
            initial_silence_ms: 初始静音检测时间（毫秒）
        """
        self.vad_processor = vad_processor
        
        # 创建自适应控制器
        self.controller = AdaptiveSilenceController(
            initial_silence_ms=initial_silence_ms
        )
        
        # 应用初始参数
        if self.vad_processor and hasattr(self.vad_processor, 'set_silence_duration'):
            self.vad_processor.set_silence_duration(initial_silence_ms)
    
    def process_recognition_result(self, beg_time: float, end_time: float, text: str):
        """
        处理识别结果并更新自适应参数
        
        Args:
            beg_time: 开始时间（秒）
            end_time: 结束时间（秒）
            text: 识别的文本
        """
        if text and text.strip():
            # 计算音频时长（秒）
            audio_duration = end_time - beg_time
            
            # 更新控制器
            changed = self.controller.update_from_recognition(text.strip(), audio_duration)
            
            # 应用新的静音检测时间
            if changed and self.vad_processor and hasattr(self.vad_processor, 'set_silence_duration'):
                new_silence_ms = self.controller.get_silence_ms()
                self.vad_processor.set_silence_duration(new_silence_ms)
    
    def get_rate_stats(self) -> Dict:
        """获取语速统计信息"""
        return self.controller.get_rate_stats()
    
    def get_status(self) -> Dict:
        """获取状态信息"""
        rate_stats = self.controller.get_rate_stats()
        return {
            'speech_rate': rate_stats.get('avg_rate', 0.0),
            'rate_category': rate_stats.get('category', 'normal'),
            'silence_ms': self.controller.get_silence_ms()
        }
    
    def reset(self):
        """重置处理器"""
        self.controller.reset()
        if self.vad_processor and hasattr(self.vad_processor, 'set_silence_duration'):
            self.vad_processor.set_silence_duration(self.controller.initial_silence_ms)
