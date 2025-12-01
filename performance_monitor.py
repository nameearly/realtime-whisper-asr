#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控模块
跟踪API调用成功率和延迟，以及本地识别速度，用于跳句机制
"""

import time
import logging
from datetime import datetime


class PerformanceMonitor:
    """性能监控器，跟踪API调用成功率和延迟，以及本地识别速度"""
    def __init__(self, window_size=10, max_delay_threshold=3.0, min_success_rate=0.5, enable_logging=True, log_file=None):
        """
        Args:
            window_size: 统计窗口大小（最近N次调用）
            max_delay_threshold: 最大延迟阈值（秒），超过此值认为延迟过高
            min_success_rate: 最小成功率阈值，低于此值认为成功率过低
            enable_logging: 是否启用日志记录
            log_file: 日志文件路径，如果为None则只输出到控制台
        """
        self.window_size = window_size
        self.max_delay_threshold = max_delay_threshold
        self.min_success_rate = min_success_rate
        self.enable_logging = enable_logging
        
        # 记录最近N次调用的结果
        self.call_history = []  # [(success, delay, timestamp), ...]
        self.pending_count = 0  # 当前待处理的请求数
        
        # 本地识别速度跟踪
        self.recognition_history = []  # [(audio_duration, processing_time, timestamp), ...]
        # audio_duration: 处理的音频时长（秒）
        # processing_time: 实际处理时间（秒）
        # timestamp: 识别完成的时间戳
        self.total_audio_duration = 0.0  # 累计处理的音频时长
        self.total_processing_time = 0.0  # 累计实际处理时间
        self.session_start_time = None  # 会话开始时间
        
        # 跳句统计
        self.skip_stats = {
            'recognition': {'count': 0, 'reasons': {}},
            'translation': {'count': 0, 'reasons': {}},
            'optimization': {'count': 0, 'reasons': {}}
        }
        
        # 设置日志
        if enable_logging:
            self.logger = logging.getLogger('PerformanceMonitor')
            self.logger.setLevel(logging.INFO)
            
            # 避免重复添加handler
            if not self.logger.handlers:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                
                # 根据配置决定是否输出到控制台
                console_log_enabled = True  # 默认启用
                try:
                    from config_manager import ConfigManager
                    config_manager = ConfigManager()
                    console_log_enabled = config_manager.get("logging.console_log_enabled", True)
                except:
                    pass
                
                # 控制台输出（如果启用）
                if console_log_enabled:
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(formatter)
                    self.logger.addHandler(console_handler)
                
                # 文件输出（如果指定）
                if log_file:
                    file_handler = logging.FileHandler(log_file, encoding='utf-8')
                    file_handler.setFormatter(formatter)
                    self.logger.addHandler(file_handler)
        else:
            self.logger = None
        
    def record_call(self, success, delay):
        """记录一次API调用结果"""
        self.call_history.append((success, delay, time.time()))
        # 只保留最近window_size次记录
        if len(self.call_history) > self.window_size:
            self.call_history.pop(0)
    
    def get_success_rate(self):
        """获取最近的成功率"""
        if not self.call_history:
            return 1.0  # 初始假设100%成功
        recent = self.call_history[-self.window_size:]
        success_count = sum(1 for success, _, _ in recent if success)
        return success_count / len(recent)
    
    def get_avg_delay(self):
        """获取最近的平均延迟"""
        if not self.call_history:
            return 0.0
        recent = self.call_history[-self.window_size:]
        delays = [delay for _, delay, _ in recent]
        return sum(delays) / len(delays) if delays else 0.0
    
    def record_recognition(self, audio_duration, processing_time=None, recognition_start_time=None):
        """
        记录一次本地识别结果
        
        Args:
            audio_duration: 处理的音频时长（秒），例如 beg_time 到 end_time 的差值
            processing_time: 实际处理时间（秒），如果为None则使用当前时间与recognition_start_time的差值
            recognition_start_time: 识别开始的时间戳（用于计算实际处理时间）
        """
        current_time = time.time()
        
        if self.session_start_time is None:
            self.session_start_time = current_time
        
        # 如果未提供processing_time，使用recognition_start_time计算实际处理时间
        if processing_time is None:
            if recognition_start_time is not None:
                processing_time = current_time - recognition_start_time
            elif self.recognition_history:
                # 如果没有提供recognition_start_time，使用上次记录的时间戳
                # 但这不准确，因为包含了等待时间
                last_timestamp = self.recognition_history[-1][2]
                processing_time = current_time - last_timestamp
            else:
                processing_time = current_time - self.session_start_time
        
        # 确保processing_time至少等于audio_duration（处理时间不能小于音频时长）
        processing_time = max(processing_time, audio_duration * 0.1)  # 至少是音频时长的10%
        
        # 记录识别结果
        self.recognition_history.append((audio_duration, processing_time, current_time))
        self.total_audio_duration += audio_duration
        self.total_processing_time += processing_time
        
        # 只保留最近window_size次记录
        if len(self.recognition_history) > self.window_size:
            old_audio, old_processing, _ = self.recognition_history.pop(0)
            self.total_audio_duration -= old_audio
            self.total_processing_time -= old_processing
    
    def get_recognition_speed_ratio(self, use_recent=True):
        """
        获取识别速度比率
        
        Args:
            use_recent: 如果True，只使用最近window_size次记录；如果False，使用所有记录
        
        Returns:
            比率值：如果返回0.5，表示实际时间30秒只能处理15秒的音频
            如果返回1.0，表示实时处理
            如果返回>1.0，表示处理速度超过实时
        """
        if not self.recognition_history:
            return 1.0  # 初始假设实时处理
        
        if use_recent:
            # 使用滑动窗口（最近window_size次）
            recent = self.recognition_history[-self.window_size:] if len(self.recognition_history) > self.window_size else self.recognition_history
            total_audio = sum(audio for audio, _, _ in recent)
            total_processing = sum(processing for _, processing, _ in recent)
        else:
            # 使用累计值
            total_audio = self.total_audio_duration
            total_processing = self.total_processing_time
        
        if total_processing == 0 or total_audio == 0:
            return 1.0
        
        return total_audio / total_processing
    
    def should_skip_recognition(self, queue_size=0):
        """
        判断是否应该跳过新的识别结果
        
        如果识别速度跟不上或队列积压过多，则跳过后续识别
        但已识别的结果必须翻译
        
        Args:
            queue_size: 当前翻译队列中的待处理任务数（如果提供）
        
        Returns:
            True: 应该跳过新的识别结果
            False: 可以处理新的识别结果
        """
        skip_reason = None
        skip_details = {}
        
        # 如果队列积压过多，直接跳过
        if queue_size >= 3:
            skip_reason = "queue_overflow"
            skip_details = {'queue_size': queue_size, 'threshold': 3}
            self._log_skip('recognition', skip_reason, skip_details)
            return True
        
        # 需要至少2次记录才能判断（降低门槛，更快响应）
        if len(self.recognition_history) < 2:
            return False
        
        # 使用最近记录计算速度比率（更准确反映当前状态）
        speed_ratio = self.get_recognition_speed_ratio(use_recent=True)
        skip_details['speed_ratio'] = speed_ratio
        skip_details['recognition_count'] = len(self.recognition_history)
        
        # 如果识别速度比率 < 0.4，说明识别速度严重跟不上（30秒只能处理不到12秒）
        # 此时跳过新的识别结果（降低阈值，更容易触发）
        if speed_ratio < 0.4:
            skip_reason = "low_speed_ratio"
            skip_details['threshold'] = 0.4
            self._log_skip('recognition', skip_reason, skip_details)
            return True
        
        # 如果识别速度比率 < 0.6，且最近几次识别都很慢，也跳过
        if speed_ratio < 0.6:
            # 检查最近3次识别的速度（如果至少有3次记录）
            if len(self.recognition_history) >= 3:
                recent = self.recognition_history[-3:]
                recent_audio = sum(audio for audio, _, _ in recent)
                recent_processing = sum(processing for _, processing, _ in recent)
                if recent_processing > 0:
                    recent_ratio = recent_audio / recent_processing
                    skip_details['recent_ratio'] = recent_ratio
                    if recent_ratio < 0.5:  # 最近3次平均速度 < 0.5
                        skip_reason = "low_recent_ratio"
                        skip_details['threshold'] = 0.5
                        self._log_skip('recognition', skip_reason, skip_details)
                        return True
        
        # 检查是否有明显的延迟累积
        if len(self.recognition_history) >= 3:
            # 计算最近3次的平均处理时间
            recent = self.recognition_history[-3:]
            avg_processing = sum(processing for _, processing, _ in recent) / len(recent)
            avg_audio = sum(audio for audio, _, _ in recent) / len(recent)
            skip_details['avg_processing'] = avg_processing
            skip_details['avg_audio'] = avg_audio
            skip_details['processing_ratio'] = avg_processing / avg_audio if avg_audio > 0 else 0
            
            # 如果平均处理时间明显大于音频时长（比如处理1秒音频需要3秒），说明严重延迟
            if avg_processing > avg_audio * 2.5:
                skip_reason = "high_processing_delay"
                skip_details['threshold'] = 2.5
                self._log_skip('recognition', skip_reason, skip_details)
                return True
        
        return False
    
    def should_skip_translation(self, queue_size=0):
        """
        判断是否应该跳过翻译步骤
        
        注意：已识别的结果必须翻译，所以这个方法主要用于优化阶段
        
        Args:
            queue_size: 当前翻译队列中的待处理任务数（如果提供）
        """
        skip_reason = None
        skip_details = {'queue_size': queue_size, 'pending_count': self.pending_count}
        
        # 已识别的结果必须翻译，所以这里只检查API调用性能和队列状态
        # 如果队列积压过多，跳过新的翻译请求
        if queue_size >= 5:
            skip_reason = "queue_overflow"
            skip_details['threshold'] = 5
            self._log_skip('translation', skip_reason, skip_details)
            return True
        
        # 如果pending_count过高，也跳过
        if self.pending_count >= 5:
            skip_reason = "high_pending_count"
            skip_details['threshold'] = 5
            self._log_skip('translation', skip_reason, skip_details)
            return True
        
        # 检查API调用性能
        if len(self.call_history) >= 3:  # 至少需要3次调用记录
            success_rate = self.get_success_rate()
            avg_delay = self.get_avg_delay()
            skip_details['success_rate'] = success_rate
            skip_details['avg_delay'] = avg_delay
            skip_details['min_success_rate'] = self.min_success_rate
            skip_details['max_delay_threshold'] = self.max_delay_threshold
            
            # 如果成功率过低或延迟过高，跳过
            if success_rate < self.min_success_rate * 0.6:
                skip_reason = "low_success_rate"
                skip_details['threshold'] = self.min_success_rate * 0.6
                self._log_skip('translation', skip_reason, skip_details)
                return True
            
            if avg_delay > self.max_delay_threshold * 2.0:
                skip_reason = "high_delay"
                skip_details['threshold'] = self.max_delay_threshold * 2.0
                self._log_skip('translation', skip_reason, skip_details)
                return True
        
        return False
    
    def should_skip_optimization(self, queue_size=0):
        """
        判断是否应该跳过优化步骤
        
        Args:
            queue_size: 当前翻译队列中的待处理任务数（如果提供）
        """
        skip_reason = None
        skip_details = {'queue_size': queue_size, 'pending_count': self.pending_count}
        
        # 如果队列积压，跳过优化
        if queue_size >= 3:
            skip_reason = "queue_overflow"
            skip_details['threshold'] = 3
            self._log_skip('optimization', skip_reason, skip_details)
            return True
        
        # 如果pending_count较高，跳过优化
        if self.pending_count >= 3:
            skip_reason = "high_pending_count"
            skip_details['threshold'] = 3
            self._log_skip('optimization', skip_reason, skip_details)
            return True
        
        # 检查API调用性能
        if len(self.call_history) >= 2:  # 至少需要2次调用记录
            success_rate = self.get_success_rate()
            avg_delay = self.get_avg_delay()
            skip_details['success_rate'] = success_rate
            skip_details['avg_delay'] = avg_delay
            skip_details['min_success_rate'] = self.min_success_rate
            skip_details['max_delay_threshold'] = self.max_delay_threshold
            
            # 优化可以更宽松一些，但如果性能很差也跳过
            if success_rate < self.min_success_rate * 0.7:
                skip_reason = "low_success_rate"
                skip_details['threshold'] = self.min_success_rate * 0.7
                self._log_skip('optimization', skip_reason, skip_details)
                return True
            
            if avg_delay > self.max_delay_threshold * 1.2:
                skip_reason = "high_delay"
                skip_details['threshold'] = self.max_delay_threshold * 1.2
                self._log_skip('optimization', skip_reason, skip_details)
                return True
        
        return False
    
    def increment_pending(self):
        """增加待处理计数"""
        self.pending_count += 1
    
    def decrement_pending(self):
        """减少待处理计数"""
        self.pending_count = max(0, self.pending_count - 1)
    
    def _log_skip(self, skip_type, reason, details):
        """
        记录跳句日志
        
        Args:
            skip_type: 跳句类型 ('recognition', 'translation', 'optimization')
            reason: 跳句原因
            details: 详细信息字典
        """
        if not self.enable_logging or not self.logger:
            return
        
        # 更新统计
        self.skip_stats[skip_type]['count'] += 1
        if reason not in self.skip_stats[skip_type]['reasons']:
            self.skip_stats[skip_type]['reasons'][reason] = 0
        self.skip_stats[skip_type]['reasons'][reason] += 1
        
        # 构建日志消息
        details_str = ', '.join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                                for k, v in details.items()])
        message = f"[跳句-{skip_type}] 原因: {reason}, 详情: {details_str}"
        
        self.logger.info(message)
    
    def get_skip_stats(self):
        """获取跳句统计信息"""
        return self.skip_stats.copy()
    
    def reset_skip_stats(self):
        """重置跳句统计"""
        self.skip_stats = {
            'recognition': {'count': 0, 'reasons': {}},
            'translation': {'count': 0, 'reasons': {}},
            'optimization': {'count': 0, 'reasons': {}}
        }
    
    def get_status(self):
        """获取当前状态信息"""
        return {
            "success_rate": self.get_success_rate(),
            "avg_delay": self.get_avg_delay(),
            "pending_count": self.pending_count,
            "total_calls": len(self.call_history),
            "recognition_speed_ratio": self.get_recognition_speed_ratio(use_recent=True),
            "recognition_speed_ratio_all": self.get_recognition_speed_ratio(use_recent=False),
            "total_audio_duration": self.total_audio_duration,
            "total_processing_time": self.total_processing_time,
            "recognition_count": len(self.recognition_history),
            "should_skip_recognition": self.should_skip_recognition(),
            "should_skip_translation": self.should_skip_translation(),
            "should_skip_optimization": self.should_skip_optimization(),
            "skip_stats": self.get_skip_stats()
        }

