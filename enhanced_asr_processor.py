#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的 ASR 处理器模块
基于 Whisper-Streaming 的设计，提供以下增强功能：
1. 可配置的 Local Agreement 策略（n=2,3,4...）
2. 动态缓冲区管理
3. 优化的 Init Prompt 提取
4. 改进的错误处理
5. 统一的接口设计
"""

import sys
import os
import numpy as np
import time

# 添加 whisper_streaming 路径
whisper_path = os.path.join(os.path.dirname(__file__), 'whisper_streaming-main', 'whisper_streaming-main')
if os.path.exists(whisper_path):
    sys.path.insert(0, whisper_path)
    from whisper_online import HypothesisBuffer, OnlineASRProcessor, VACOnlineASRProcessor
else:
    raise ImportError(f"找不到 whisper_streaming 目录: {whisper_path}")


class EnhancedHypothesisBuffer(HypothesisBuffer):
    """
    增强的 HypothesisBuffer，支持可配置的 Local Agreement 策略
    
    原版：Local Agreement-2（连续2次一致）
    增强：支持 Local Agreement-n（n可配置）
    """
    
    def __init__(self, agreement_n=2, logfile=sys.stderr):
        """
        Args:
            agreement_n: Local Agreement 的 n 值（连续 n 次一致才确认）
                        n=2: 平衡延迟和准确性（默认）
                        n=3: 更准确但延迟稍高
                        n=4: 最高准确性但延迟更高
            logfile: 日志文件
        """
        super().__init__(logfile=logfile)
        self.agreement_n = agreement_n
        self.history = []  # 存储最近 n 次的结果，用于 Local Agreement-n
        self.max_history = agreement_n
    
    def insert(self, new, offset):
        """
        插入新的转录结果（重写以支持历史记录）
        """
        # 调用父类方法
        super().insert(new, offset)
        
        # 保存当前 buffer + new 的组合状态到历史记录
        # 注意：我们需要保存 buffer + new 的组合，因为这是完整的当前状态
        current_state = self.buffer + self.new
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append(current_state.copy())
    
    def flush(self):
        """
        增强的 flush 方法，支持 Local Agreement-n
        
        原版逻辑（n=2）：
        - 比较 self.buffer（上次）和 self.new（本次）
        - 返回公共前缀
        
        增强逻辑（n>2）：
        - 比较最近 n 次的结果
        - 返回所有 n 次结果的最长公共前缀
        """
        if self.agreement_n == 2:
            # 使用原版逻辑（更快）
            return super().flush()
        
        # Local Agreement-n 逻辑
        # 需要至少 n 次历史记录
        if len(self.history) < self.agreement_n:
            # 历史记录不足，返回空（等待更多数据）
            # 但需要更新 buffer 和 new
            self.buffer = self.buffer + self.new
            self.new = []
            return []
        
        # 获取最近 n 次的结果
        recent_history = self.history[-self.agreement_n:]
        
        # 找到所有历史记录的最长公共前缀
        if not recent_history:
            self.buffer = self.buffer + self.new
            self.new = []
            return []
        
        # 获取所有历史记录中的词列表（只比较词文本，不比较时间戳）
        all_words = []
        for hist in recent_history:
            words = [word for _, _, word in hist]
            all_words.append(words)
        
        # 找到所有列表的最长公共前缀
        if not all_words:
            self.buffer = self.buffer + self.new
            self.new = []
            return []
        
        # 找到最短列表的长度
        min_len = min(len(words) for words in all_words)
        
        # 检查每个位置是否在所有列表中一致
        common_prefix_len = 0
        for i in range(min_len):
            # 检查位置 i 的词是否在所有列表中相同
            first_word = all_words[0][i]
            if all(words[i] == first_word for words in all_words):
                common_prefix_len += 1
            else:
                break
        
        # 提取公共前缀（使用最后一次历史记录的时间戳）
        commit = []
        if common_prefix_len > 0:
            last_hist = recent_history[-1]
            for i in range(common_prefix_len):
                if i < len(last_hist):
                    commit.append(last_hist[i])
                    self.last_commited_word = last_hist[i][2]
                    self.last_commited_time = last_hist[i][1]
        
        # 更新状态
        if commit:
            # 从所有历史记录中移除已确认的词
            for hist in recent_history:
                for _ in range(common_prefix_len):
                    if hist:
                        hist.pop(0)
            
            # 更新 commited_in_buffer
            self.commited_in_buffer.extend(commit)
        
        # 更新 buffer 和 new
        # buffer 应该是最后一次历史记录中未确认的部分
        if recent_history:
            self.buffer = recent_history[-1].copy()
        else:
            self.buffer = []
        self.new = []
        
        return commit


class DynamicBufferManager:
    """
    动态缓冲区管理器
    根据系统状态和性能指标自动调整缓冲区大小
    """
    
    def __init__(self, initial_trimming_sec=15, min_trimming_sec=5, max_trimming_sec=30):
        """
        Args:
            initial_trimming_sec: 初始缓冲区修剪阈值（秒）
            min_trimming_sec: 最小阈值（秒）
            max_trimming_sec: 最大阈值（秒）
        """
        self.current_trimming_sec = initial_trimming_sec
        self.min_trimming_sec = min_trimming_sec
        self.max_trimming_sec = max_trimming_sec
        
        # 性能指标
        self.recent_delays = []  # 最近的处理延迟
        self.recent_memory_usage = []  # 最近的内存使用率
        self.max_delay_samples = 10
        self.max_memory_samples = 10
    
    def record_delay(self, delay):
        """记录处理延迟"""
        self.recent_delays.append(delay)
        if len(self.recent_delays) > self.max_delay_samples:
            self.recent_delays.pop(0)
    
    def record_memory_usage(self, usage_percent):
        """记录内存使用率（0-100）"""
        self.recent_memory_usage.append(usage_percent)
        if len(self.recent_memory_usage) > self.max_memory_samples:
            self.recent_memory_usage.pop(0)
    
    def adjust_trimming_sec(self):
        """
        根据性能指标调整缓冲区修剪阈值
        
        策略：
        - 如果延迟高或内存使用率高 → 减小阈值（更频繁修剪）
        - 如果延迟低且内存充足 → 增大阈值（保留更多上下文）
        """
        if not self.recent_delays:
            return self.current_trimming_sec
        
        avg_delay = sum(self.recent_delays) / len(self.recent_delays)
        avg_memory = sum(self.recent_memory_usage) / len(self.recent_memory_usage) if self.recent_memory_usage else 50
        
        # 延迟阈值：3秒
        delay_threshold = 3.0
        # 内存阈值：80%
        memory_threshold = 80.0
        
        new_trimming_sec = self.current_trimming_sec
        
        # 如果延迟高或内存使用率高，减小阈值
        if avg_delay > delay_threshold or avg_memory > memory_threshold:
            new_trimming_sec = max(
                self.min_trimming_sec,
                self.current_trimming_sec - 2.0
            )
        # 如果延迟低且内存充足，增大阈值
        elif avg_delay < delay_threshold * 0.5 and avg_memory < memory_threshold * 0.7:
            new_trimming_sec = min(
                self.max_trimming_sec,
                self.current_trimming_sec + 2.0
            )
        
        if abs(new_trimming_sec - self.current_trimming_sec) > 0.5:
            self.current_trimming_sec = new_trimming_sec
            return True  # 表示需要更新
        
        return False
    
    def get_trimming_sec(self):
        """获取当前修剪阈值"""
        return self.current_trimming_sec


class EnhancedOnlineASRProcessor(OnlineASRProcessor):
    """
    增强的 OnlineASRProcessor
    提供以下增强功能：
    1. 可配置的 Local Agreement 策略
    2. 动态缓冲区管理
    3. 优化的 Init Prompt 提取
    4. 改进的错误处理
    """
    
    def __init__(self, asr, tokenizer=None, buffer_trimming=("segment", 15), 
                 logfile=sys.stderr, agreement_n=2, enable_dynamic_buffer=True):
        """
        Args:
            asr: ASR 后端对象
            tokenizer: 分词器（可选）
            buffer_trimming: 缓冲区修剪策略 (mode, seconds)
            logfile: 日志文件
            agreement_n: Local Agreement 的 n 值（默认2）
            enable_dynamic_buffer: 是否启用动态缓冲区管理
        """
        # 初始化父类（但使用增强的 HypothesisBuffer）
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming
        self.agreement_n = agreement_n
        self.enable_dynamic_buffer = enable_dynamic_buffer
        
        # 动态缓冲区管理器
        if enable_dynamic_buffer:
            self.buffer_manager = DynamicBufferManager(
                initial_trimming_sec=self.buffer_trimming_sec,
                min_trimming_sec=5,
                max_trimming_sec=30
            )
        else:
            self.buffer_manager = None
        
        # 初始化
        self.init()
    
    def init(self, offset=None):
        """初始化（使用增强的 HypothesisBuffer）"""
        self.audio_buffer = np.array([], dtype=np.float32)
        # 使用增强的 HypothesisBuffer
        self.transcript_buffer = EnhancedHypothesisBuffer(
            agreement_n=self.agreement_n,
            logfile=self.logfile
        )
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []
    
    def prompt(self):
        """
        优化的 Init Prompt 提取
        
        改进点：
        1. 提取更多上下文（从200字符增加到300字符）
        2. 智能截断（在词边界处截断，而非字符边界）
        3. 保留更多已确认文本作为上下文
        """
        prompt_limit = 300  # 增加到300字符
        context_limit = 500  # 上下文限制
        
        # 提取已确认文本
        prompt_parts = []
        prompt_length = 0
        
        # 从后往前提取（最新的文本）
        for item in reversed(self.commited):
            word = item[2]
            word_len = len(word) + 1  # +1 for separator
            
            if prompt_length + word_len > prompt_limit:
                break
            
            prompt_parts.append(word)
            prompt_length += word_len
        
        prompt = self.asr.sep.join(reversed(prompt_parts))
        
        # 提取仍在缓冲区内的已确认文本作为上下文
        non_prompt_parts = []
        non_prompt_length = 0
        
        # 从 commited 中提取不在 prompt 中的部分
        prompt_words = set(prompt_parts)
        for item in reversed(self.commited):
            word = item[2]
            if word not in prompt_words:
                word_len = len(word) + 1
                if non_prompt_length + word_len > context_limit:
                    break
                non_prompt_parts.append(word)
                non_prompt_length += word_len
        
        non_prompt = self.asr.sep.join(reversed(non_prompt_parts))
        
        return prompt, non_prompt
    
    def process_iter(self):
        """
        增强的 process_iter，包含动态缓冲区调整和错误处理
        """
        try:
            # 动态调整缓冲区大小
            if self.enable_dynamic_buffer and self.buffer_manager:
                if self.buffer_manager.adjust_trimming_sec():
                    new_sec = self.buffer_manager.get_trimming_sec()
                    self.buffer_trimming_sec = new_sec
                    # 可以在这里记录日志
                    # print(f"调整缓冲区修剪阈值: {new_sec:.1f}秒", file=self.logfile)
            
            # 调用父类方法
            result = super().process_iter()
            
            # 记录性能指标（用于动态调整）
            if self.enable_dynamic_buffer and self.buffer_manager and result[0] is not None:
                # 计算处理延迟（简单估算：基于音频缓冲区大小）
                audio_duration = len(self.audio_buffer) / self.SAMPLING_RATE
                # 这里可以添加更精确的延迟测量
                # 暂时使用音频时长作为参考
                self.buffer_manager.record_delay(audio_duration)
            
            return result
            
        except Exception as e:
            # 改进的错误处理
            print(f"处理迭代时发生错误: {e}", file=self.logfile)
            import traceback
            traceback.print_exc(file=self.logfile)
            
            # 尝试恢复：重置缓冲区
            try:
                self.init(offset=self.buffer_time_offset)
            except:
                pass
            
            return (None, None, "")
    
    def set_agreement_n(self, n):
        """
        动态设置 Local Agreement 的 n 值
        
        Args:
            n: 新的 n 值（2, 3, 4...）
        """
        if n < 2:
            n = 2
        self.agreement_n = n
        # 重新创建 HypothesisBuffer
        self.transcript_buffer = EnhancedHypothesisBuffer(
            agreement_n=n,
            logfile=self.logfile
        )
        self.transcript_buffer.last_commited_time = self.buffer_time_offset


class EnhancedVACOnlineASRProcessor(VACOnlineASRProcessor):
    """
    增强的 VACOnlineASRProcessor
    在 VACOnlineASRProcessor 基础上添加动态 VAD 调整功能
    """
    
    def __init__(self, online_chunk_size, asr, tokenizer=None, logfile=sys.stderr,
                 buffer_trimming=("segment", 15), agreement_n=2, 
                 enable_dynamic_buffer=True, initial_silence_ms=500,
                 min_silence_ms=200, max_silence_ms=1000, vad_threshold=0.5):
        """
        Args:
            online_chunk_size: 在线处理块大小
            asr: ASR 后端对象
            tokenizer: 分词器（可选）
            logfile: 日志文件
            buffer_trimming: 缓冲区修剪策略
            agreement_n: Local Agreement 的 n 值
            enable_dynamic_buffer: 是否启用动态缓冲区管理
            initial_silence_ms: 初始静音检测时间（毫秒）
            min_silence_ms: 最小静音检测时间（毫秒）
            max_silence_ms: 最大静音检测时间（毫秒）
        """
        # 使用增强的 OnlineASRProcessor
        self.online_chunk_size = online_chunk_size
        
        # 创建增强的 OnlineASRProcessor
        self.online = EnhancedOnlineASRProcessor(
            asr=asr,
            tokenizer=tokenizer,
            logfile=logfile,
            buffer_trimming=buffer_trimming,
            agreement_n=agreement_n,
            enable_dynamic_buffer=enable_dynamic_buffer
        )
        
        # 使用动态 VAD 迭代器
        import torch
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        
        # 导入动态 VAD 迭代器
        # 为了避免循环导入，我们在这里直接使用 VADIterator 并扩展
        from silero_vad_iterator import VADIterator
        import numpy as np
        
        # 创建动态 VAD 包装器
        class DynamicVADWrapper:
            def __init__(self, model, initial_silence_ms=500, min_silence_ms=200, max_silence_ms=1000, threshold=0.5):
                self.min_silence_ms = min_silence_ms
                self.max_silence_ms = max_silence_ms
                self.current_silence_ms = initial_silence_ms
                self.model = model
                self.threshold = threshold
                self.vad = VADIterator(model, threshold=threshold, min_silence_duration_ms=initial_silence_ms)
                self.buffer = np.array([], dtype=np.float32)
            
            def set_silence_duration(self, silence_ms):
                silence_ms = max(self.min_silence_ms, min(self.max_silence_ms, silence_ms))
                if abs(silence_ms - self.current_silence_ms) > 50:
                    self.current_silence_ms = silence_ms
                    self.vad.min_silence_samples = self.vad.sampling_rate * silence_ms / 1000
                    return True
                return False
            
            def reset_states(self):
                self.vad.reset_states()
                self.buffer = np.array([], dtype=np.float32)
            
            def __call__(self, x, return_seconds=False):
                self.buffer = np.append(self.buffer, x)
                ret = None
                while len(self.buffer) >= 512:
                    r = self.vad(self.buffer[:512], return_seconds=return_seconds)
                    self.buffer = self.buffer[512:]
                    if ret is None:
                        ret = r
                    elif r is not None:
                        if 'end' in r:
                            ret['end'] = r['end']
                        if 'start' in r and 'end' in ret:
                            del ret['end']
                return ret if ret != {} else None
        
        DynamicVADIterator = DynamicVADWrapper
        self.vac = DynamicVADIterator(
            model,
            initial_silence_ms=initial_silence_ms,
            min_silence_ms=min_silence_ms,
            max_silence_ms=max_silence_ms,
            threshold=vad_threshold
        )
        
        self.logfile = logfile
        self.init()
    
    def set_silence_duration(self, silence_ms):
        """动态设置静音检测时间"""
        return self.vac.set_silence_duration(silence_ms)
    
    def set_agreement_n(self, n):
        """动态设置 Local Agreement 的 n 值"""
        self.online.set_agreement_n(n)

