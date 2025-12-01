#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR组件模块
包含自定义的VAD迭代器和ASR处理器类
"""

import sys
import numpy as np


class DynamicVADIterator:
    """支持动态调整静音检测时间的VAD迭代器（基于FixedVADIterator）"""
    def __init__(self, model, initial_silence_ms=500, min_silence_ms=200, max_silence_ms=1000, threshold=0.5):
        """
        Args:
            model: silero VAD模型
            initial_silence_ms: 初始静音检测时间（毫秒）
            min_silence_ms: 最小静音检测时间（毫秒）- 密集说话时使用
            max_silence_ms: 最大静音检测时间（毫秒）- 稀疏说话时使用
            threshold: VAD语音检测阈值（0.0-1.0），越高越不敏感，默认0.5
        """
        from silero_vad_iterator import VADIterator  # type: ignore
        
        self.min_silence_ms = min_silence_ms
        self.max_silence_ms = max_silence_ms
        self.current_silence_ms = initial_silence_ms
        self.model = model
        self.threshold = threshold
        
        # 使用VADIterator作为基础，因为我们需要直接访问min_silence_samples来动态调整
        # 参考：whisper_streaming-main/silero_vad_iterator.py 第46行
        self.vad = VADIterator(model, threshold=threshold, min_silence_duration_ms=initial_silence_ms)
        self.buffer = np.array([], dtype=np.float32)
        
    def set_silence_duration(self, silence_ms):
        """
        动态设置静音检测时间
        
        参考 whisper_streaming-main/silero_vad_iterator.py:
        - min_silence_samples 在第46行计算：sampling_rate * min_silence_duration_ms / 1000
        - 在第92行使用来判断是否达到静音时间：if self.current_sample - self.temp_end < self.min_silence_samples
        """
        silence_ms = max(self.min_silence_ms, min(self.max_silence_ms, silence_ms))
        if abs(silence_ms - self.current_silence_ms) > 50:  # 只有变化超过50ms才更新
            self.current_silence_ms = silence_ms
            # 直接更新min_silence_samples，这是VADIterator中用于判断静音的关键参数
            # 参考：whisper_streaming-main/silero_vad_iterator.py 第46行和第92行
            self.vad.min_silence_samples = self.vad.sampling_rate * silence_ms / 1000
            return True
        return False
    
    def reset_states(self):
        """重置状态（参考 FixedVADIterator.reset_states）"""
        self.vad.reset_states()
        self.buffer = np.array([], dtype=np.float32)
    
    def __call__(self, x, return_seconds=False):
        """
        调用VAD迭代器（实现FixedVADIterator的逻辑）
        
        参考：whisper_streaming-main/silero_vad_iterator.py 第116-130行
        处理任意长度的音频块，内部按512样本块处理
        """
        self.buffer = np.append(self.buffer, x)
        ret = None
        while len(self.buffer) >= 512:
            r = self.vad(self.buffer[:512], return_seconds=return_seconds)
            self.buffer = self.buffer[512:]
            if ret is None:
                ret = r
            elif r is not None:
                if 'end' in r:
                    ret['end'] = r['end']  # the latter end
                if 'start' in r and 'end' in ret:
                    # Remove end, merging this segment with the previous one.
                    del ret['end']
        return ret if ret != {} else None


class DynamicVACOnlineASRProcessor:
    """支持动态调整静音检测时间的VACOnlineASRProcessor"""
    SAMPLING_RATE = 16000
    
    def __init__(self, online_chunk_size, asr, tokenizer=None, logfile=sys.stderr, 
                 buffer_trimming=("segment", 15), initial_silence_ms=500, 
                 min_silence_ms=200, max_silence_ms=1000, vad_threshold=0.5):
        from whisper_online import OnlineASRProcessor  # type: ignore
        import torch
        
        self.online_chunk_size = online_chunk_size
        self.online = OnlineASRProcessor(asr, tokenizer=tokenizer, logfile=logfile, 
                                        buffer_trimming=buffer_trimming)
        
        # 使用动态VAD迭代器
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self.vac = DynamicVADIterator(model, initial_silence_ms, min_silence_ms, max_silence_ms, vad_threshold)
        
        self.logfile = logfile
        self.init()
    
    def set_silence_duration(self, silence_ms):
        """动态设置静音检测时间"""
        return self.vac.set_silence_duration(silence_ms)
    
    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        self.status = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0
    
    def clear_buffer(self):
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([], dtype=np.float32)
    
    def insert_audio_chunk(self, audio):
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)
        
        if res is not None:
            frame = list(res.values())[0] - self.buffer_offset
            if 'start' in res and 'end' not in res:
                self.status = 'voice'
                send_audio = self.audio_buffer[frame:]
                self.online.init(offset=(frame + self.buffer_offset) / self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()
            elif 'end' in res and 'start' not in res:
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[:frame]
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
            else:
                beg = res["start"] - self.buffer_offset
                end = res["end"] - self.buffer_offset
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[beg:end]
                self.online.init(offset=(beg + self.buffer_offset) / self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
        else:
            if self.status == 'voice':
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                self.buffer_offset += max(0, len(self.audio_buffer) - self.SAMPLING_RATE)
                self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE:]
    
    def process_iter(self):
        """
        处理迭代（参考 VACOnlineASRProcessor.process_iter）
        
        参考：whisper_streaming-main/whisper_online.py 第712-721行
        """
        if self.is_currently_final:
            return self.finish()
        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter()
            return ret
        else:
            # 原始代码会打印日志，但我们已经在主循环中处理状态显示，所以不打印
            # print("no online update, only VAD", self.status, file=self.logfile)
            return (None, None, "")
    
    def finish(self):
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret


def create_custom_faster_whisper_asr(FasterWhisperASR):
    """
    创建自定义FasterWhisperASR类的工厂函数
    
    Args:
        FasterWhisperASR: 父类（从whisper_online导入）
    
    Returns:
        自定义的FasterWhisperASR类
    """
    class CustomFasterWhisperASR(FasterWhisperASR):
        """支持自定义 device、compute_type 和底层参数的 FasterWhisperASR"""
        
        def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, 
                     device="cuda", compute_type="float16", 
                     device_index=0, num_workers=1, cpu_threads=None,
                     logfile=sys.stderr, adaptive_params=None, transcribe_kwargs=None):
            """
            初始化自定义FasterWhisperASR
            
            Args:
                lan: 语言代码
                modelsize: 模型大小
                cache_dir: 模型缓存目录
                model_dir: 模型目录
                device: 设备类型（"cuda" 或 "cpu"）
                compute_type: 计算类型（"float16", "int8" 等）
                device_index: GPU设备索引
                num_workers: 工作线程数
                cpu_threads: CPU线程数（仅CPU模式）
                logfile: 日志文件
                adaptive_params: 自适应参数控制器
                transcribe_kwargs: transcribe方法的额外参数
            """
            self.device = device
            self.compute_type = compute_type
            self.device_index = device_index
            self.num_workers = num_workers
            self.cpu_threads = cpu_threads
            self.logfile = logfile
            # 使用传入的transcribe_kwargs，如果没有则使用空字典
            self.transcribe_kargs = transcribe_kwargs if transcribe_kwargs else {}
            self.adaptive_params = adaptive_params  # 自适应参数控制器
            if lan == "auto":
                self.original_language = None
            else:
                self.original_language = lan
            # 自定义加载模型
            self.model = self.load_model(modelsize, cache_dir, model_dir)
        
        def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
            from faster_whisper import WhisperModel
            import logging
            
            # 设置faster-whisper的日志级别为WARNING，过滤掉INFO级别的"Processing audio"和"Detected language"消息
            logging.getLogger("faster_whisper").setLevel(logging.WARNING)
            
            if model_dir is not None:
                model_size_or_path = model_dir
            elif modelsize is not None:
                model_size_or_path = modelsize
            else:
                raise ValueError("modelsize or model_dir parameter must be set")
            
            # 构建模型参数
            model_kwargs = {
                'device': self.device,
                'compute_type': self.compute_type,
                'download_root': cache_dir,
                'num_workers': self.num_workers,
            }
            
            # GPU 模式：添加设备索引
            if self.device == "cuda":
                model_kwargs['device_index'] = self.device_index
            
            # CPU 模式：添加线程数（如果指定）
            if self.device == "cpu" and self.cpu_threads is not None:
                # faster-whisper 使用 num_workers 控制 CPU 线程
                model_kwargs['num_workers'] = self.cpu_threads
            
            # 使用自定义参数创建模型
            model = WhisperModel(model_size_or_path, **model_kwargs)
            return model
        
        def transcribe(self, audio, init_prompt=""):
            """重写 transcribe 方法以支持自适应参数"""
            # 如果有自适应参数，获取当前参数
            if self.adaptive_params:
                adaptive_kwargs = self.adaptive_params.get_transcribe_kwargs()
                # 合并到 transcribe_kargs
                transcribe_kwargs = {**self.transcribe_kargs, **adaptive_kwargs}
            else:
                transcribe_kwargs = self.transcribe_kargs
            
            # 调用父类的 transcribe 逻辑，但使用自适应参数
            # 注意：faster-whisper 的 transcribe 方法签名
            segments, info = self.model.transcribe(
                audio, 
                language=self.original_language, 
                initial_prompt=init_prompt, 
                beam_size=transcribe_kwargs.get('beam_size', 5),
                temperature=transcribe_kwargs.get('temperature', 0.0),
                word_timestamps=True, 
                condition_on_previous_text=True, 
                **{k: v for k, v in transcribe_kwargs.items() if k not in ['beam_size', 'temperature']}
            )
            return list(segments)
        
        def ts_words(self, segments):
            """提取时间戳单词（兼容父类接口）"""
            o = []
            for s in segments:
                for word in s.words:
                    o.append((word.start, word.end, word.word))
            return o
        
        def segments_end_ts(self, segments):
            """提取段落结束时间戳（兼容父类接口）"""
            return [s.end for s in segments]
        
        def set_translate_task(self):
            """设置翻译任务（兼容父类接口）"""
            self.transcribe_kargs["task"] = "translate"
        
        def use_vad(self):
            """启用VAD（兼容父类接口）"""
            self.transcribe_kargs["vad_filter"] = True
    
    return CustomFasterWhisperASR
