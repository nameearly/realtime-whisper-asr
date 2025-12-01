#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
支持从JSON配置文件加载配置，并提供默认值和验证
"""

import json
import os
from typing import Dict, Any, Optional


class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG = {
        "skip_detector": {
            "similarity_threshold": 0.85,
            "time_window": 3.0,
            "min_length": 2,
            "use_edit_distance": True
        },
        "device_protector": {
            "max_retries": 3,
            "retry_delay": 1.0,
            "check_interval": 0.5
        },
        "audio": {
            "sampling_rate": 16000,
            "channels": 1,
            "blocksize": 512
        },
        "performance_monitor": {
            "enable": True,
            "update_interval": 5.0,  # 更新间隔（秒）
            "show_stats": True,  # 是否显示统计信息
            "show_device_status": True  # 是否显示设备状态
        },
        "ui": {
            "show_colors": True,  # 是否使用彩色输出
            "show_progress": True,  # 是否显示进度
            "verbose_errors": True  # 是否显示详细错误信息
        },
        "logging": {
            "console_log_enabled": True,  # 是否输出日志到控制台
            "skip_log_enabled": True,
            "performance_log_enabled": True,
            "device_log_enabled": True
        },
        "asr_optimization": {
            "agreement_n": 3,  # Local Agreement-n的n值
            "vac_chunk_size": 0.08,  # VAC音频块大小（秒）
            "beam_size": 5,  # Whisper beam_size参数
            "temperature": 0.0,  # Whisper temperature参数
            "vad_threshold": 0.6  # VAD语音检测阈值（0.0-1.0），越高越不敏感
        },
        "vocal_separation": {
            "enable": False,  # 是否启用人声分离
            "method": "demucs",  # 分离方法：demucs, spleeter, filter, none
            "demucs_model": "htdemucs",  # Demucs模型名称
            "demucs_model_path": "",  # Demucs模型路径（可选），留空使用默认路径
            "spleeter_model": "2stems",  # Spleeter模型类型
            "filter_low_cut": 85.0,  # 频域滤波低截止频率
            "filter_high_cut": 3400.0  # 频域滤波高截止频率
        }
    }
    
    def __init__(self, config_file: str = "config.json"):
        """
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self) -> bool:
        """
        从文件加载配置
        
        Returns:
            是否成功加载
        """
        if not os.path.exists(self.config_file):
            # 配置文件不存在，创建默认配置
            self.save_config()
            return False
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # 合并配置（文件配置覆盖默认配置）
            self.config = self._merge_config(self.DEFAULT_CONFIG, file_config)
            
            # 验证配置
            self._validate_config()
            
            return True
        except json.JSONDecodeError as e:
            print(f"⚠ 配置文件格式错误: {e}")
            print("使用默认配置")
            return False
        except Exception as e:
            print(f"⚠ 加载配置文件失败: {e}")
            print("使用默认配置")
            return False
    
    def save_config(self) -> bool:
        """
        保存配置到文件
        
        Returns:
            是否成功保存
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"⚠ 保存配置文件失败: {e}")
            return False
    
    def _merge_config(self, default: Dict, override: Dict) -> Dict:
        """递归合并配置"""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def _validate_config(self):
        """验证配置值的有效性"""
        # 验证 skip_detector
        skip = self.config.get("skip_detector", {})
        if "similarity_threshold" in skip:
            threshold = skip["similarity_threshold"]
            if not 0.0 <= threshold <= 1.0:
                print(f"⚠ 相似度阈值无效: {threshold}，使用默认值 0.85")
                skip["similarity_threshold"] = 0.85
        
        if "time_window" in skip:
            time_window = skip["time_window"]
            if time_window < 0:
                print(f"⚠ 时间窗口无效: {time_window}，使用默认值 3.0")
                skip["time_window"] = 3.0
        
        if "min_length" in skip:
            min_length = skip["min_length"]
            if min_length < 1:
                print(f"⚠ 最小文本长度无效: {min_length}，使用默认值 2")
                skip["min_length"] = 2
        
        # 验证 device_protector
        device = self.config.get("device_protector", {})
        if "max_retries" in device:
            max_retries = device["max_retries"]
            if max_retries < 1:
                print(f"⚠ 最大重试次数无效: {max_retries}，使用默认值 3")
                device["max_retries"] = 3
        
        if "retry_delay" in device:
            retry_delay = device["retry_delay"]
            if retry_delay < 0:
                print(f"⚠ 重试延迟无效: {retry_delay}，使用默认值 1.0")
                device["retry_delay"] = 1.0
        
        # 验证 audio
        audio = self.config.get("audio", {})
        if "sampling_rate" in audio:
            sampling_rate = audio["sampling_rate"]
            if sampling_rate < 8000 or sampling_rate > 48000:
                print(f"⚠ 采样率无效: {sampling_rate}，使用默认值 16000")
                audio["sampling_rate"] = 16000
        
        if "blocksize" in audio:
            blocksize = audio["blocksize"]
            if blocksize < 64 or blocksize > 4096:
                print(f"⚠ 块大小无效: {blocksize}，使用默认值 512")
                audio["blocksize"] = 512
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号分隔的路径）
        
        Args:
            key_path: 配置路径，如 "skip_detector.similarity_threshold"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> bool:
        """
        设置配置值（支持点号分隔的路径）
        
        Args:
            key_path: 配置路径
            value: 配置值
            
        Returns:
            是否成功设置
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        return True
    
    def get_skip_detector_config(self) -> Dict:
        """获取跳句检测器配置"""
        return self.config.get("skip_detector", {})
    
    def get_device_protector_config(self) -> Dict:
        """获取设备保护器配置"""
        return self.config.get("device_protector", {})
    
    def get_audio_config(self) -> Dict:
        """获取音频配置"""
        return self.config.get("audio", {})
    
    def get_performance_monitor_config(self) -> Dict:
        """获取性能监控配置"""
        return self.config.get("performance_monitor", {})
    
    def get_ui_config(self) -> Dict:
        """获取UI配置"""
        return self.config.get("ui", {})
    
    def get_logging_config(self) -> Dict:
        """获取日志配置"""
        return self.config.get("logging", {})
    
    def get_language_specific_config(self, language: str, config_key: str) -> Dict:
        """
        获取语言特定配置（如果存在），否则返回默认配置
        
        Args:
            language: 语言代码（如 'zh', 'en'），'auto' 使用默认配置
            config_key: 配置键（如 'skip_detector', 'speech_rate_adaptive', 'asr_optimization'）
            
        Returns:
            合并后的配置字典（语言特定配置优先，然后默认配置）
        """
        # 获取默认配置
        default_config = self.config.get(config_key, {})
        
        # 如果是 auto 或未指定语言，返回默认配置
        if language == "auto" or not language:
            return default_config
        
        # 获取语言特定配置
        language_specific = self.config.get("language_specific", {})
        lang_config = language_specific.get(language, {})
        
        # 如果该语言有特定配置，合并（语言特定配置优先）
        if lang_config and config_key in lang_config:
            # 合并配置：语言特定配置覆盖默认配置
            merged = self._merge_config(default_config, lang_config[config_key])
            return merged
        
        # 没有语言特定配置，返回默认配置
        return default_config

