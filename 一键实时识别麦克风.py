#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键实时识别麦克风
直接运行此脚本即可开始实时语音识别

配置说明：
- 所有配置通过 config.json 文件管理，程序启动时自动加载
- 首次运行会自动创建默认配置文件
- 支持语言特定配置（中文、英文等），自动根据识别语言选择对应配置
- 主要配置项：
  * 识别语言：程序启动时选择（支持运行时切换）
  * 模型大小：程序启动时选择
  * 跳句检测：config.json 中的 skip_detector 配置
  * 语速自适应：config.json 中的 speech_rate_adaptive 配置
  * ASR优化：config.json 中的 asr_optimization 配置
  * 人声分离：config.json 中的 vocal_separation 配置
  * 音频去重：config.json 中的 audio_deduplication 配置

详细配置说明请参考 config.json 文件中的注释。
"""

import sys
import os
import time
import logging
from datetime import datetime
import queue
import threading

# 导入新模块
try:
    from audio_device_protector import AudioDeviceProtector
    DEVICE_PROTECTOR_AVAILABLE = True
except ImportError:
    DEVICE_PROTECTOR_AVAILABLE = False

try:
    from improved_skip_detector import ImprovedSkipDetector
    SKIP_DETECTOR_AVAILABLE = True
except ImportError:
    SKIP_DETECTOR_AVAILABLE = False

try:
    from audio_deduplicator import AudioDeduplicator
    AUDIO_DEDUPLICATOR_AVAILABLE = True
except ImportError:
    AUDIO_DEDUPLICATOR_AVAILABLE = False

try:
    from config_manager import ConfigManager
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False

try:
    from performance_display import PerformanceDisplay
    PERFORMANCE_DISPLAY_AVAILABLE = True
except ImportError:
    PERFORMANCE_DISPLAY_AVAILABLE = False

try:
    from vocal_separation import create_separator
    VOCAL_SEPARATION_AVAILABLE = True
except ImportError:
    VOCAL_SEPARATION_AVAILABLE = False

# 导入ASR组件
try:
    from asr_components import DynamicVADIterator, DynamicVACOnlineASRProcessor, create_custom_faster_whisper_asr
    ASR_COMPONENTS_AVAILABLE = True
except ImportError:
    ASR_COMPONENTS_AVAILABLE = False
    print("⚠ ASR组件模块不可用，将使用内置类")

# 添加 whisper_streaming 路径
whisper_path = os.path.join(os.path.dirname(__file__), 'whisper_streaming-main', 'whisper_streaming-main')
if os.path.exists(whisper_path):
    sys.path.insert(0, whisper_path)
else:
    print(f"错误: 找不到 whisper_streaming 目录: {whisper_path}")
    sys.exit(1)

def check_dependencies():
    """检查依赖"""
    missing = []
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
    
    try:
        import sounddevice as sd
    except ImportError:
        missing.append("sounddevice")
    
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        missing.append("faster-whisper")
    
    # requests 用于调用翻译 API（可选）
    try:
        import requests
    except ImportError:
        pass  # requests 是可选的，只在启用翻译时需要
    
    if missing:
        print("缺少以下依赖，请先安装：")
        print(f"pip install {' '.join(missing)}")
        return False
    return True

def detect_hardware():
    """检测硬件配置"""
    import psutil
    import os
    
    hardware_info = {
        'cpu_cores': psutil.cpu_count(logical=False),  # 物理核心数
        'cpu_threads': psutil.cpu_count(logical=True),  # 逻辑核心数
        'ram_gb': round(psutil.virtual_memory().total / (1024**3), 1),  # 内存GB
        'gpu_available': False,
        'gpu_memory_gb': 0,
        'gpu_memory_free_gb': 0,
        'gpu_name': None,
        'gpu_count': 0,
        'cuda_version': None,
        'gpu_devices': []
    }
    
    # 检测 NVIDIA GPU
    try:
        import subprocess
        # 获取 GPU 信息
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,driver_version', 
                                '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            hardware_info['gpu_count'] = len(lines)
            if lines:
                # 使用第一个 GPU
                gpu_info = lines[0].split(',')
                hardware_info['gpu_available'] = True
                hardware_info['gpu_name'] = gpu_info[1].strip()
                hardware_info['gpu_memory_gb'] = round(int(gpu_info[2].strip()) / 1024, 1)
                hardware_info['gpu_memory_free_gb'] = round(int(gpu_info[3].strip()) / 1024, 1)
                
                # 收集所有 GPU 信息
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 4:
                        hardware_info['gpu_devices'].append({
                            'index': int(parts[0].strip()),
                            'name': parts[1].strip(),
                            'memory_total_gb': round(int(parts[2].strip()) / 1024, 1),
                            'memory_free_gb': round(int(parts[3].strip()) / 1024, 1)
                        })
        
        # 检测 CUDA 版本
        try:
            cuda_result = subprocess.run(['nvcc', '--version'], 
                                       capture_output=True, text=True, timeout=3)
            if cuda_result.returncode == 0:
                for line in cuda_result.stdout.split('\n'):
                    if 'release' in line.lower():
                        import re
                        match = re.search(r'release\s+(\d+\.\d+)', line, re.IGNORECASE)
                        if match:
                            hardware_info['cuda_version'] = match.group(1)
        except:
            pass
    except:
        pass  # nvidia-smi 不可用或没有GPU
    
    return hardware_info

def optimize_low_level_params(hardware_info, use_gpu, model_size):
    """根据 GPU 情况优化底层参数"""
    params = {
        'num_workers': 1,  # CPU 线程数（实时场景通常用 1）
        'device_index': 0,  # GPU 设备索引
        'cpu_threads': None,  # CPU 线程数（用于 CPU 模式）
        'enable_memory_efficient': False,  # 内存优化
        'optimization_level': None,  # 优化级别
    }
    
    if use_gpu and hardware_info['gpu_available']:
        gpu_free = hardware_info['gpu_memory_free_gb']
        
        # 选择 GPU 设备（如果有多个，选择显存最多的）
        if hardware_info['gpu_count'] > 1:
            best_gpu = max(hardware_info['gpu_devices'], 
                          key=lambda x: x['memory_free_gb'])
            params['device_index'] = best_gpu['index']
            params['reason'] = f"检测到 {hardware_info['gpu_count']} 个 GPU，选择 GPU {best_gpu['index']} ({best_gpu['name']})"
        else:
            params['device_index'] = 0
        
        # 根据显存情况调整优化
        if gpu_free < 2:
            params['enable_memory_efficient'] = True
            params['optimization_level'] = 'aggressive'
            params['reason'] = (params.get('reason', '') + 
                              f"\n  - 显存紧张 ({gpu_free:.1f}GB 可用)，启用内存优化")
        elif gpu_free < 4:
            params['enable_memory_efficient'] = True
            params['optimization_level'] = 'moderate'
            params['reason'] = (params.get('reason', '') + 
                              f"\n  - 显存中等 ({gpu_free:.1f}GB 可用)，启用适度内存优化")
        else:
            params['reason'] = (params.get('reason', '') + 
                              f"\n  - 显存充足 ({gpu_free:.1f}GB 可用)，使用标准配置")
        
        # 根据模型大小调整
        if model_size in ['large-v2', 'large-v3', 'large']:
            params['enable_memory_efficient'] = True
            params['reason'] = (params.get('reason', '') + 
                              f"\n  - 大模型 ({model_size})，启用内存优化")
    else:
        # CPU 模式优化
        cpu_threads = hardware_info['cpu_threads']
        # 实时场景下，使用较少的线程避免延迟
        # 但可以根据核心数适当调整
        if cpu_threads >= 16:
            params['cpu_threads'] = min(8, cpu_threads // 2)  # 最多 8 线程
            params['reason'] = f"CPU {cpu_threads} 线程，使用 {params['cpu_threads']} 个线程"
        elif cpu_threads >= 8:
            params['cpu_threads'] = min(4, cpu_threads // 2)
            params['reason'] = f"CPU {cpu_threads} 线程，使用 {params['cpu_threads']} 个线程"
        else:
            params['cpu_threads'] = cpu_threads
            params['reason'] = f"CPU {cpu_threads} 线程，使用全部线程"
    
    return params

def recommend_config(hardware_info, use_gpu):
    """
    根据硬件配置推荐运行参数（基于真实性能数据）
    
    参考信息：
    - tiny: ~39M参数, 75MB, GPU约0.5GB, CPU实时性较好
    - base: ~74M参数, 142MB, GPU约1GB, CPU实时性一般
    - small: ~244M参数, 466MB, GPU约2GB, CPU实时性较差
    - medium: ~769M参数, 1.4GB, GPU约5GB, CPU不适合实时
    - large-v2/v3: ~1550M参数, 3GB, GPU约10GB, CPU不适合实时
    """
    recommendations = {
        'device': 'cuda' if use_gpu and hardware_info['gpu_available'] else 'cpu',
        'compute_type': None,
        'model_size': None,
        'reason': []
    }
    
    if use_gpu and hardware_info['gpu_available']:
        # GPU 模式 - 基于真实显存需求
        gpu_memory = hardware_info['gpu_memory_gb']
        gpu_free = hardware_info['gpu_memory_free_gb']
        
        # large-v2/v3 需要约 10GB 显存（float16）
        if gpu_memory >= 10 and gpu_free >= 8:
            recommendations['model_size'] = 'large-v2'
            recommendations['compute_type'] = 'float16'
            recommendations['reason'].append(f"GPU 显存 {gpu_memory}GB (可用 {gpu_free:.1f}GB) 充足，可运行 large-v2")
            recommendations['reason'].append("注意：large 模型在实时场景下延迟较高（3-5秒），建议 medium")
        # medium 需要约 5GB 显存
        elif gpu_memory >= 6 and gpu_free >= 4:
            recommendations['model_size'] = 'medium'
            recommendations['compute_type'] = 'float16'
            recommendations['reason'].append(f"GPU 显存 {gpu_memory}GB (可用 {gpu_free:.1f}GB)，推荐 medium 模型")
            recommendations['reason'].append("medium 模型：准确度高，实时延迟约 1-2 秒（推荐）")
        # small 需要约 2GB 显存
        elif gpu_memory >= 4 and gpu_free >= 2.5:
            recommendations['model_size'] = 'small'
            recommendations['compute_type'] = 'float16'
            recommendations['reason'].append(f"GPU 显存 {gpu_memory}GB (可用 {gpu_free:.1f}GB)，推荐 small 模型")
            recommendations['reason'].append("small 模型：平衡准确度和速度，实时延迟约 0.5-1 秒")
        # base 需要约 1GB 显存
        elif gpu_memory >= 2 and gpu_free >= 1.5:
            recommendations['model_size'] = 'base'
            recommendations['compute_type'] = 'float16'
            recommendations['reason'].append(f"GPU 显存 {gpu_memory}GB (可用 {gpu_free:.1f}GB)，推荐 base 模型")
            recommendations['reason'].append("base 模型：速度快，实时延迟约 0.3-0.5 秒，准确度中等")
        else:
            # 显存不足，使用 int8 量化
            recommendations['model_size'] = 'base'
            recommendations['compute_type'] = 'int8_float16'
            recommendations['reason'].append(f"GPU 显存 {gpu_memory}GB (可用 {gpu_free:.1f}GB) 较小，使用 base 模型（int8 量化）")
    else:
        # CPU 模式 - 基于真实性能数据
        cpu_cores = hardware_info['cpu_cores']
        cpu_threads = hardware_info['cpu_threads']
        ram_gb = hardware_info['ram_gb']
        
        # CPU 实时场景：只有 tiny 和 base 适合，small 以上延迟太高
        if cpu_threads >= 16 and ram_gb >= 16:
            # 高性能 CPU：可以尝试 small，但推荐 base
            recommendations['model_size'] = 'base'
            recommendations['compute_type'] = 'int8'
            recommendations['reason'].append(f"CPU {cpu_cores}核/{cpu_threads}线程 + {ram_gb}GB 内存")
            recommendations['reason'].append("CPU 模式：推荐 base 模型（int8），实时延迟约 1-2 秒")
            recommendations['reason'].append("注意：small 以上模型在 CPU 上延迟过高（>3秒），不适合实时")
        elif cpu_threads >= 8 and ram_gb >= 8:
            recommendations['model_size'] = 'base'
            recommendations['compute_type'] = 'int8'
            recommendations['reason'].append(f"CPU {cpu_cores}核/{cpu_threads}线程 + {ram_gb}GB 内存，推荐 base 模型")
        else:
            recommendations['model_size'] = 'tiny'
            recommendations['compute_type'] = 'int8'
            recommendations['reason'].append(f"CPU {cpu_cores}核/{cpu_threads}线程 + {ram_gb}GB 内存，推荐 tiny 模型")
            recommendations['reason'].append("tiny 模型：CPU 上速度最快，实时延迟约 0.5-1 秒")
    
    return recommendations

def recommend_demucs_config(hardware_info, use_gpu, whisper_model_size=None):
    """
    根据GPU显存推荐Demucs模型配置，考虑Whisper模型的显存占用
    
    显存需求参考（基于实际测试）：
    - Whisper模型显存占用（float16）：
      * tiny: ~0.5GB
      * base: ~1GB
      * small: ~2GB
      * medium: ~5GB
      * large-v2/v3: ~10GB
    
    - Demucs模型显存占用（推理时）：
      * htdemucs: ~1.5-2GB（轻量级，推荐）
      * htdemucs_ft: ~3-4GB（更高质量但更慢）
      * htdemucs_6s: ~2-2.5GB（6种音源分离）
      * hdemucs_mmi: ~2.5-3GB（混合模型）
      * mdx: ~2-3GB（高质量，较慢）
      * mdx_extra: ~3-4GB（最高质量，最慢）
    
    注意：实际显存占用还取决于音频长度和批处理大小
    """
    recommendations = {
        'demucs_model': None,
        'enable': False,
        'reason': [],
        'warnings': []
    }
    
    if not use_gpu or not hardware_info['gpu_available']:
        # CPU模式：不推荐使用Demucs（太慢）
        recommendations['enable'] = False
        recommendations['reason'].append("CPU模式：不推荐使用Demucs（处理速度太慢，不适合实时）")
        recommendations['reason'].append("建议：使用filter方法（频域滤波）或关闭人声分离")
        return recommendations
    
    gpu_memory = hardware_info['gpu_memory_gb']
    gpu_free = hardware_info['gpu_memory_free_gb']
    
    # 估算Whisper模型的显存占用
    whisper_vram = 0
    if whisper_model_size:
        whisper_vram_map = {
            'tiny': 0.5,
            'base': 1.0,
            'small': 2.0,
            'medium': 5.0,
            'large-v1': 10.0,
            'large-v2': 10.0,
            'large-v3': 10.0,
            'large': 10.0
        }
        whisper_vram = whisper_vram_map.get(whisper_model_size.lower(), 2.0)
    
    # 计算可用于Demucs的显存（预留1GB给系统和其他开销）
    available_for_demucs = gpu_free - whisper_vram - 1.0
    
    recommendations['reason'].append(f"GPU总显存: {gpu_memory}GB")
    recommendations['reason'].append(f"当前可用显存: {gpu_free:.1f}GB")
    if whisper_model_size:
        recommendations['reason'].append(f"Whisper模型 ({whisper_model_size}) 预计占用: ~{whisper_vram}GB")
        recommendations['reason'].append(f"可用于Demucs的显存: ~{available_for_demucs:.1f}GB")
    
    # 根据可用显存推荐Demucs模型
    if available_for_demucs >= 3.5:
        # 显存充足：可以使用高质量模型
        recommendations['demucs_model'] = 'htdemucs_ft'
        recommendations['enable'] = True
        recommendations['reason'].append("✓ 推荐: htdemucs_ft（更高质量，显存充足）")
        recommendations['reason'].append("  显存占用: ~3-4GB，分离质量最高")
    elif available_for_demucs >= 2.5:
        # 显存中等：使用标准模型
        recommendations['demucs_model'] = 'htdemucs'
        recommendations['enable'] = True
        recommendations['reason'].append("✓ 推荐: htdemucs（轻量级，平衡质量和速度）")
        recommendations['reason'].append("  显存占用: ~1.5-2GB，适合实时处理")
    elif available_for_demucs >= 1.5:
        # 显存紧张：使用轻量级模型
        recommendations['demucs_model'] = 'htdemucs'
        recommendations['enable'] = True
        recommendations['warnings'].append("⚠ 显存紧张，建议使用 htdemucs（最轻量级）")
        recommendations['warnings'].append("  如果出现OOM错误，考虑：")
        recommendations['warnings'].append("  1. 降低Whisper模型大小")
        recommendations['warnings'].append("  2. 使用filter方法替代Demucs")
        recommendations['warnings'].append("  3. 关闭人声分离")
    else:
        # 显存不足：不推荐使用Demucs
        recommendations['enable'] = False
        recommendations['warnings'].append("❌ 显存不足，无法同时运行Whisper和Demucs")
        recommendations['warnings'].append(f"  需要至少 {whisper_vram + 2.5:.1f}GB 显存（Whisper + Demucs + 系统开销）")
        recommendations['warnings'].append("  建议：")
        recommendations['warnings'].append("  1. 使用filter方法（频域滤波，无需额外显存）")
        recommendations['warnings'].append("  2. 降低Whisper模型大小")
        recommendations['warnings'].append("  3. 关闭人声分离")
    
    return recommendations

# 注意：旧的API调用相关代码已移除，翻译功能已迁移到 translation_manager.py 模块

# ========== 异步输出机制（避免输出阻塞主循环） ==========
class AsyncOutput:
    """异步输出类，使用队列和后台线程避免输出操作阻塞主循环"""
    
    def __init__(self):
        self.output_queue = queue.Queue(maxsize=100)  # 限制队列大小，避免内存无限增长
        self.running = False
        self.thread = None
    
    def start(self):
        """启动后台输出线程"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._output_worker, daemon=True)
        self.thread.start()
    
    def stop(self):
        """停止后台输出线程"""
        self.running = False
        # 添加停止标记
        try:
            self.output_queue.put_nowait(("__STOP__", None))
        except queue.Full:
            pass
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _output_worker(self):
        """后台输出工作线程"""
        while self.running:
            try:
                # 使用超时避免无限等待
                item = self.output_queue.get(timeout=0.1)
                if item[0] == "__STOP__":
                    break
                output_type, content = item
                
                if output_type == "print":
                    # 普通print输出
                    print(content, flush=True)
                elif output_type == "print_no_newline":
                    # print输出（不换行）
                    print(content, end='', flush=True)
                elif output_type == "stdout_flush":
                    # stdout flush
                    sys.stdout.flush()
                
                self.output_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # 输出失败不应该影响主程序，静默处理
                pass
    
    def print(self, content):
        """异步print（非阻塞）"""
        try:
            self.output_queue.put_nowait(("print", content))
        except queue.Full:
            # 队列满时，直接输出（避免丢失重要信息）
            print(content, flush=True)
    
    def print_no_newline(self, content):
        """异步print（不换行，非阻塞）"""
        try:
            self.output_queue.put_nowait(("print_no_newline", content))
        except queue.Full:
            print(content, end='', flush=True)
    
    def flush(self):
        """异步flush（非阻塞）"""
        try:
            self.output_queue.put_nowait(("stdout_flush", None))
        except queue.Full:
            sys.stdout.flush()

# 全局异步输出实例
_async_output = AsyncOutput()

def main():
    print("=" * 60)
    print("一键实时麦克风语音识别")
    print("=" * 60)
    print()
    
    # 加载配置
    config_manager = None
    if CONFIG_MANAGER_AVAILABLE:
        config_manager = ConfigManager()
        print("✓ 配置已加载")
    else:
        print("⚠ 配置管理模块不可用，使用默认配置")
    
    # 初始化性能显示
    perf_display = None
    if PERFORMANCE_DISPLAY_AVAILABLE:
        ui_config = config_manager.get_ui_config() if config_manager else {}
        perf_config = config_manager.get_performance_monitor_config() if config_manager else {}
        perf_display = PerformanceDisplay(
            enable_colors=ui_config.get('show_colors', True),
            update_interval=perf_config.get('update_interval', 5.0)
        )
    
    # 检查依赖
    if not check_dependencies():
        if perf_display:
            perf_display.display_error("依赖检查", "缺少必要的依赖包", "请运行: pip install -r requirements.txt")
        sys.exit(1)
    
    import numpy as np
    import sounddevice as sd
    from whisper_online import FasterWhisperASR, WhisperTimestampedASR, OnlineASRProcessor, VACOnlineASRProcessor  # type: ignore
    
    # 创建支持自定义 device 和 compute_type 的包装类
    if ASR_COMPONENTS_AVAILABLE:
        CustomFasterWhisperASR = create_custom_faster_whisper_asr(FasterWhisperASR)
    else:
        # 回退到内置实现（如果模块不可用）
        class CustomFasterWhisperASR(FasterWhisperASR):
            """支持自定义 device、compute_type 和底层参数的 FasterWhisperASR（回退实现）"""
            def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, 
                         device="cuda", compute_type="float16", 
                         device_index=0, num_workers=1, cpu_threads=None,
                         logfile=sys.stderr, adaptive_params=None, transcribe_kwargs=None):
                self.device = device
                self.compute_type = compute_type
                self.device_index = device_index
                self.num_workers = num_workers
                self.cpu_threads = cpu_threads
                self.logfile = logfile
                self.transcribe_kargs = transcribe_kwargs if transcribe_kwargs else {}
                self.adaptive_params = adaptive_params
                if lan == "auto":
                    self.original_language = None
                else:
                    self.original_language = lan
                self.model = self.load_model(modelsize, cache_dir, model_dir)
            
            def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
                from faster_whisper import WhisperModel
                import logging
                logging.getLogger("faster_whisper").setLevel(logging.WARNING)
                if model_dir is not None:
                    model_size_or_path = model_dir
                elif modelsize is not None:
                    model_size_or_path = modelsize
                else:
                    raise ValueError("modelsize or model_dir parameter must be set")
                model_kwargs = {
                    'device': self.device,
                    'compute_type': self.compute_type,
                    'download_root': cache_dir,
                    'num_workers': self.num_workers,
                }
                if self.device == "cuda":
                    model_kwargs['device_index'] = self.device_index
                if self.device == "cpu" and self.cpu_threads is not None:
                    model_kwargs['num_workers'] = self.cpu_threads
                model = WhisperModel(model_size_or_path, **model_kwargs)
                return model
            
            def transcribe(self, audio, init_prompt=""):
                if self.adaptive_params:
                    adaptive_kwargs = self.adaptive_params.get_transcribe_kwargs()
                    transcribe_kwargs = {**self.transcribe_kargs, **adaptive_kwargs}
                else:
                    transcribe_kwargs = self.transcribe_kargs
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
    
    # 检测硬件配置
    print("正在检测硬件配置...")
    try:
        import psutil
        hardware = detect_hardware()
        print(f"✓ 硬件检测完成")
        print(f"  - CPU: {hardware['cpu_cores']} 物理核心 / {hardware['cpu_threads']} 逻辑核心")
        print(f"  - 内存: {hardware['ram_gb']} GB")
        if hardware['gpu_available']:
            print(f"  - GPU: {hardware['gpu_name']} ({hardware['gpu_memory_gb']} GB)")
        else:
            print(f"  - GPU: 未检测到或不可用")
        print()
    except ImportError:
        print("⚠ 未安装 psutil，无法检测硬件配置")
        print("  可以安装: pip install psutil")
        print("  将使用默认配置")
        hardware = {'cpu_cores': 8, 'cpu_threads': 16, 'ram_gb': 16, 'gpu_available': False, 'gpu_memory_gb': 0}
        print()
    except Exception as e:
        print(f"⚠ 硬件检测失败: {e}")
        print("  将使用默认配置")
        hardware = {'cpu_cores': 8, 'cpu_threads': 16, 'ram_gb': 16, 'gpu_available': False, 'gpu_memory_gb': 0}
        print()
    
    # 询问用户选择运行模式
    print("=" * 60)
    print("选择运行模式")
    print("=" * 60)
    if hardware['gpu_available']:
        print("检测到可用 GPU，可以选择：")
        print("  [1] GPU 模式（更快，推荐）")
        print("  [2] CPU 模式（更稳定，不依赖 CUDA）")
    else:
        print("未检测到可用 GPU，将使用 CPU 模式")
    
    use_gpu = False
    if hardware['gpu_available']:
        while True:
            try:
                choice = input("\n请选择运行模式 (1/2，直接回车使用 GPU): ").strip()
                if choice == "" or choice == "1":
                    use_gpu = True
                    break
                elif choice == "2":
                    use_gpu = False
                    break
                else:
                    print("请输入 1 或 2")
            except KeyboardInterrupt:
                print("\n退出程序")
                sys.exit(0)
    else:
        use_gpu = False
        print("自动选择 CPU 模式")
    
    print()
    
    # 选择识别后端
    print("=" * 60)
    print("选择识别后端")
    print("=" * 60)
    print("  [1] faster-whisper（推荐，速度快，约4倍加速）")
    print("  [2] whisper（原始版本，安装简单，GPU支持更好）")
    print()
    
    backend_choice = None
    while backend_choice is None:
        try:
            choice = input("请选择后端 (直接回车使用 faster-whisper): ").strip()
            if choice == "" or choice == "1":
                backend_choice = "faster-whisper"
            elif choice == "2":
                backend_choice = "whisper"
            else:
                print("无效选择，请输入 1 或 2")
        except KeyboardInterrupt:
            print("\n使用默认后端: faster-whisper")
            backend_choice = "faster-whisper"
            break
    
    print(f"✓ 已选择后端: {backend_choice}")
    print()
    
    # 根据硬件配置推荐参数
    recommendations = recommend_config(hardware, use_gpu)
    
    print("=" * 60)
    print("推荐配置")
    print("=" * 60)
    print(f"运行设备: {recommendations['device'].upper()}")
    print(f"推荐模型: {recommendations['model_size']}")
    print(f"计算类型: {recommendations['compute_type']}")
    for reason in recommendations['reason']:
        print(f"  - {reason}")
    print()
    
    # 询问用户是否使用推荐配置
    while True:
        try:
            confirm = input("是否使用推荐配置？(y/n，直接回车使用推荐配置): ").strip().lower()
            if confirm == "" or confirm in ['y', 'yes', '是']:
                model_size = recommendations['model_size']
                device = recommendations['device']
                compute_type = recommendations['compute_type']
                print(f"✓ 已应用推荐配置: {model_size} 模型, {device.upper()} 设备, {compute_type} 计算类型")
                break
            elif confirm in ['n', 'no', '否']:
                # 让用户手动选择模型
                print("\n可用模型: tiny, base, small, medium, large-v2, large-v3")
                model_choice = input("请选择模型 (直接回车使用 medium): ").strip().lower()
                if model_choice == "":
                    model_size = "medium"
                elif model_choice in ['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']:
                    model_size = model_choice
                else:
                    print("无效选择，使用 medium")
                    model_size = "medium"
                
                if use_gpu:
                    device = "cuda"
                    compute_type = "float16"
                else:
                    device = "cpu"
                    compute_type = "int8"
                
                print(f"✓ 已应用自定义配置: {model_size} 模型, {device.upper()} 设备, {compute_type} 计算类型")
                
                # 如果选择了大模型，给出提示
                if model_size in ['large-v2', 'large-v3', 'large']:
                    print()
                    print("⚠ 注意：大模型 (large-v2/v3) 在实时场景下：")
                    print("  - 处理速度较慢（延迟较高）")
                    print("  - 需要更多显存和计算资源")
                    print("  - 建议使用 medium 模型以获得更好的实时体验")
                    print()
                
                break
            else:
                print("请输入 y 或 n")
        except KeyboardInterrupt:
            print("\n退出程序")
            sys.exit(0)
    
    # 优化底层参数
    print("正在优化底层参数...")
    low_level_params = optimize_low_level_params(hardware, use_gpu, model_size)
    if 'reason' in low_level_params:
        print("底层优化:")
        for line in low_level_params['reason'].split('\n'):
            if line.strip():
                print(f"  {line.strip()}")
    print()
    
    # 人声分离模型推荐和选择（如果启用了人声分离）
    if VOCAL_SEPARATION_AVAILABLE and config_manager:
        sep_config = config_manager.get("vocal_separation", {})
        if sep_config.get("enable", False):
            method = sep_config.get("method", "demucs").lower()
            
            # Demucs 模型推荐和选择
            if method == "demucs":
                print("=" * 60)
                print("Demucs 模型推荐")
                print("=" * 60)
                
                # 如果启用了GPU，进行显存推荐
                if use_gpu and hardware['gpu_available']:
                    demucs_recommendations = recommend_demucs_config(hardware, use_gpu, model_size)
                    for reason in demucs_recommendations['reason']:
                        print(f"  {reason}")
                    if demucs_recommendations['warnings']:
                        print()
                        for warning in demucs_recommendations['warnings']:
                            print(f"  {warning}")
                    print()
                    
                    # 如果推荐了模型，让用户选择
                    if demucs_recommendations['demucs_model']:
                        recommended_model = demucs_recommendations['demucs_model']
                        current_model = sep_config.get("demucs_model", "").strip()
                        
                        print(f"推荐模型: {recommended_model}")
                        if current_model:
                            print(f"当前配置: {current_model}")
                        
                        print("\n可用模型:")
                        print("  [1] htdemucs - 轻量级，~1.5-2GB显存（推荐实时）")
                        print("  [2] htdemucs_ft - 更高质量，~3-4GB显存")
                        print("  [3] htdemucs_6s - 6种音源分离，~2-2.5GB显存")
                        print("  [4] hdemucs_mmi - 混合模型，~2.5-3GB显存")
                        print("  [5] mdx - 高质量，~2-3GB显存")
                        print("  [6] mdx_extra - 最高质量，~3-4GB显存")
                        
                        model_map = {
                            '1': 'htdemucs',
                            '2': 'htdemucs_ft',
                            '3': 'htdemucs_6s',
                            '4': 'hdemucs_mmi',
                            '5': 'mdx',
                            '6': 'mdx_extra'
                        }
                        
                        try:
                            choice = input(f"\n请选择模型 (1-6，直接回车使用推荐 {recommended_model}): ").strip()
                            if choice == "":
                                selected_model = recommended_model
                            elif choice in model_map:
                                selected_model = model_map[choice]
                            else:
                                print(f"无效选择，使用推荐模型: {recommended_model}")
                                selected_model = recommended_model
                            
                            # 更新配置
                            sep_config["demucs_model"] = selected_model
                            config_manager.set("vocal_separation", sep_config)
                            config_manager.save()
                            print(f"✓ 已更新 config.json: demucs_model = {selected_model}")
                            print()
                        except KeyboardInterrupt:
                            print("\n使用当前配置")
                            print()
                    elif not demucs_recommendations['enable']:
                        print("⚠ 建议：根据当前显存情况，不建议启用 Demucs")
                        print("  可以考虑使用 filter 方法（频域滤波）或关闭人声分离")
                        print()
                else:
                    # CPU模式或没有GPU，提供基本选择
                    current_model = sep_config.get("demucs_model", "").strip()
                    if not current_model:
                        print("⚠ CPU模式下不推荐使用Demucs（速度太慢）")
                        print("  建议使用 filter 方法或关闭人声分离")
                        print()
                    else:
                        print(f"当前配置: {current_model}")
                        print("⚠ 注意：CPU模式下Demucs处理速度较慢，可能影响实时性")
                        print()
            
            # Spleeter 模型推荐和选择
            elif method == "spleeter":
                print("=" * 60)
                print("Spleeter 模型选择")
                print("=" * 60)
                print("可用模型:")
                print("  [1] 2stems - 人声+伴奏（推荐，最快）")
                print("  [2] 4stems - 人声、鼓、贝斯、其他")
                print("  [3] 5stems - 人声、鼓、贝斯、钢琴、其他")
                
                current_model = sep_config.get("spleeter_model", "2stems")
                print(f"\n当前配置: {current_model}")
                
                model_map = {
                    '1': '2stems',
                    '2': '4stems',
                    '3': '5stems'
                }
                
                try:
                    choice = input("请选择模型 (1-3，直接回车保持当前配置): ").strip()
                    if choice in model_map:
                        selected_model = model_map[choice]
                        sep_config["spleeter_model"] = selected_model
                        config_manager.set("vocal_separation", sep_config)
                        config_manager.save()
                        print(f"✓ 已更新 config.json: spleeter_model = {selected_model}")
                    elif choice == "":
                        print("保持当前配置")
                    else:
                        print("无效选择，保持当前配置")
                    print()
                except KeyboardInterrupt:
                    print("\n保持当前配置")
                    print()
    
    # 重新加载配置（如果用户更新了模型配置）
    if config_manager:
        config_manager.load_config()
    
    # 配置参数
    SAMPLING_RATE = 16000  # Whisper 需要的采样率
    CHUNK_DURATION = 1.0   # 基础处理间隔（VAC 模式下会自动调整）
    
    # VAC (Voice Activity Controller) 配置
    # VAC 使用语音活动检测，会在检测到语音结束时（500ms 静音）自动处理
    # 这样可以实现按句子/半句处理，而不是固定时间间隔
    use_vac = True  # 是否使用 VAC（推荐启用，实现按句子处理）
    
    # 从配置文件读取优化参数（如果可用）
    if CONFIG_MANAGER_AVAILABLE and config_manager:
        asr_opt = config_manager.get("asr_optimization", {})
        vac_chunk_size = asr_opt.get("vac_chunk_size", 0.08)  # 默认0.08秒（增加上下文）
        agreement_n = asr_opt.get("agreement_n", 3)  # 默认3（更准确）
        vad_initial_silence_ms = config_manager.get("speech_rate_adaptive.initial_silence_ms", 1000)
        vad_min_silence_ms = config_manager.get("speech_rate_adaptive.min_silence_ms", 500)
        vad_max_silence_ms = config_manager.get("speech_rate_adaptive.max_silence_ms", 1500)
        beam_size = asr_opt.get("beam_size", 5)
        temperature = asr_opt.get("temperature", 0.0)
        print(f"✓ 已加载ASR优化配置: agreement_n={agreement_n}, vac_chunk_size={vac_chunk_size}, VAD静音={vad_min_silence_ms}-{vad_max_silence_ms}ms")
    else:
        vac_chunk_size = 0.08  # 默认值：0.08秒（增加上下文）
        agreement_n = 3  # 默认值：3（更准确）
        vad_initial_silence_ms = 1000
        vad_min_silence_ms = 500
        vad_max_silence_ms = 1500
        vad_threshold = 0.6  # 默认0.6，减少背景音乐干扰
        beam_size = 5
        temperature = 0.0
    
    # 语言和模型配置
    # 输出中文：设置 input_language="zh" 或 "auto"，task="transcribe"
    # 输出英文：设置 input_language="en" 或 "auto"，task="transcribe"
    # 中文翻译成英文：设置 input_language="zh"，task="translate"
    # 注意：translate 模式只能翻译成英文，不能翻译成其他语言
    # 默认语言（如果用户不选择，将使用此值）
    default_language = "auto"  # 输入语音的语言: "zh"（中文）、"en"（英文）、"auto"（自动检测）等
    task = "transcribe"      # 任务类型: "transcribe"（转录，输出与输入相同的语言）或 "translate"（翻译成英文）
    # model_size 已在上面根据硬件配置设置
    
    # 让用户选择识别语言（在加载模型之前）
    print("=" * 60)
    print("选择识别语言")
    print("=" * 60)
    print("常用语言代码: zh(中文), en(英文), ja(日文), ko(韩文), es(西班牙语), fr(法语), de(德语), ru(俄语)")
    print("完整语言列表: af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh,yue")
    print()
    
    # 语言代码验证列表
    valid_languages = ['auto', 'zh', 'en', 'ja', 'ko', 'es', 'fr', 'de', 'ru', 'it', 'pt', 'ar', 'hi', 'th', 'vi', 'id', 'nl', 'pl', 'tr', 'cs', 'sv', 'no', 'da', 'fi', 'el', 'he', 'uk', 'ro', 'hu', 'bg', 'hr', 'sk', 'sl', 'et', 'lv', 'lt', 'mt', 'ga', 'cy', 'af', 'am', 'as', 'az', 'ba', 'be', 'bn', 'bo', 'br', 'bs', 'ca', 'eu', 'fa', 'fo', 'gl', 'gu', 'ha', 'haw', 'hy', 'is', 'jw', 'ka', 'kk', 'km', 'kn', 'la', 'lb', 'ln', 'lo', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'ne', 'nn', 'oc', 'pa', 'ps', 'sa', 'sd', 'si', 'sn', 'so', 'sq', 'sr', 'su', 'sw', 'ta', 'te', 'tg', 'tk', 'tl', 'tt', 'ur', 'uz', 'yi', 'yo', 'yue']
    
    input_language = None
    while input_language is None:
        try:
            lang_choice = input(f"请输入语言代码（直接回车使用默认: {default_language}）: ").strip()
            if not lang_choice:
                # 使用默认语言
                input_language = default_language
            else:
                lang_choice = lang_choice.lower()
                if lang_choice in valid_languages:
                    input_language = lang_choice
                else:
                    print(f"⚠ 未识别的语言代码 '{lang_choice}'，请重新输入")
                    print("提示: 输入 'auto' 可自动检测语言")
        except KeyboardInterrupt:
            print("\n使用默认语言")
            input_language = default_language
            break
    
    print(f"✓ 已选择语言: {input_language} ({'自动检测' if input_language == 'auto' else input_language})")
    print()
    
    # 让用户选择任务类型（在加载模型之前）
    print("=" * 60)
    print("选择任务类型")
    print("=" * 60)
    print("  - transcribe: 转录模式，输出与输入相同的语言")
    print("  - translate: 翻译模式，直接翻译成英文（如果懂英语，推荐选择此选项）")
    print()
    
    default_task = "transcribe"
    task = None
    while task is None:
        try:
            task_choice = input(f"请输入任务类型（直接回车使用默认: {default_task}）: ").strip().lower()
            if not task_choice:
                task = default_task
            elif task_choice in ["transcribe", "translate"]:
                task = task_choice
            else:
                print(f"⚠ 无效的任务类型 '{task_choice}'，请输入 'transcribe' 或 'translate'")
        except KeyboardInterrupt:
            print("\n使用默认任务类型")
            task = default_task
            break
    
    print(f"✓ 已选择任务类型: {task} ({'转录' if task == 'transcribe' else '翻译成英文'})")
    print()
    
    # 根据语言重新读取配置（语言特定配置优先）
    if CONFIG_MANAGER_AVAILABLE and config_manager:
        # 获取语言特定的 ASR 优化配置
        asr_opt = config_manager.get_language_specific_config(input_language, "asr_optimization")
        vac_chunk_size = asr_opt.get("vac_chunk_size", 0.08)
        agreement_n = asr_opt.get("agreement_n", 3)
        beam_size = asr_opt.get("beam_size", 5)
        temperature = asr_opt.get("temperature", 0.0)
        vad_threshold = asr_opt.get("vad_threshold", 0.6)
        
        # 获取语言特定的语速自适应配置
        speech_rate_config = config_manager.get_language_specific_config(input_language, "speech_rate_adaptive")
        vad_initial_silence_ms = speech_rate_config.get("initial_silence_ms", 1000)
        vad_min_silence_ms = speech_rate_config.get("min_silence_ms", 500)
        vad_max_silence_ms = speech_rate_config.get("max_silence_ms", 1500)
        
        print(f"✓ 已加载语言特定配置 ({input_language}): agreement_n={agreement_n}, vac_chunk_size={vac_chunk_size}, VAD静音={vad_min_silence_ms}-{vad_max_silence_ms}ms")
    else:
        # 没有配置管理器，使用默认值（已在前面设置）
        pass
    
    # ========== 翻译功能配置 ==========
    # 如果task是translate，不需要API翻译（Whisper直接翻译成英文）
    # 如果task是transcribe，需要API翻译成中文
    enable_translation = (task == "transcribe")
    
    # 初始化翻译管理器（如果需要）
    translation_manager = None
    translate_interval = 10.0  # 默认值
    if enable_translation:
        try:
            from translation_manager import TranslationManager
            
            # 从配置获取翻译间隔
            if CONFIG_MANAGER_AVAILABLE and config_manager:
                translate_interval = config_manager.get("translate_interval", 10.0)
            
            # 定义翻译结果输出回调（使用异步输出，与识别结果保持一致）
            def translation_output_callback(original_text: str, translated_text: str):
                """翻译结果输出回调"""
                # 使用全局的异步输出实例
                _async_output.print(f"🌐 {translated_text}")
                _async_output.flush()
            
            translation_manager = TranslationManager(
                translate_interval=translate_interval,
                output_callback=translation_output_callback
            )
            translation_manager.start()
            print(f"✓ 翻译管理器已启动（间隔: {translate_interval}秒）")
        except ImportError:
            print("⚠ 翻译管理器模块不可用，翻译功能将禁用")
            enable_translation = False
        except Exception as e:
            print(f"⚠ 翻译管理器初始化失败: {e}，翻译功能将禁用")
            enable_translation = False
    
    # 语言代码列表（常用）
    # 完整列表: af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh,yue
    
    # 确定输出语言
    if task == "translate":
        output_language = "英文"
    elif input_language == "auto":
        output_language = "自动检测（与输入相同）"
    else:
        output_language = input_language
    
    print(f"配置:")
    print(f"  - 输入语言: {input_language}")
    if input_language == "auto":
        print("    ⚠ 提示: 如果知道输入语言，建议明确指定（如 'zh' 中文）以提高识别准确度")
    print(f"  - 输出语言: {output_language}")
    print(f"  - 任务类型: {task} ({'转录' if task == 'transcribe' else '翻译成英文'})")
    print(f"  - 模型: {model_size}")
    if model_size in ['large-v2', 'large-v3', 'large']:
        print("    ⚠ 注意: 大模型在实时场景下速度较慢，建议使用 medium")
    if use_vac:
        print(f"  - VAC (语音活动检测): 已启用")
        print(f"    ✓ 自动按句子/半句处理（检测到 500ms 静音时触发）")
        print(f"    ✓ 更自然的识别节奏，减少延迟")
    else:
        print(f"  - VAC (语音活动检测): 未启用")
        print(f"    ⚠ 将使用固定时间间隔处理")
    if enable_translation:
        print(f"  - API翻译: 已启用（将翻译成中文，间隔: {translate_interval}秒）")
    else:
        if task == "translate":
            print(f"  - API翻译: 不需要（Whisper直接翻译成英文）")
        else:
            print(f"  - API翻译: 已禁用")
    print(f"  - 采样率: {SAMPLING_RATE} Hz")
    if not use_vac:
        # 只有非 VAC 模式才显示处理间隔
        if model_size in ['large-v2', 'large-v3', 'large']:
            actual_interval = max(CHUNK_DURATION, 2.0)
            print(f"  - 处理间隔: {actual_interval} 秒 (大模型自动调整为更长间隔)")
        else:
            print(f"  - 处理间隔: {CHUNK_DURATION} 秒")
    print()
    
    # 创建 ASR 对象
    asr = None
    load_errors = []
    
    if backend_choice == "whisper":
        # 使用原始 whisper 后端
        print(f"正在加载 Whisper 模型 ({model_size})...")
        model_cache_dir = os.path.join(os.path.dirname(__file__), "models")
        try:
            asr = WhisperTimestampedASR(lan=input_language, modelsize=model_size, cache_dir=model_cache_dir)
            if task == "translate":
                asr.set_translate_task()
            print("✓ Whisper 模型加载成功")
        except ImportError as e:
            print(f"✗ 模型加载失败: {e}")
            print("提示: 确保已安装 whisper 和 whisper-timestamped:")
            print("  pip install openai-whisper whisper-timestamped")
            sys.exit(1)
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            print("提示: 确保已安装 whisper: pip install openai-whisper whisper-timestamped")
            sys.exit(1)
    else:
        # 使用 faster-whisper 后端
        print(f"正在加载 fast-Whisper 模型 ({model_size}, {device.upper()}, {compute_type})...")
        model_cache_dir = os.path.join(os.path.dirname(__file__), "models_fast")
        
        # 尝试加载模型（应用底层优化参数）
        if device == "cuda":
            try:
                # 创建transcribe_kwargs，包含beam_size和temperature
                transcribe_kwargs = {
                    'beam_size': beam_size,
                    'temperature': temperature
                }
                asr = CustomFasterWhisperASR(
                    lan=input_language, 
                    modelsize=model_size, 
                    cache_dir=model_cache_dir,
                    device="cuda",
                    compute_type=compute_type,
                    device_index=low_level_params['device_index'],
                    num_workers=low_level_params['num_workers'],
                    transcribe_kwargs=transcribe_kwargs
                )
                print("✓ 模型加载成功（GPU 模式）")
                if low_level_params['device_index'] > 0:
                    print(f"  使用 GPU {low_level_params['device_index']}")
            except Exception as e:
                load_errors.append(("GPU", e))
                print(f"⚠ GPU 加载失败: {e}")
                print("  尝试自动切换到 CPU 模式...")
                device = "cpu"
                compute_type = "int8"
                # 重新优化 CPU 参数
                low_level_params = optimize_low_level_params(hardware, False, model_size)
        
        if asr is None:
            # CPU 模式或 GPU 失败后使用 CPU
            try:
                # 创建transcribe_kwargs，包含beam_size和temperature
                transcribe_kwargs = {
                    'beam_size': beam_size,
                    'temperature': temperature
                }
                asr = CustomFasterWhisperASR(
                    lan=input_language, 
                    modelsize=model_size, 
                    cache_dir=model_cache_dir,
                    device="cpu",
                    compute_type="int8",
                    num_workers=low_level_params['num_workers'],
                    cpu_threads=low_level_params.get('cpu_threads'),
                    transcribe_kwargs=transcribe_kwargs
                )
                print("✓ 模型加载成功（CPU 模式）")
                if low_level_params.get('cpu_threads'):
                    print(f"  使用 {low_level_params['cpu_threads']} 个 CPU 线程")
            except Exception as e:
                print(f"✗ CPU 模式也加载失败: {e}")
                print("提示: 确保已安装 faster-whisper: pip install faster-whisper")
                if load_errors:
                    print("\nGPU 加载错误详情:")
                    for mode, error in load_errors:
                        print(f"  {mode}: {error}")
                sys.exit(1)
    
    # 设置任务类型（translate 模式）
    if task == "translate":
        asr.set_translate_task()
        print("✓ 已设置为翻译模式（将翻译成英文）")
    
    def list_audio_devices(force_refresh=False):
        """列出可用的音频输入设备"""
        if force_refresh:
            # 强制刷新设备列表
            # 注意：sounddevice 的 query_devices() 每次都会重新查询系统设备
            # 但如果系统刚识别到新设备，可能需要短暂等待
            print("正在刷新设备列表，请稍候...")
            time.sleep(0.2)  # 短暂等待，让系统有时间识别新设备
        
        # 重新查询设备列表（每次都会重新查询系统设备列表）
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        default_input = sd.query_devices(kind='input')['name']
        
        print("可用的音频输入设备:")
        for i, dev in enumerate(input_devices):
            default = " (默认)" if dev['name'] == default_input else ""
            print(f"  [{i}] {dev['name']}{default}")
        print()
        return input_devices
    
    def select_device(input_devices):
        """让用户选择音频设备"""
        if len(input_devices) == 1:
            print(f"自动选择唯一设备: {input_devices[0]['name']}")
            return 0
        
        while True:
            try:
                choice = input("请选择设备编号（直接回车使用默认设备，输入 'r' 刷新设备列表）: ").strip()
                if choice.lower() == 'r':
                    # 刷新设备列表
                    print("\n正在刷新设备列表...")
                    refreshed_devices = list_audio_devices(force_refresh=True)
                    if len(refreshed_devices) != len(input_devices):
                        print(f"✓ 检测到设备变化（之前: {len(input_devices)} 个，现在: {len(refreshed_devices)} 个）")
                        input_devices = refreshed_devices
                    else:
                        # 检查设备名称是否有变化
                        old_names = {dev['name'] for dev in input_devices}
                        new_names = {dev['name'] for dev in refreshed_devices}
                        if old_names != new_names:
                            print(f"✓ 检测到设备变化")
                            input_devices = refreshed_devices
                        else:
                            print("设备列表未变化")
                    continue
                elif choice == "":
                    # 使用默认设备
                    default_input = sd.query_devices(kind='input')
                    for i, dev in enumerate(input_devices):
                        if dev['name'] == default_input['name']:
                            return i
                    return 0  # 如果找不到，返回第一个
                else:
                    device_idx = int(choice)
                    if 0 <= device_idx < len(input_devices):
                        return device_idx
                    else:
                        print(f"无效的设备编号，请输入 0-{len(input_devices)-1} 之间的数字，或输入 'r' 刷新设备列表")
            except ValueError:
                print("请输入有效的数字，或输入 'r' 刷新设备列表")
            except KeyboardInterrupt:
                return None
    
    def record_session(online, stream, device_idx=None, 
                      model_size="medium", chunk_duration=1.0, use_vac=False,
                      config_manager=None, perf_display=None, device_protector=None,
                      input_language="auto", use_async_output=True, translation_manager=None):
        """执行一次录音会话（支持翻译功能）
        
        Args:
            online: ASR处理器对象
            stream: 已打开的音频流对象（不再在函数内部打开/关闭）
            device_idx: 设备索引（可选，用于首次选择设备）
            model_size: 模型大小
            chunk_duration: 处理间隔
            use_vac: 是否使用VAC模式
            use_async_output: 是否使用异步输出（避免输出阻塞主循环）
        """
        # 启动异步输出（如果启用）
        if use_async_output:
            _async_output.start()
        
        try:
            # 重新初始化处理器
            online.init()
        except Exception as e:
            error_msg = f"ASR处理器初始化失败: {e}"
            if perf_display:
                perf_display.display_error("初始化失败", error_msg, "请检查模型是否正确加载")
            else:
                print(f"✗ {error_msg}")
            return False
        
        # 说话密集程度检测（用于动态调整静音检测时间）
        recognition_times = []  # 记录最近几次识别结果的时间戳
        last_silence_adjustment_time = time.time()  # 上次调整静音检测时间的时间
        silence_adjustment_interval = 2.0  # 每2秒检查一次是否需要调整
        
        # 调试：确认进入主循环
        try:
            # 计算读取块大小（不再打开音频流，使用传入的stream）
            if use_vac:
                read_chunk_size = int(0.04 * SAMPLING_RATE)  # VAC 推荐：0.04 秒（512 样本）
            elif model_size in ['large-v2', 'large-v3', 'large']:
                read_chunk_size = int(0.5 * SAMPLING_RATE)  # 大模型：每次读取 0.5 秒
            else:
                read_chunk_size = int(0.3 * SAMPLING_RATE)  # 中小模型：每次读取 0.3 秒
            
            # 使用传入的stream，不再使用with语句
            if stream is None:
                if perf_display:
                    perf_display.display_error("音频流错误", "音频流未打开", "请重新启动程序")
                else:
                    print("⚠ 错误：音频流未打开")
                return False
            
            # 初始化会话变量
            last_process_time = time.time()
            last_activity_time = time.time()
            last_heartbeat_time = time.time()
            no_audio_warning_shown = False
            
            # 初始化跳句日志记录器
            skip_logger = logging.getLogger('SkipLogger')
            skip_logger.setLevel(logging.INFO)
            
            # 根据配置决定是否输出到控制台
            console_log_enabled = True  # 默认启用
            if config_manager:
                console_log_enabled = config_manager.get("logging.console_log_enabled", True)
            
            # 如果已有handlers，需要根据配置重新配置（特别是控制台输出）
            if skip_logger.handlers:
                # 移除所有现有的控制台handler（StreamHandler）
                handlers_to_remove = [h for h in skip_logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
                for handler in handlers_to_remove:
                    skip_logger.removeHandler(handler)
                    handler.close()
            
            # 避免重复添加handler（只在没有handler时添加）
            if not skip_logger.handlers:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                
                # 控制台输出（如果启用）
                if console_log_enabled:
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(formatter)
                    skip_logger.addHandler(console_handler)
                
                # 文件输出（保存到logs目录）
                log_dir = "logs"
                os.makedirs(log_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(log_dir, f"skip_{timestamp}.log")
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(formatter)
                skip_logger.addHandler(file_handler)
                skip_logger.info(f"跳句日志已启用，日志文件: {log_file}")
            else:
                # 如果已有handlers，只添加控制台handler（如果需要）
                if console_log_enabled:
                    # 检查是否已有控制台handler
                    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) 
                                     for h in skip_logger.handlers)
                    if not has_console:
                        formatter = logging.Formatter(
                            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S'
                        )
                        console_handler = logging.StreamHandler()
                        console_handler.setFormatter(formatter)
                        skip_logger.addHandler(console_handler)
            
            # 会话开始提示已在主循环中显示，这里不需要重复显示
            
            # 初始化改进的跳句检测器（如果可用）
            skip_detector = None
            if SKIP_DETECTOR_AVAILABLE:
                # 根据语言获取特定配置
                if config_manager:
                    skip_config = config_manager.get_language_specific_config(
                        input_language, "skip_detector"
                    )
                    skip_detector = ImprovedSkipDetector(
                        similarity_threshold=skip_config.get("similarity_threshold", 0.85),
                        time_window=skip_config.get("time_window", 3.0),
                        min_length=skip_config.get("min_length", 2),
                        use_edit_distance=skip_config.get("use_edit_distance", True)
                    )
                else:
                    # 没有配置管理器，使用默认值
                    skip_detector = ImprovedSkipDetector(
                        similarity_threshold=0.85,
                        time_window=3.0,
                        min_length=2,
                        use_edit_distance=True
                    )
            
            # 初始化音频级别去重器（如果可用）
            audio_deduplicator = None
            if AUDIO_DEDUPLICATOR_AVAILABLE and config_manager:
                dedup_config = config_manager.get("audio_deduplication", {})
                if dedup_config.get("enable", False):
                    try:
                        audio_deduplicator = AudioDeduplicator(
                            similarity_threshold=dedup_config.get("similarity_threshold", 0.95),
                            time_window=dedup_config.get("time_window", 3.0),
                            min_audio_length=dedup_config.get("min_audio_length", 0.1),
                            enable=True
                        )
                        if perf_display:
                            perf_display.set_audio_deduplicator(audio_deduplicator)
                            perf_display.display_info("音频去重已启用：将在识别前过滤重复音频")
                    except Exception as e:
                        # 音频去重器初始化失败，不影响主程序运行
                        if perf_display:
                            perf_display.display_warning(f"音频去重初始化失败: {e}")
                        else:
                            print(f"⚠ 音频去重初始化失败: {e}")
                        audio_deduplicator = None
            
            # 初始化人声分离器（如果启用）
            vocal_separator = None
            if VOCAL_SEPARATION_AVAILABLE and config_manager:
                sep_config = config_manager.get("vocal_separation", {})
                if sep_config.get("enable", False):
                    method = sep_config.get("method", "demucs")
                    if method.lower() != "none":
                        try:
                            if method.lower() == "demucs":
                                model_name = sep_config.get("demucs_model", "htdemucs")
                                model_path = sep_config.get("demucs_model_path", "")
                                # 如果路径为空字符串，则使用None（使用默认路径）
                                model_path = model_path if model_path and model_path.strip() else None
                                vocal_separator = create_separator("demucs", SAMPLING_RATE, 
                                                                   model_name=model_name, 
                                                                   model_path=model_path)
                            elif method.lower() == "spleeter":
                                model_type = sep_config.get("spleeter_model", "2stems")
                                vocal_separator = create_separator("spleeter", SAMPLING_RATE, model_type=model_type)
                            elif method.lower() == "filter":
                                low_cut = sep_config.get("filter_low_cut", 85.0)
                                high_cut = sep_config.get("filter_high_cut", 3400.0)
                                vocal_separator = create_separator("filter", SAMPLING_RATE, low_cut=low_cut, high_cut=high_cut)
                            
                            if vocal_separator and vocal_separator.is_available():
                                if perf_display:
                                    perf_display.display_success(f"人声分离已启用: {method}")
                                else:
                                    print(f"✓ 人声分离已启用: {method}")
                        except Exception as e:
                            if perf_display:
                                perf_display.display_warning(f"人声分离初始化失败: {e}")
                            else:
                                print(f"⚠ 人声分离初始化失败: {e}")
                            vocal_separator = None
            
            if use_vac:
                # VAC 模式：简化逻辑，VAC 会自动处理语音活动检测
                # 只需要持续读取音频并插入，VAC 会在检测到语音结束时自动处理
                last_recognized_text = ""  # 用于去重（兼容旧代码）
                # print(f"[DEBUG] 进入VAC模式主循环", flush=True)
                # loop_count = 0
                while True:
                    # loop_count += 1
                    # if loop_count % 100 == 0:  # 每100次循环输出一次，避免刷屏
                    #     print(f"[DEBUG] 主循环运行中... (已循环 {loop_count} 次)", flush=True)
                    
                    try:
                        # 从麦克风读取音频
                        audio_chunk, overflowed = stream.read(read_chunk_size)
                        audio_chunk = audio_chunk.flatten()
                        
                        if overflowed:
                            print("⚠ 音频缓冲区溢出", end='\r')
                        
                        # 检查是否有实际音频数据
                        if np.any(np.abs(audio_chunk) > 1e-6):
                            last_activity_time = time.time()
                            no_audio_warning_shown = False
                        else:
                            if time.time() - last_activity_time > 5 and not no_audio_warning_shown:
                                # 显示状态提示
                                print("⚠ 检测到长时间无音频输入，请检查麦克风...", end='\r')
                                no_audio_warning_shown = True
                        
                        # 人声分离（如果启用）
                        if vocal_separator and vocal_separator.is_available():
                            try:
                                # 分离人声和背景音乐
                                vocal_audio, _ = vocal_separator.separate(audio_chunk)
                                # 使用分离后的人声音频
                                audio_chunk = vocal_audio
                            except Exception as e:
                                # 分离失败，使用原始音频
                                pass
                        
                        # 音频级别去重（如果启用）- 在进入ASR模型之前检测重复音频
                        should_skip_audio = False
                        if audio_deduplicator:
                            try:
                                skip_audio, skip_reason, skip_details = audio_deduplicator.should_skip(
                                    audio_chunk, 
                                    sample_rate=SAMPLING_RATE,
                                    current_time=time.time()
                                )
                                if skip_audio:
                                    should_skip_audio = True
                                    # 跳过此音频块，不发送到ASR模型
                                    # 可选：记录日志（但为了性能，这里不记录）
                                    continue
                            except Exception as e:
                                # 去重检测失败，继续处理（保守策略）
                                pass
                        
                        # 直接插入音频，VAC 会自动检测语音活动
                        # VAC 会在检测到 500ms 静音时自动触发处理
                        online.insert_audio_chunk(audio_chunk)
                        
                        current_time = time.time()
                        
                        # 定期调用 process_iter（VAC 模式下，它会在检测到语音结束时返回结果）
                        has_result = False
                        if (current_time - last_process_time) >= 0.5:  # 每 0.5 秒检查一次
                            # print(f"[DEBUG] 调用 process_iter()...", flush=True)
                            try:
                                result = online.process_iter()
                                
                                # 输出结果（VAC 会在检测到句子结束时返回结果）
                                if result[0] is not None:
                                    # 有识别结果，清除nonvoice提示
                                    print("\r" + " " * 50 + "\r", end='', flush=True)
                                    has_result = True
                                    
                                    beg_time, end_time, text = result
                                    # 验证时间戳有效性（避免时间戳异常导致的问题）
                                    # 注意：end_time == beg_time 可能是正常的极短片段（如单字），允许通过
                                    if end_time < beg_time:
                                        # 时间戳异常（结束时间小于开始时间），跳过此次结果
                                        if perf_display:
                                            perf_display.display_warning(f"时间戳异常: {beg_time:.2f}s-{end_time:.2f}s，已跳过")
                                        else:
                                            print(f"⚠ 时间戳异常: {beg_time:.2f}s-{end_time:.2f}s，已跳过")
                                        last_process_time = current_time
                                        continue
                                    
                                    # 如果时间戳相等，检查文本长度，如果文本过长可能是异常
                                    if end_time == beg_time and text and len(text.strip()) > 50:
                                        # 时间戳相等但文本很长，可能是异常，跳过
                                        if perf_display:
                                            perf_display.display_warning(f"时间戳异常: {beg_time:.2f}s (文本过长:{len(text.strip())}字)，已跳过")
                                        else:
                                            print(f"⚠ 时间戳异常: {beg_time:.2f}s (文本过长:{len(text.strip())}字)，已跳过")
                                        last_process_time = current_time
                                        continue
                                    
                                    if text and text.strip():
                                        text_clean = text.strip()
                                        
                                        # 使用改进的跳句检测器（如果可用）
                                        should_skip = False
                                        skip_reason = None
                                        skip_details = None
                                        
                                        if skip_detector is not None:
                                            should_skip, skip_reason, skip_details = skip_detector.should_skip(text_clean, current_time)
                                            
                                            if should_skip:
                                                # 记录跳句日志
                                                if skip_details:
                                                    details_str = ', '.join([f"{k}={v}" if not isinstance(v, float) else f"{k}={v:.3f}" 
                                                                           for k, v in skip_details.items() if k != 'type'])
                                                    skip_logger.info(f"[跳句-recognition] 原因: {skip_reason}, 详情: {details_str}, 时间: {beg_time:.2f}-{end_time:.2f}s")
                                                else:
                                                    skip_logger.info(f"[跳句-recognition] 原因: {skip_reason}, 时间: {beg_time:.2f}-{end_time:.2f}s")
                                                last_process_time = current_time
                                                continue
                                        else:
                                            # 回退到基础去重逻辑
                                            if text_clean == last_recognized_text or \
                                               (last_recognized_text and text_clean in last_recognized_text and len(text_clean) < len(last_recognized_text)):
                                                # 这是重复或部分结果，跳过
                                                # 记录跳句日志
                                                if text_clean == last_recognized_text:
                                                    skip_reason = "duplicate"
                                                    skip_details = f"完全重复: '{text_clean}' == '{last_recognized_text}'"
                                                else:
                                                    skip_reason = "partial"
                                                    skip_details = f"部分重复: '{text_clean}' 是 '{last_recognized_text}' 的一部分"
                                                skip_logger.info(f"[跳句-recognition] 原因: {skip_reason}, 详情: {skip_details}, 时间: {beg_time:.2f}-{end_time:.2f}s")
                                                last_process_time = current_time
                                                continue
                                        
                                        # 更新最后识别的文本（兼容旧代码）
                                        last_recognized_text = text_clean
                                        
                                        # 有识别结果，清除所有状态提示
                                        if use_async_output:
                                            _async_output.print_no_newline("\r" + " " * 100 + "\r")
                                        else:
                                            print("\r" + " " * 100 + "\r", end='', flush=True)
                                        
                                        # 显示识别结果（使用异步输出避免阻塞）
                                        if use_async_output:
                                            _async_output.print(f"💬 {text}")
                                        else:
                                            print(f"💬 {text}", flush=True)
                                        
                                        # 如果启用翻译，添加到翻译队列
                                        if translation_manager is not None:
                                            translation_manager.add_text(text_clean)
                                        
                                        # 记录识别结果时间戳（用于说话密集程度检测）
                                        recognition_times.append(current_time)
                                        # 只保留最近5次识别结果的时间戳
                                        if len(recognition_times) > 5:
                                            recognition_times.pop(0)
                                        
                                        # 异步flush（如果启用）
                                        if use_async_output:
                                            _async_output.flush()
                                        else:
                                            sys.stdout.flush()
                                        last_activity_time = current_time
                                    
                                    last_process_time = current_time
                                    
                                    # 动态调整静音检测时间（每2秒检查一次）
                                    if use_vac and hasattr(online, 'set_silence_duration') and \
                                       (current_time - last_silence_adjustment_time) >= silence_adjustment_interval:
                                        if len(recognition_times) >= 2:
                                            # 计算最近几次识别结果的平均时间间隔
                                            intervals = []
                                            for i in range(1, len(recognition_times)):
                                                intervals.append(recognition_times[i] - recognition_times[i-1])
                                            avg_interval = sum(intervals) / len(intervals) if intervals else 5.0
                                            
                                            # 根据平均时间间隔调整静音检测时间
                                            # 间隔短（<2秒）= 密集说话，缩短静音检测时间（200-300ms）
                                            # 间隔长（>5秒）= 稀疏说话，延长静音检测时间（800-1000ms）
                                            if avg_interval < 2.0:
                                                # 密集说话：使用较短的静音检测时间
                                                new_silence_ms = int(200 + (avg_interval / 2.0) * 100)  # 200-300ms
                                            elif avg_interval > 5.0:
                                                # 稀疏说话：使用较长的静音检测时间
                                                new_silence_ms = int(600 + min((avg_interval - 5.0) / 5.0, 1.0) * 400)  # 600-1000ms
                                            else:
                                                # 中等密度：使用中等静音检测时间
                                                new_silence_ms = int(300 + (avg_interval - 2.0) / 3.0 * 300)  # 300-600ms
                                            
                                            # 应用调整
                                            if online.set_silence_duration(new_silence_ms):
                                                # 只在成功调整时更新时间和输出提示
                                                last_silence_adjustment_time = current_time
                                                # 可选：输出调整信息（调试用，可以注释掉）
                                                # print(f"\r[静音检测: {new_silence_ms}ms (间隔: {avg_interval:.1f}s)]", end='', flush=True)
                                        else:
                                            last_silence_adjustment_time = current_time
                            except Exception as e:
                                print(f"\n⚠ 处理错误: {e}")
                                print("继续录音中...")
                                sys.stdout.flush()
                                last_process_time = current_time
                        
                        # 如果没有识别结果，显示nonvoice闪烁提示（每0.2秒更新一次，实现流畅闪烁）
                        # 如果没有识别结果，显示nonvoice闪烁提示
                        if not has_result and (current_time - last_process_time) >= 0.2:
                            # 使用时间戳创建闪烁效果（每0.5秒切换一次）
                            blink_state = int(current_time * 2) % 2
                            if blink_state == 0:
                                print("\r🔇 nonvoice", end='', flush=True)
                            else:
                                print("\r   nonvoice", end='', flush=True)
                    
                    except sd.PortAudioError as e:
                        print(f"\n✗ 音频流错误: {e}")
                        raise
                    except Exception as e:
                        print(f"\n⚠ 读取音频时发生错误: {e}")
                        time.sleep(0.1)
                        continue
                    
                    # 心跳检测
                    current_time = time.time()
                    if current_time - last_heartbeat_time > 10:
                        if current_time - last_activity_time < 2:
                            last_heartbeat_time = current_time
                        else:
                            # 显示状态提示
                            print("⏳ 录音中... (等待语音输入)", end='\r')
                            last_heartbeat_time = current_time
            
            else:
                # 非 VAC 模式：使用原有的缓冲和处理逻辑
                temp_buffer = np.array([], dtype=np.float32)
                min_buffer_size = int(chunk_duration * SAMPLING_RATE)
                last_recognized_text = ""  # 用于去重（兼容旧代码）
                
                while True:
                    try:
                        # 从麦克风读取音频（使用设备保护器或直接读取）
                        if device_protector is not None:
                            audio_chunk, overflowed, read_error = device_protector.read_audio(read_chunk_size)
                            if audio_chunk is None:
                                if read_error:
                                    if "设备已恢复" in read_error:
                                        # 设备已恢复，继续下一次循环
                                        if perf_display:
                                            perf_display.display_success("音频设备已恢复")
                                        continue
                                    else:
                                        # 尝试恢复
                                        if perf_display:
                                            perf_display.display_warning(f"读取音频失败: {read_error}")
                                            perf_display.display_progress("正在尝试恢复音频流...")
                                        success, new_stream, recover_error = device_protector.recover_stream(
                                            samplerate=SAMPLING_RATE,
                                            channels=1,
                                            blocksize=read_chunk_size,
                                            dtype='float32'
                                        )
                                        if success:
                                            stream = new_stream
                                            if perf_display:
                                                perf_display.clear()
                                                perf_display.display_success("音频流已恢复，继续录音")
                                            continue
                                        else:
                                            if perf_display:
                                                perf_display.display_error(
                                                    "设备恢复失败",
                                                    recover_error,
                                                    "请检查麦克风是否已连接并启用"
                                                )
                                            raise sd.PortAudioError(recover_error)
                            audio_chunk = audio_chunk.flatten()
                        else:
                            audio_chunk, overflowed = stream.read(read_chunk_size)
                            audio_chunk = audio_chunk.flatten()
                        
                        if overflowed:
                            if perf_display:
                                perf_display.display_warning("音频缓冲区溢出")
                            else:
                                print("⚠ 音频缓冲区溢出", end='\r')
                        
                        # 检查是否有实际音频数据
                        if np.any(np.abs(audio_chunk) > 1e-6):
                            last_activity_time = time.time()
                            no_audio_warning_shown = False
                        else:
                            if time.time() - last_activity_time > 5 and not no_audio_warning_shown:
                                # 显示状态提示
                                print("⚠ 检测到长时间无音频输入，请检查麦克风...", end='\r')
                                no_audio_warning_shown = True
                        
                        # 累积到临时缓冲区
                        temp_buffer = np.append(temp_buffer, audio_chunk)
                        
                        # 按处理间隔定期处理
                        current_time = time.time()
                        time_elapsed = current_time - last_process_time
                        
                        if (time_elapsed >= chunk_duration and len(temp_buffer) >= min_buffer_size) or \
                           (len(temp_buffer) >= min_buffer_size * 2):
                            
                            # 音频级别去重（如果启用）- 在进入ASR模型之前检测重复音频
                            should_skip_audio = False
                            if audio_deduplicator:
                                try:
                                    skip_audio, skip_reason, skip_details = audio_deduplicator.should_skip(
                                        temp_buffer, 
                                        sample_rate=SAMPLING_RATE,
                                        current_time=current_time
                                    )
                                    if skip_audio:
                                        should_skip_audio = True
                                        # 跳过此音频块，不发送到ASR模型
                                        temp_buffer = np.array([], dtype=np.float32)
                                        last_process_time = current_time
                                        continue
                                except Exception as e:
                                    # 去重检测失败，继续处理（保守策略）
                                    pass
                            
                            online.insert_audio_chunk(temp_buffer)
                            temp_buffer = np.array([], dtype=np.float32)
                            
                            try:
                                result = online.process_iter()
                                # if result[0] is not None:
                                #     beg_time, end_time, text = result
                                #     print(f"[DEBUG] process_iter返回结果: text='{text}', 长度={len(text) if text else 0}", flush=True)
                                
                                if result[0] is not None:
                                    beg_time, end_time, text = result
                                    # 验证时间戳有效性（避免时间戳异常导致的问题）
                                    # 注意：end_time == beg_time 可能是正常的极短片段（如单字），允许通过
                                    if end_time < beg_time:
                                        # 时间戳异常（结束时间小于开始时间），跳过此次结果
                                        if perf_display:
                                            perf_display.display_warning(f"时间戳异常: {beg_time:.2f}s-{end_time:.2f}s，已跳过")
                                        else:
                                            print(f"⚠ 时间戳异常: {beg_time:.2f}s-{end_time:.2f}s，已跳过")
                                        continue
                                    
                                    # 如果时间戳相等，检查文本长度，如果文本过长可能是异常
                                    if end_time == beg_time and text and len(text.strip()) > 50:
                                        # 时间戳相等但文本很长，可能是异常，跳过
                                        if perf_display:
                                            perf_display.display_warning(f"时间戳异常: {beg_time:.2f}s (文本过长:{len(text.strip())}字)，已跳过")
                                        else:
                                            print(f"⚠ 时间戳异常: {beg_time:.2f}s (文本过长:{len(text.strip())}字)，已跳过")
                                        continue
                                    
                                    if text and text.strip():
                                        # 验证时间戳有效性
                                        if end_time <= beg_time:
                                            # 时间戳异常，跳过此次结果
                                            print(f"⚠ 时间戳异常: {beg_time:.2f}s-{end_time:.2f}s，已跳过")
                                            continue
                                        
                                        text_clean = text.strip()
                                        
                                        # 使用改进的跳句检测器（如果可用）
                                        should_skip = False
                                        skip_reason = None
                                        skip_details = None
                                        
                                        if skip_detector is not None:
                                            should_skip, skip_reason, skip_details = skip_detector.should_skip(text_clean, current_time)
                                            
                                            if should_skip:
                                                # 记录跳句日志
                                                if skip_details:
                                                    details_str = ', '.join([f"{k}={v}" if not isinstance(v, float) else f"{k}={v:.3f}" 
                                                                           for k, v in skip_details.items() if k != 'type'])
                                                    skip_logger.info(f"[跳句-recognition] 原因: {skip_reason}, 详情: {details_str}, 时间: {beg_time:.2f}-{end_time:.2f}s")
                                                else:
                                                    skip_logger.info(f"[跳句-recognition] 原因: {skip_reason}, 时间: {beg_time:.2f}-{end_time:.2f}s")
                                                continue
                                        else:
                                            # 回退到基础去重逻辑
                                            if text_clean == last_recognized_text or \
                                               (last_recognized_text and text_clean in last_recognized_text and len(text_clean) < len(last_recognized_text)):
                                                # 这是重复或部分结果，跳过
                                                # 记录跳句日志
                                                if text_clean == last_recognized_text:
                                                    skip_reason = "duplicate"
                                                    skip_details = f"完全重复: '{text_clean}' == '{last_recognized_text}'"
                                                else:
                                                    skip_reason = "partial"
                                                    skip_details = f"部分重复: '{text_clean}' 是 '{last_recognized_text}' 的一部分"
                                                skip_logger.info(f"[跳句-recognition] 原因: {skip_reason}, 详情: {skip_details}, 时间: {beg_time:.2f}-{end_time:.2f}s")
                                                continue
                                        
                                        # 更新最后识别的文本（兼容旧代码）
                                        last_recognized_text = text_clean
                                        
                                        # 有识别结果，清除所有状态提示
                                        print("\r" + " " * 100 + "\r", end='', flush=True)
                                        
                                        # 显示识别结果
                                        print(f"💬 {text}", flush=True)
                                        
                                        sys.stdout.flush()
                                        last_activity_time = current_time
                            except Exception as e:
                                print(f"\n⚠ 处理错误: {e}")
                                print("继续录音中...")
                                sys.stdout.flush()
                            
                            last_process_time = current_time
                    
                    except sd.PortAudioError as e:
                        print(f"\n✗ 音频流错误: {e}")
                        raise
                    except Exception as e:
                        print(f"\n⚠ 读取音频时发生错误: {e}")
                        time.sleep(0.1)
                        continue
                    
                    # 心跳检测
                    current_time = time.time()
                    if current_time - last_heartbeat_time > 10:
                        if current_time - last_activity_time < 2:
                            last_heartbeat_time = current_time
                        else:
                            print("⏳ 录音中... (等待音频输入)", end='\r')
                            last_heartbeat_time = current_time
                    
        except KeyboardInterrupt:
            print()
            print("\n正在停止当前会话...")
            
            # 翻译功能已移除，无需清理
            
            # whisper_streaming 内部已经管理了所有音频，直接调用 finish 即可
            # 获取最后的结果
            try:
                final_result = online.finish()
                if final_result[0] is not None:
                    beg_time, end_time, text = final_result
                    # 验证时间戳有效性
                    if end_time > beg_time and text.strip():
                        # 有识别结果，清除所有状态提示
                        print("\r" + " " * 100 + "\r", end='', flush=True)
                        
                        # 显示识别结果
                        print(f"💬 {text}")
            except:
                pass
            
            print("✓ 当前会话已停止")
            return True
        
        except sd.PortAudioError as e:
            print(f"\n✗ 音频设备错误: {e}")
            print("提示: 请检查麦克风是否已连接并启用")
            # whisper_streaming 内部已经管理了所有音频，直接调用 finish 即可
            try:
                final_result = online.finish()
                if final_result[0] is not None:
                    beg_time, end_time, text = final_result
                    # 验证时间戳有效性（允许相等，但检查文本长度）
                    if end_time >= beg_time and text.strip():
                        # 如果时间戳相等但文本过长，可能是异常
                        if end_time == beg_time and len(text.strip()) > 50:
                            # 跳过异常的长文本
                            pass
                        else:
                            # 有识别结果，清除所有状态提示
                            print("\r" + " " * 100 + "\r", end='', flush=True)
                            
                            # 显示识别结果
                            print(f"💬 {text}")
            except:
                pass
            return False
        except KeyboardInterrupt:
            # 用户中断，正常退出
            print("\n\n用户中断录音")
            return False
        except Exception as e:
            error_msg = f"发生未预期的错误: {e}"
            print(f"\n✗ {error_msg}")
            import traceback
            traceback.print_exc()
            
            # 尝试显示错误信息
            if perf_display:
                perf_display.display_error("录音会话错误", str(e), "请检查日志获取详细信息")
            
            # whisper_streaming 内部已经管理了所有音频，直接调用 finish 即可
            try:
                final_result = online.finish()
                if final_result[0] is not None:
                    beg_time, end_time, text = final_result
                    # 验证时间戳有效性（允许相等，但检查文本长度）
                    if end_time >= beg_time and text.strip():
                        # 如果时间戳相等但文本过长，可能是异常
                        if end_time == beg_time and len(text.strip()) > 50:
                            # 跳过异常的长文本
                            pass
                        else:
                            # 有识别结果，清除所有状态提示
                            print("\r" + " " * 100 + "\r", end='', flush=True)
                            
                            # 显示识别结果
                            print(f"💬 {text}")
            except:
                pass
            return False
    
    # 创建在线处理对象（只需要创建一次）
    # 根据配置选择使用 VAC 或普通处理器
    if use_vac:
        try:
            # 检查是否有 torch（VAC 需要）
            import torch
            print("正在初始化 VAC (语音活动检测)...")
            
            # 创建过滤logfile，过滤掉"no online update, only VAD"消息
            class FilteredLogFile:
                def __init__(self, original_file):
                    self.original_file = original_file
                
                def write(self, text):
                    # 完全过滤掉所有输出，因为我们在主循环中自己处理状态显示
                    pass
                
                def flush(self):
                    self.original_file.flush()
            
            filtered_logfile = FilteredLogFile(sys.stderr)
            
            # 尝试使用增强的处理器（如果可用）
            try:
                from enhanced_asr_processor import EnhancedVACOnlineASRProcessor
                # 使用增强的VAC处理器（支持 Local Agreement-n、动态缓冲区等）
                online = EnhancedVACOnlineASRProcessor(
                    online_chunk_size=vac_chunk_size,
                    asr=asr,
                    tokenizer=None,  # 使用 segment 模式，不需要 tokenizer
                    logfile=filtered_logfile,
                    buffer_trimming=("segment", 15),  # 缓冲区修剪：segment 模式，15秒阈值
                    agreement_n=agreement_n,  # Local Agreement-n（从配置读取）
                    enable_dynamic_buffer=True,  # 启用动态缓冲区管理
                    initial_silence_ms=vad_initial_silence_ms,  # 初始静音检测时间（从配置读取）
                    min_silence_ms=vad_min_silence_ms,  # 最小静音检测时间（从配置读取）
                    max_silence_ms=vad_max_silence_ms,  # 最大静音检测时间（从配置读取）
                    vad_threshold=vad_threshold  # VAD阈值（从配置读取）
                )
                print("✓ 增强 VAC 处理器初始化成功")
                print(f"  - 支持 Local Agreement-{agreement_n} 策略")
                print(f"  - VAD静音检测: {vad_min_silence_ms}-{vad_max_silence_ms}ms")
                print(f"  - 音频块大小: {vac_chunk_size}秒")
                print("  - 支持动态缓冲区管理")
                print("  - 优化的 Init Prompt 提取")
            except ImportError:
                # 回退到原版动态VAC处理器
                online = DynamicVACOnlineASRProcessor(
                vac_chunk_size,  # VAC 音频块大小
                asr,
                tokenizer=None,  # 使用 segment 模式，不需要 tokenizer
                logfile=filtered_logfile,
                    buffer_trimming=("segment", 15),  # 缓冲区修剪：segment 模式，15秒阈值
                    initial_silence_ms=500,  # 初始静音检测时间
                    min_silence_ms=200,  # 最小静音检测时间（密集说话时）
                    max_silence_ms=1000  # 最大静音检测时间（稀疏说话时）
            )
                print("✓ VAC 初始化成功（支持动态调整）")
            print("✓ VAC 初始化成功（支持动态调整）")
            print("  - 将自动检测语音活动，按句子/半句处理")
            print("  - 初始静音检测时间: 500ms")
            print("  - 动态调整范围: 200ms（密集）~ 1000ms（稀疏）")
        except ImportError:
            print("⚠ 未安装 torch，无法使用 VAC")
            print("  将使用普通模式（固定时间间隔）")
            print("  可以安装: pip install torch torchaudio")
            use_vac = False
            # 尝试使用增强的处理器（如果可用）
            try:
                from enhanced_asr_processor import EnhancedOnlineASRProcessor
                online = EnhancedOnlineASRProcessor(
                    asr=asr,
                    tokenizer=None,
                    buffer_trimming=("segment", 15),
                    logfile=filtered_logfile,
                    agreement_n=agreement_n,
                    enable_dynamic_buffer=True
                )
                print("✓ 增强处理器初始化成功（无VAC）")
            except ImportError:
                online = OnlineASRProcessor(asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=filtered_logfile)
                print("✓ 普通处理器初始化成功（无VAC）")
        except Exception as e:
            print(f"⚠ VAC 初始化失败: {e}")
            print("  将使用普通模式（固定时间间隔）")
            use_vac = False
            # 尝试使用增强的处理器（如果可用）
            try:
                from enhanced_asr_processor import EnhancedOnlineASRProcessor
                online = EnhancedOnlineASRProcessor(
                    asr=asr,
                    tokenizer=None,
                    buffer_trimming=("segment", 15),
                    logfile=filtered_logfile,
                    agreement_n=agreement_n,
                    enable_dynamic_buffer=True
                )
                print("✓ 增强处理器初始化成功（无VAC，回退模式）")
            except ImportError:
                online = OnlineASRProcessor(asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=filtered_logfile)
                print("✓ 普通处理器初始化成功（无VAC，回退模式）")
    else:
        # 尝试使用增强的处理器（如果可用）
        try:
            from enhanced_asr_processor import EnhancedOnlineASRProcessor
            online = EnhancedOnlineASRProcessor(
                asr=asr,
                tokenizer=None,
                buffer_trimming=("segment", 15),
                logfile=sys.stderr,
                agreement_n=2,
                enable_dynamic_buffer=True
            )
            print("✓ 增强处理器初始化成功（无VAC模式）")
        except ImportError:
            online = OnlineASRProcessor(asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=sys.stderr)
            print("✓ 普通处理器初始化成功（无VAC模式）")
    
    print()
    print("=" * 60)
    print("开始录音会话")
    print("=" * 60)
    print()
    
    # 列出设备并选择（只在程序启动时选择一次）
    print("正在检测音频设备...")
    input_devices = list_audio_devices(force_refresh=True)
    
    # 选择设备
    device_idx = select_device(input_devices)
    if device_idx is None:
        print("未选择设备，退出程序")
        return
    
    selected_device = input_devices[device_idx]
    print(f"已选择设备: {selected_device['name']}")
    print()
    
    # 计算读取块大小
    if use_vac:
        read_chunk_size = int(0.04 * SAMPLING_RATE)  # VAC 推荐：0.04 秒（512 样本）
    elif model_size in ['large-v2', 'large-v3', 'large']:
        read_chunk_size = int(0.5 * SAMPLING_RATE)  # 大模型：每次读取 0.5 秒
    else:
        read_chunk_size = int(0.3 * SAMPLING_RATE)  # 中小模型：每次读取 0.3 秒
    
    # 打开音频流（在整个程序运行期间保持打开，避免反复占用/释放设备）
    if perf_display:
        perf_display.display_progress("正在打开麦克风...")
    else:
        print("正在打开麦克风...")
    
    # 使用音频设备保护器（如果可用）
    device_protector = None
    stream = None
    
    if DEVICE_PROTECTOR_AVAILABLE:
        device_config = config_manager.get_device_protector_config() if config_manager else {}
        device_protector = AudioDeviceProtector(
            max_retries=device_config.get('max_retries', 3),
            retry_delay=device_config.get('retry_delay', 1.0),
            check_interval=device_config.get('check_interval', 0.5)
        )
        success, stream, error = device_protector.open_stream(
            device_index=selected_device['index'],
            samplerate=SAMPLING_RATE,
            channels=1,
            blocksize=read_chunk_size,
            dtype='float32'
        )
        if not success:
            if perf_display:
                perf_display.clear()
                perf_display.display_error(
                    "设备打开失败",
                    error,
                    "请检查麦克风是否已连接并启用，或是否被其他程序占用"
                )
            else:
                print(f"✗ 无法打开音频流: {error}")
                print("提示: 请检查麦克风是否已连接并启用，或是否被其他程序占用")
            return
        
        # 设置性能显示器的设备保护器
        if perf_display:
            perf_display.set_device_protector(device_protector)
            perf_display.clear()
            perf_display.display_success("麦克风已就绪（设备保护已启用）")
        else:
            print("✓ 麦克风已就绪（设备保护已启用）")
        print()
    else:
        # 回退到基础模式
        try:
            stream = sd.InputStream(
                samplerate=SAMPLING_RATE,
                channels=1,
                dtype='float32',
                blocksize=read_chunk_size,
                device=selected_device['index']
            )
            stream.start()
            if perf_display:
                perf_display.clear()
                perf_display.display_success("麦克风已就绪")
            else:
                print("✓ 麦克风已就绪")
            print()
        except Exception as e:
            if perf_display:
                perf_display.clear()
                perf_display.display_error(
                    "设备打开失败",
                    str(e),
                    "请检查麦克风是否已连接并启用"
                )
            else:
                print(f"✗ 无法打开音频流: {e}")
                print("提示: 请检查麦克风是否已连接并启用")
            return
    
    # 主循环：可以多次录音会话（音频流保持打开）
    session_count = 0
    try:
        while True:
            session_count += 1
            if session_count > 1:
                print()
                print("=" * 60)
                print(f"开始新的录音会话 #{session_count}")
                print("=" * 60)
                print()
                
                # 询问是否要更改语言（可以重新加载模型）
                print(f"当前识别语言: {input_language} ({'自动检测' if input_language == 'auto' else input_language})")
                try:
                    lang_choice = input("是否更改识别语言？(直接回车保持当前，或输入语言代码，如 'zh'/'en'/'auto'): ").strip()
                    
                    if lang_choice:
                        lang_choice = lang_choice.lower()
                        # 验证语言代码
                        if lang_choice in valid_languages:
                            if lang_choice != input_language:
                                # 需要更改语言，重新加载模型
                                print(f"正在更改语言为: {lang_choice} ({'自动检测' if lang_choice == 'auto' else lang_choice})")
                                print("正在重新加载模型（这可能需要几秒钟）...")
                                
                                # 释放旧模型（释放GPU内存）
                                try:
                                    del asr
                                    del online
                                    import gc
                                    gc.collect()
                                    if device == "cuda":
                                        import torch
                                        torch.cuda.empty_cache()
                                except:
                                    pass
                                
                                # 更新语言
                                input_language = lang_choice
                                
                                # 根据新语言重新读取配置
                                if CONFIG_MANAGER_AVAILABLE and config_manager:
                                    # 获取语言特定的 ASR 优化配置
                                    asr_opt = config_manager.get_language_specific_config(input_language, "asr_optimization")
                                    vac_chunk_size = asr_opt.get("vac_chunk_size", 0.08)
                                    agreement_n = asr_opt.get("agreement_n", 3)
                                    beam_size = asr_opt.get("beam_size", 5)
                                    temperature = asr_opt.get("temperature", 0.0)
                                    vad_threshold = asr_opt.get("vad_threshold", 0.6)
                                    
                                    # 获取语言特定的语速自适应配置
                                    speech_rate_config = config_manager.get_language_specific_config(input_language, "speech_rate_adaptive")
                                    vad_initial_silence_ms = speech_rate_config.get("initial_silence_ms", 1000)
                                    vad_min_silence_ms = speech_rate_config.get("min_silence_ms", 500)
                                    vad_max_silence_ms = speech_rate_config.get("max_silence_ms", 1500)
                                    
                                    print(f"✓ 已加载语言特定配置 ({input_language}): agreement_n={agreement_n}, VAD静音={vad_min_silence_ms}-{vad_max_silence_ms}ms")
                                
                                # 重新创建 ASR 对象
                                if device == "cuda":
                                    try:
                                        transcribe_kwargs = {
                                            'beam_size': beam_size,
                                            'temperature': temperature
                                        }
                                        asr = CustomFasterWhisperASR(
                                            lan=input_language, 
                                            modelsize=model_size, 
                                            cache_dir=model_cache_dir,
                                            device="cuda",
                                            compute_type=compute_type,
                                            device_index=low_level_params['device_index'],
                                            num_workers=low_level_params['num_workers'],
                                            transcribe_kwargs=transcribe_kwargs
                                        )
                                        print("✓ 模型重新加载成功（GPU 模式）")
                                    except Exception as e:
                                        print(f"⚠ GPU 重新加载失败: {e}")
                                        print("  尝试使用 CPU 模式...")
                                        device = "cpu"
                                        compute_type = "int8"
                                        low_level_params = optimize_low_level_params(hardware, False, model_size)
                                
                                if asr is None or device == "cpu":
                                    try:
                                        transcribe_kwargs = {
                                            'beam_size': beam_size,
                                            'temperature': temperature
                                        }
                                        asr = CustomFasterWhisperASR(
                                            lan=input_language, 
                                            modelsize=model_size, 
                                            cache_dir=model_cache_dir,
                                            device="cpu",
                                            compute_type="int8",
                                            num_workers=low_level_params['num_workers'],
                                            cpu_threads=low_level_params.get('cpu_threads'),
                                            transcribe_kwargs=transcribe_kwargs
                                        )
                                        print("✓ 模型重新加载成功（CPU 模式）")
                                    except Exception as e:
                                        print(f"✗ 模型重新加载失败: {e}")
                                        print("提示: 将使用之前的模型和语言设置")
                                        # 恢复之前的语言
                                        input_language = lang_choice  # 已经更改了，保持新语言
                                
                                # 重新创建 online 处理器
                                if use_vac:
                                    # 确保 filtered_logfile 存在
                                    if 'filtered_logfile' not in locals():
                                        class FilteredLogFile:
                                            def __init__(self, original_file):
                                                self.original_file = original_file
                                            def write(self, text):
                                                pass
                                            def flush(self):
                                                self.original_file.flush()
                                        filtered_logfile = FilteredLogFile(sys.stderr)
                                    
                                    try:
                                        from enhanced_asr_processor import EnhancedVACOnlineASRProcessor
                                        online = EnhancedVACOnlineASRProcessor(
                                            online_chunk_size=vac_chunk_size,
                                            asr=asr,
                                            tokenizer=None,
                                            logfile=filtered_logfile,
                                            buffer_trimming=("segment", 15),
                                            initial_silence_ms=vad_initial_silence_ms,
                                            min_silence_ms=vad_min_silence_ms,
                                            max_silence_ms=vad_max_silence_ms,
                                            agreement_n=agreement_n,
                                            enable_dynamic_buffer=True,
                                            vad_threshold=vad_threshold
                                        )
                                        print("✓ 增强 VAC 处理器重新初始化成功")
                                    except ImportError:
                                        online = DynamicVACOnlineASRProcessor(
                                            vac_chunk_size,
                                            asr,
                                            tokenizer=None,
                                            logfile=filtered_logfile,
                                            buffer_trimming=("segment", 15),
                                            initial_silence_ms=vad_initial_silence_ms,
                                            min_silence_ms=vad_min_silence_ms,
                                            max_silence_ms=vad_max_silence_ms,
                                            vad_threshold=vad_threshold
                                        )
                                        print("✓ VAC 处理器重新初始化成功")
                                else:
                                    try:
                                        from enhanced_asr_processor import EnhancedOnlineASRProcessor
                                        online = EnhancedOnlineASRProcessor(
                                            asr=asr,
                                            tokenizer=None,
                                            buffer_trimming=("segment", 15),
                                            logfile=sys.stderr,
                                            agreement_n=2,
                                            enable_dynamic_buffer=True
                                        )
                                        print("✓ 增强处理器重新初始化成功")
                                    except ImportError:
                                        online = OnlineASRProcessor(asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=sys.stderr)
                                        print("✓ 普通处理器重新初始化成功")
                                
                                print(f"✓ 语言已更改为: {input_language} ({'自动检测' if input_language == 'auto' else input_language})")
                            else:
                                print(f"语言未改变，仍为: {input_language}")
                        else:
                            print(f"⚠ 未识别的语言代码 '{lang_choice}'，保持当前语言: {input_language}")
                    else:
                        print(f"保持当前语言: {input_language}")
                except KeyboardInterrupt:
                    print("\n保持当前语言")
                print()
            
            print("按 Ctrl+C 停止当前会话")
            print("-" * 60)
            print()
            
            # 执行录音会话
            # 根据模型大小调整处理间隔
            if model_size in ['large-v2', 'large-v3', 'large']:
                actual_chunk_duration = max(CHUNK_DURATION, 2.0)  # 大模型至少 2 秒
            else:
                actual_chunk_duration = CHUNK_DURATION
            
            success = record_session(
                online, 
                stream,  # 传入已打开的音频流
                device_idx=device_idx,
                model_size=model_size,
                chunk_duration=actual_chunk_duration,
                use_vac=use_vac,
                config_manager=config_manager,  # 传递配置管理器
                perf_display=perf_display,  # 传递性能显示器
                device_protector=device_protector,  # 传递设备保护器
                input_language=input_language,  # 传递当前语言
                translation_manager=translation_manager  # 传递翻译管理器
            )
            
            if not success:
                break
            
            # 询问是否继续
            print()
            try:
                choice = input("是否继续录音？(y/n，直接回车继续): ").strip().lower()
                if choice in ['n', 'no', '退出', 'exit', 'quit']:
                    print("退出程序")
                    break
                # 其他情况（包括直接回车）都继续
            except KeyboardInterrupt:
                print("\n退出程序")
                break
    finally:
        # 程序退出时关闭音频流
        print("\n正在关闭音频流...")
        try:
            if device_protector is not None:
                device_protector.close()
            else:
                if stream is not None:
                    stream.stop()
                    stream.close()
            print("✓ 音频流已关闭")
        except:
            pass
        
        # 停止翻译管理器（如果启用）
        if translation_manager is not None:
            try:
                translation_manager.stop()
                stats = translation_manager.get_stats()
                print(f"\n翻译统计: 添加={stats['total_added']}, 翻译={stats['total_translated']}, 失败={stats['total_failed']}, 重试={stats['total_retried']}")
            except:
                pass

if __name__ == "__main__":
    main()

