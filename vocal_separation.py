#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人声分离模块
用于从音频中分离人声和背景音乐，提高语音识别准确率

支持的方案：
1. Demucs（推荐）：高质量，支持实时，有轻量级模型
2. Spleeter：成熟稳定，但主要用于离线处理
3. 简单频域滤波：快速但效果有限
"""

import numpy as np
from typing import Optional, Tuple
import warnings


class VocalSeparator:
    """人声分离器基类"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Args:
            sample_rate: 采样率（Hz）
        """
        self.sample_rate = sample_rate
        self.enabled = False
    
    def separate(self, audio: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        分离人声和背景音乐
        
        Args:
            audio: 输入音频（numpy数组，float32，范围-1.0到1.0）
            
        Returns:
            (vocal_audio, background_audio): 人声音频和背景音频
            background_audio 可能为 None（如果不需要）
        """
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """检查是否可用（依赖是否安装）"""
        return False


class DemucsSeparator(VocalSeparator):
    """
    Demucs 人声分离器（推荐）
    
    优点：
    - 高质量分离
    - 支持实时处理
    - 有轻量级模型（htdemucs）
    
    安装：
    pip install demucs
    
    参考：https://github.com/facebookresearch/demucs
    """
    
    def __init__(self, sample_rate: int = 16000, model_name: str = "htdemucs", model_path: Optional[str] = None):
        """
        Args:
            sample_rate: 采样率（Hz）
            model_name: 模型名称
                - "htdemucs": 轻量级，适合实时（推荐）
                - "htdemucs_ft": 更高质量但更慢
                - "mdx": 最高质量但最慢
            model_path: 模型路径（可选），如果指定则从此路径加载模型
        """
        super().__init__(sample_rate)
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.device = None
        self._try_load_model()
    
    def _try_load_model(self):
        """尝试加载模型
        
        流程：
        1. 如果指定了模型路径，先检查该路径下是否有模型
        2. 如果找到，直接加载
        3. 如果没找到，从默认路径下载，然后复制到指定路径
        4. 如果没指定路径，使用默认路径（会自动下载）
        """
        try:
            import torch
            from pathlib import Path
            from demucs.pretrained import get_model
            
            # 检查CUDA是否可用
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 加载模型（参数名是 name，不是 model_name）
            # 如果指定了模型路径，使用 repo 参数
            if self.model_path:
                model_path = Path(self.model_path)
                model_dir = model_path / self.model_name
                if model_dir.exists() and any(model_dir.iterdir()):
                    # 模型已存在，直接加载
                    print(f"✓ 从指定路径加载模型: {self.model_path}/{self.model_name}")
                    self.model = get_model(name=self.model_name, repo=model_path)
                else:
                    # 模型不存在，先使用默认路径下载，然后复制
                    # 如果路径不存在，尝试创建
                    if not model_path.exists():
                        model_path.mkdir(parents=True, exist_ok=True)
                    print(f"⚠ 指定路径下未找到模型 {self.model_name}")
                    print(f"正在下载模型（首次运行需要下载，请稍候...）")
                    try:
                        # 先尝试从默认路径下载
                        # 下载到默认路径（Demucs会自动处理缓存）
                        temp_model = get_model(name=self.model_name)
                        # 如果下载成功，使用下载的模型
                        self.model = temp_model
                        print(f"✓ 模型已下载到默认路径，正在使用...")
                    except Exception as download_error:
                        # 下载失败，尝试从指定路径加载（可能用户已手动下载）
                        print(f"⚠ 从默认路径下载失败: {download_error}")
                        print(f"尝试使用指定路径（可能需要手动下载模型）...")
                        try:
                            self.model = get_model(name=self.model_name, repo=model_path)
                        except Exception as load_error:
                            raise Exception(f"无法从指定路径加载模型: {load_error}")
            else:
                # 未指定路径，使用默认路径（会自动下载）
                self.model = get_model(name=self.model_name)
            
            if self.model is None:
                raise Exception("模型加载失败：self.model 为 None")
            
            self.model.to(self.device)
            self.model.eval()
            
            self.enabled = True
            print(f"✓ Demucs 模型加载成功: {self.model_name} ({self.device})")
            if self.model_path:
                print(f"  模型路径: {self.model_path}")
        except ImportError:
            self.enabled = False
            print("⚠ Demucs 未安装，跳过人声分离")
            print("  安装: pip install demucs")
        except Exception as e:
            self.enabled = False
            error_msg = str(e)
            print(f"⚠ Demucs 模型加载失败: {error_msg}")
            if "neither a single pre-trained model" in error_msg.lower():
                print(f"  可能原因：模型 {self.model_name} 未下载或路径配置错误")
                print(f"  解决方案：")
                print(f"  1. 检查网络连接，确保可以下载模型")
                print(f"  2. 如果使用自定义路径，确保路径正确且可写")
                print(f"  3. 尝试将 demucs_model_path 设置为空字符串，使用默认路径")
                print(f"  4. 手动下载模型：python -c \"from demucs.pretrained import get_model; get_model('{self.model_name}')\"")
    
    def separate(self, audio: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """使用 Demucs 分离人声"""
        if not self.enabled or self.model is None:
            return audio, None
        
        try:
            import torch
            
            # 确保音频是单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # 转换为torch tensor（需要是2D: [channels, samples]）
            # Demucs期望输入是 [1, samples] 或 [2, samples]
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            
            # 分离（尝试不同的API）
            with torch.no_grad():
                try:
                    # 新版本API：直接调用separate方法
                    sources = self.model.separate(audio_tensor)
                except AttributeError:
                    # 如果模型没有separate方法，尝试使用apply_model
                    try:
                        sources = self.model.apply_model(audio_tensor)
                    except:
                        # 如果还是失败，尝试使用separate_tensor
                        from demucs import separate
                        sources = separate(self.model, audio_tensor)
            
            # 处理返回结果
            # Demucs返回格式可能是：
            # - [batch, sources, channels, samples] 或
            # - [sources, channels, samples]
            if isinstance(sources, (list, tuple)):
                sources = torch.stack(sources)
            
            # 确保是4D tensor: [batch, sources, channels, samples]
            if len(sources.shape) == 3:
                sources = sources.unsqueeze(0)
            
            # 提取人声（vocals是第4个，索引3）
            # 如果只有2个源（vocals和accompaniment），索引是1
            if sources.shape[1] >= 4:
                vocal_audio = sources[0, 3, 0].cpu().numpy()  # [batch, source, channel, samples]
                # 提取背景音乐（drums + bass + other）
                background_audio = (sources[0, 0, 0] + sources[0, 1, 0] + sources[0, 2, 0]).cpu().numpy()
            elif sources.shape[1] == 2:
                # 2-stems模型：vocals和accompaniment
                vocal_audio = sources[0, 0, 0].cpu().numpy()
                background_audio = sources[0, 1, 0].cpu().numpy()
            else:
                # 未知格式，返回原始音频
                warnings.warn(f"Demucs返回格式未知: {sources.shape}，返回原始音频")
                return audio, None
            
            return vocal_audio, background_audio
            
        except Exception as e:
            warnings.warn(f"Demucs分离失败: {e}，返回原始音频")
            import traceback
            traceback.print_exc()
            return audio, None
    
    def is_available(self) -> bool:
        return self.enabled


class SpleeterSeparator(VocalSeparator):
    """
    Spleeter 人声分离器
    
    优点：
    - 成熟稳定
    - 预训练模型质量好
    
    缺点：
    - 主要用于离线处理
    - 实时性能较差
    
    安装：
    pip install spleeter
    
    参考：https://github.com/deezer/spleeter
    """
    
    def __init__(self, sample_rate: int = 16000, model_type: str = "2stems"):
        """
        Args:
            sample_rate: 采样率（Hz）
            model_type: 模型类型
                - "2stems": 人声和伴奏（推荐，最快）
                - "4stems": 人声、鼓、贝斯、其他
                - "5stems": 人声、鼓、贝斯、钢琴、其他
        """
        super().__init__(sample_rate)
        self.model_type = model_type
        self.separator = None
        self._try_load_model()
    
    def _try_load_model(self):
        """尝试加载模型"""
        try:
            from spleeter.separator import Separator
            
            self.separator = Separator(f'spleeter:{self.model_type}')
            self.enabled = True
            print(f"✓ Spleeter 模型加载成功: {self.model_type}")
        except ImportError:
            self.enabled = False
            print("⚠ Spleeter 未安装，跳过人声分离")
            print("  安装: pip install spleeter")
        except Exception as e:
            self.enabled = False
            print(f"⚠ Spleeter 模型加载失败: {e}")
    
    def separate(self, audio: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """使用 Spleeter 分离人声"""
        if not self.enabled or self.separator is None:
            return audio, None
        
        try:
            # 确保音频是单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Spleeter需要立体声，转换为2声道
            audio_stereo = np.stack([audio, audio], axis=1)
            
            # 分离
            prediction = self.separator.separate(audio_stereo)
            
            # 提取人声和伴奏
            vocal_audio = prediction['vocals'][:, 0]  # 取左声道
            background_audio = prediction['accompaniment'][:, 0]  # 取左声道
            
            return vocal_audio, background_audio
            
        except Exception as e:
            warnings.warn(f"Spleeter分离失败: {e}，返回原始音频")
            return audio, None
    
    def is_available(self) -> bool:
        return self.enabled


class SimpleFilterSeparator(VocalSeparator):
    """
    简单频域滤波分离器
    
    优点：
    - 无需额外依赖
    - 实时性能好
    - 计算开销小
    
    缺点：
    - 效果有限
    - 可能误过滤人声
    
    原理：
    - 人声主要在 85-255 Hz（基频）和 300-3400 Hz（共振峰）
    - 音乐主要在低频（< 200 Hz）和高频（> 4000 Hz）
    - 使用带通滤波器保留人声频段
    """
    
    def __init__(self, sample_rate: int = 16000, 
                 low_cut: float = 85.0, high_cut: float = 3400.0):
        """
        Args:
            sample_rate: 采样率（Hz）
            low_cut: 低截止频率（Hz），保留高于此频率的信号
            high_cut: 高截止频率（Hz），保留低于此频率的信号
        """
        super().__init__(sample_rate)
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.enabled = True
    
    def separate(self, audio: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """使用频域滤波分离人声"""
        try:
            from scipy import signal
            
            # 确保音频是单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # 设计带通滤波器（保留人声频段）
            nyquist = self.sample_rate / 2
            low = self.low_cut / nyquist
            high = self.high_cut / nyquist
            
            # 使用Butterworth滤波器
            b, a = signal.butter(4, [low, high], btype='band')
            
            # 应用滤波器
            vocal_audio = signal.filtfilt(b, a, audio)
            
            # 背景音乐 = 原始音频 - 人声（简化处理）
            background_audio = audio - vocal_audio
            
            return vocal_audio, background_audio
            
        except ImportError:
            warnings.warn("scipy 未安装，无法使用频域滤波，返回原始音频")
            return audio, None
        except Exception as e:
            warnings.warn(f"频域滤波失败: {e}，返回原始音频")
            return audio, None
    
    def is_available(self) -> bool:
        try:
            import scipy
            return True
        except ImportError:
            return False


def create_separator(method: str = "demucs", sample_rate: int = 16000, **kwargs) -> VocalSeparator:
    """
    创建人声分离器
    
    Args:
        method: 分离方法
            - "demucs": 使用 Demucs（推荐）
            - "spleeter": 使用 Spleeter
            - "filter": 使用简单频域滤波
            - "none": 不使用分离
        sample_rate: 采样率（Hz）
        **kwargs: 其他参数（传递给具体的分离器）
        
    Returns:
        VocalSeparator 实例
    """
    if method.lower() == "none" or method.lower() == "off":
        return VocalSeparator(sample_rate)
    
    elif method.lower() == "demucs":
        model_name = kwargs.get("model_name", "htdemucs")
        model_path = kwargs.get("model_path", None)
        return DemucsSeparator(sample_rate, model_name, model_path)
    
    elif method.lower() == "spleeter":
        model_type = kwargs.get("model_type", "2stems")
        return SpleeterSeparator(sample_rate, model_type)
    
    elif method.lower() == "filter":
        low_cut = kwargs.get("low_cut", 85.0)
        high_cut = kwargs.get("high_cut", 3400.0)
        return SimpleFilterSeparator(sample_rate, low_cut, high_cut)
    
    else:
        raise ValueError(f"未知的分离方法: {method}，支持: demucs, spleeter, filter, none")


# 使用示例
if __name__ == "__main__":
    # 测试代码
    import numpy as np
    
    # 创建测试音频（1秒，16kHz）
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波
    
    # 测试 Demucs
    print("测试 Demucs...")
    separator = create_separator("demucs", sample_rate)
    if separator.is_available():
        vocal, bg = separator.separate(test_audio)
        print(f"  人声形状: {vocal.shape}, 背景形状: {bg.shape if bg is not None else None}")
    
    # 测试简单滤波
    print("测试简单滤波...")
    separator = create_separator("filter", sample_rate)
    if separator.is_available():
        vocal, bg = separator.separate(test_audio)
        print(f"  人声形状: {vocal.shape}, 背景形状: {bg.shape if bg is not None else None}")

