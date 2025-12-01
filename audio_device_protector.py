#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频设备保护模块
提供设备占用检测、重试机制和自动恢复功能
"""

import time
import sounddevice as sd
import numpy as np
from typing import Optional, Dict, List, Tuple


class AudioDeviceProtector:
    """音频设备保护器，提供设备占用检测和自动恢复功能"""
    
    def __init__(self, max_retries=3, retry_delay=1.0, check_interval=0.5):
        """
        Args:
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            check_interval: 设备检查间隔（秒）
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.check_interval = check_interval
        self.stream = None
        self.device_index = None
        self.device_name = None
        self.is_streaming = False
        self.last_error = None
        self.recovery_count = 0
    
    def check_device_available(self, device_index: int) -> Tuple[bool, Optional[str]]:
        """
        检查设备是否可用
        
        Args:
            device_index: 设备索引
            
        Returns:
            (是否可用, 错误信息)
        """
        try:
            # 尝试查询设备信息
            device_info = sd.query_devices(device_index)
            if device_info['max_input_channels'] == 0:
                return False, "设备不支持输入"
            
            # 尝试打开一个测试流（短暂打开后立即关闭）
            test_stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype='float32',
                blocksize=512,
                device=device_index
            )
            test_stream.start()
            time.sleep(0.1)  # 短暂等待
            test_stream.stop()
            test_stream.close()
            
            return True, None
        except sd.PortAudioError as e:
            error_msg = str(e)
            if "Invalid device" in error_msg or "device unavailable" in error_msg.lower():
                return False, f"设备不可用: {error_msg}"
            elif "device unavailable" in error_msg.lower() or "busy" in error_msg.lower():
                return False, f"设备被占用: {error_msg}"
            else:
                return False, f"设备错误: {error_msg}"
        except Exception as e:
            return False, f"未知错误: {str(e)}"
    
    def open_stream(self, device_index: int, samplerate: int = 16000, 
                   channels: int = 1, blocksize: int = 512,
                   dtype: str = 'float32') -> Tuple[bool, Optional[sd.InputStream], Optional[str]]:
        """
        打开音频流（带重试机制）
        
        Args:
            device_index: 设备索引
            samplerate: 采样率
            channels: 声道数
            blocksize: 块大小
            dtype: 数据类型
            
        Returns:
            (是否成功, 流对象, 错误信息)
        """
        self.device_index = device_index
        
        # 获取设备名称
        try:
            device_info = sd.query_devices(device_index)
            self.device_name = device_info['name']
        except:
            self.device_name = f"设备 {device_index}"
        
        # 先检查设备是否可用
        is_available, error_msg = self.check_device_available(device_index)
        if not is_available:
            return False, None, error_msg
        
        # 尝试打开流（带重试）
        last_error = None
        for attempt in range(self.max_retries):
            try:
                stream = sd.InputStream(
                    samplerate=samplerate,
                    channels=channels,
                    dtype=dtype,
                    blocksize=blocksize,
                    device=device_index
                )
                stream.start()
                
                # 验证流是否正常工作（读取一小块数据）
                try:
                    test_data, _ = stream.read(blocksize)
                    if test_data is not None and len(test_data) > 0:
                        self.stream = stream
                        self.is_streaming = True
                        self.last_error = None
                        return True, stream, None
                except Exception as e:
                    stream.stop()
                    stream.close()
                    last_error = f"流验证失败: {str(e)}"
                    continue
                
            except sd.PortAudioError as e:
                error_msg = str(e)
                last_error = error_msg
                
                if "Invalid device" in error_msg:
                    # 设备无效，不需要重试
                    return False, None, f"设备无效: {error_msg}"
                elif "device unavailable" in error_msg.lower() or "busy" in error_msg.lower():
                    # 设备被占用，等待后重试
                    if attempt < self.max_retries - 1:
                        print(f"⚠ 设备被占用，等待 {self.retry_delay} 秒后重试 ({attempt + 1}/{self.max_retries})...")
                        time.sleep(self.retry_delay)
                        # 重新检查设备可用性
                        is_available, check_error = self.check_device_available(device_index)
                        if not is_available:
                            return False, None, check_error
                    else:
                        return False, None, f"设备被占用，已重试 {self.max_retries} 次: {error_msg}"
                else:
                    # 其他错误，也重试
                    if attempt < self.max_retries - 1:
                        print(f"⚠ 打开设备失败，等待 {self.retry_delay} 秒后重试 ({attempt + 1}/{self.max_retries})...")
                        time.sleep(self.retry_delay)
                    else:
                        return False, None, f"打开设备失败，已重试 {self.max_retries} 次: {error_msg}"
            
            except Exception as e:
                last_error = f"未知错误: {str(e)}"
                if attempt < self.max_retries - 1:
                    print(f"⚠ 打开设备时发生错误，等待 {self.retry_delay} 秒后重试 ({attempt + 1}/{self.max_retries})...")
                    time.sleep(self.retry_delay)
                else:
                    return False, None, last_error
        
        return False, None, last_error or "打开设备失败"
    
    def check_stream_health(self) -> Tuple[bool, Optional[str]]:
        """
        检查流的健康状态
        
        Returns:
            (是否健康, 错误信息)
        """
        if self.stream is None:
            return False, "流未打开"
        
        if not self.is_streaming:
            return False, "流未运行"
        
        try:
            # 尝试读取一小块数据（非阻塞）
            # 注意：read() 默认是阻塞的，但我们只是检查流是否有效
            # 如果流已关闭或出错，会抛出异常
            if not self.stream.active:
                return False, "流未激活"
            
            return True, None
        except Exception as e:
            return False, f"流健康检查失败: {str(e)}"
    
    def recover_stream(self, samplerate: int = 16000, channels: int = 1, 
                      blocksize: int = 512, dtype: str = 'float32') -> Tuple[bool, Optional[sd.InputStream], Optional[str]]:
        """
        尝试恢复流
        
        Args:
            samplerate: 采样率
            channels: 声道数
            blocksize: 块大小
            dtype: 数据类型
            
        Returns:
            (是否成功, 流对象, 错误信息)
        """
        if self.device_index is None:
            return False, None, "未指定设备索引"
        
        # 先关闭旧流（如果存在）
        if self.stream is not None:
            try:
                if self.is_streaming:
                    self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None
            self.is_streaming = False
        
        # 等待一小段时间，让设备释放
        time.sleep(0.5)
        
        # 重新打开流
        self.recovery_count += 1
        print(f"🔄 尝试恢复音频流 (第 {self.recovery_count} 次)...")
        success, stream, error = self.open_stream(
            self.device_index, samplerate, channels, blocksize, dtype
        )
        
        if success:
            print(f"✓ 音频流恢复成功")
        else:
            print(f"✗ 音频流恢复失败: {error}")
        
        return success, stream, error
    
    def read_audio(self, frames: int) -> Tuple[Optional[np.ndarray], bool, Optional[str]]:
        """
        从流读取音频数据（带错误处理和自动恢复）
        
        Args:
            frames: 要读取的帧数
            
        Returns:
            (音频数据, 是否溢出, 错误信息)
        """
        if self.stream is None or not self.is_streaming:
            return None, False, "流未打开或未运行"
        
        try:
            audio_data, overflowed = self.stream.read(frames)
            self.last_error = None
            return audio_data, overflowed, None
        
        except sd.PortAudioError as e:
            error_msg = str(e)
            self.last_error = error_msg
            
            # 检查是否是设备被占用或断开
            if "device unavailable" in error_msg.lower() or "busy" in error_msg.lower():
                # 尝试恢复
                success, new_stream, recover_error = self.recover_stream(
                    samplerate=self.stream.samplerate,
                    channels=self.stream.channels,
                    blocksize=self.stream.blocksize,
                    dtype='float32'
                )
                if success:
                    return None, False, "设备已恢复，请重试"
                else:
                    return None, False, f"设备错误且恢复失败: {recover_error}"
            else:
                return None, False, f"读取音频失败: {error_msg}"
        
        except Exception as e:
            self.last_error = str(e)
            return None, False, f"读取音频时发生未知错误: {str(e)}"
    
    def stop(self):
        """停止流"""
        if self.stream is not None and self.is_streaming:
            try:
                self.stream.stop()
                self.is_streaming = False
            except:
                pass
    
    def close(self):
        """关闭流"""
        if self.stream is not None:
            try:
                if self.is_streaming:
                    self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None
            self.is_streaming = False
    
    def get_status(self) -> Dict:
        """获取当前状态"""
        is_healthy, health_error = self.check_stream_health()
        return {
            'device_index': self.device_index,
            'device_name': self.device_name,
            'is_streaming': self.is_streaming,
            'is_healthy': is_healthy,
            'health_error': health_error,
            'last_error': self.last_error,
            'recovery_count': self.recovery_count
        }

