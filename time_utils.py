#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间处理工具模块
提供统一的时间单位转换和验证功能
"""

from typing import Tuple


def ms_to_seconds(ms: int) -> float:
    """
    毫秒转秒
    
    Args:
        ms: 毫秒数
        
    Returns:
        秒数（浮点数）
    """
    return ms / 1000.0


def seconds_to_ms(seconds: float) -> int:
    """
    秒转毫秒
    
    Args:
        seconds: 秒数（浮点数）
        
    Returns:
        毫秒数（整数）
    """
    return int(seconds * 1000)


def validate_timestamps(beg_time: float, end_time: float) -> Tuple[bool, float]:
    """
    验证时间戳并计算时长
    
    Args:
        beg_time: 开始时间（秒）
        end_time: 结束时间（秒）
        
    Returns:
        (是否有效, 时长（秒）)
    """
    if end_time <= beg_time:
        return False, 0.0
    return True, end_time - beg_time


def calculate_audio_duration(beg_time: float, end_time: float) -> float:
    """
    计算音频时长（秒）
    
    Args:
        beg_time: 开始时间（秒）
        end_time: 结束时间（秒）
        
    Returns:
        音频时长（秒），如果无效则返回 0.0
    """
    is_valid, duration = validate_timestamps(beg_time, end_time)
    return duration


def format_duration(seconds: float) -> str:
    """
    格式化时长显示
    
    Args:
        seconds: 时长（秒）
        
    Returns:
        格式化后的字符串
    """
    if seconds < 0:
        return "0.00s"
    elif seconds < 1.0:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60.0:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        if minutes < 60:
            return f"{minutes}m{secs:.1f}s"
        else:
            hours = int(minutes // 60)
            mins = minutes % 60
            return f"{hours}h{mins}m{secs:.1f}s"


def clamp_duration(duration: float, min_duration: float = 0.0, max_duration: float = 3600.0) -> float:
    """
    限制时长在合理范围内
    
    Args:
        duration: 时长（秒）
        min_duration: 最小时长（秒）
        max_duration: 最大时长（秒）
        
    Returns:
        限制后的时长（秒）
    """
    return max(min_duration, min(max_duration, duration))

