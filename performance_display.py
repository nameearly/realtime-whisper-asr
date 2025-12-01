#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控和实时显示模块
提供跳句统计、设备状态等信息的实时显示
"""

import time
import sys
from typing import Dict, Optional
from datetime import datetime


class PerformanceDisplay:
    """性能监控和实时显示"""
    
    # ANSI 颜色代码
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m'
    }
    
    def __init__(self, enable_colors: bool = True, update_interval: float = 5.0):
        """
        Args:
            enable_colors: 是否启用彩色输出
            update_interval: 更新间隔（秒）
        """
        self.enable_colors = enable_colors and sys.stdout.isatty()
        self.update_interval = update_interval
        self.last_update_time = 0
        self.skip_detector = None
        self.audio_deduplicator = None
        self.device_protector = None
        self.start_time = time.time()
        self.last_stats = {}
    
    def set_skip_detector(self, skip_detector):
        """设置跳句检测器（用于获取统计信息）"""
        self.skip_detector = skip_detector
    
    def set_audio_deduplicator(self, audio_deduplicator):
        """设置音频去重器（用于获取统计信息）"""
        self.audio_deduplicator = audio_deduplicator
    
    def set_device_protector(self, device_protector):
        """设置设备保护器（用于获取状态信息）"""
        self.device_protector = device_protector
    
    def _colorize(self, text: str, color: str) -> str:
        """添加颜色"""
        if not self.enable_colors:
            return text
        color_code = self.COLORS.get(color, '')
        return f"{color_code}{text}{self.COLORS['reset']}"
    
    def _format_duration(self, seconds: float) -> str:
        """格式化时长"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}分{secs}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}小时{minutes}分钟"
    
    def _clear_line(self):
        """清除当前行"""
        print("\r" + " " * 100 + "\r", end='', flush=True)
    
    def display_stats(self, force: bool = False):
        """
        显示统计信息
        
        Args:
            force: 是否强制更新（忽略时间间隔）
        """
        current_time = time.time()
        
        # 检查是否需要更新
        if not force and (current_time - self.last_update_time) < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # 收集统计信息
        stats_lines = []
        
        # 运行时间
        runtime = current_time - self.start_time
        stats_lines.append(f"运行时间: {self._colorize(self._format_duration(runtime), 'cyan')}")
        
        # 跳句统计
        if self.skip_detector:
            skip_stats = self.skip_detector.get_stats()
            total_checked = skip_stats.get('total_checked', 0)
            total_skipped = skip_stats.get('total_skipped', 0)
            skip_rate = skip_stats.get('skip_rate', 0)
            
            if total_checked > 0:
                stats_lines.append(f"检查: {self._colorize(str(total_checked), 'green')} | 跳过: {self._colorize(str(total_skipped), 'yellow')} ({skip_rate:.1f}%)")
        
        # 音频去重统计
        if self.audio_deduplicator:
            audio_stats = self.audio_deduplicator.get_stats()
            total_checked = audio_stats.get('total_checked', 0)
            skipped_duplicate = audio_stats.get('skipped_duplicate', 0)
            skipped_similar = audio_stats.get('skipped_similar', 0)
            total_skipped = skipped_duplicate + skipped_similar
            audio_time_skipped = audio_stats.get('total_audio_time_skipped', 0.0)
            
            if total_checked > 0:
                skip_rate = (total_skipped / total_checked * 100) if total_checked > 0 else 0
                audio_info = f"音频去重: {self._colorize(str(total_checked), 'green')} | 跳过: {self._colorize(str(total_skipped), 'yellow')} ({skip_rate:.1f}%)"
                if audio_time_skipped > 0:
                    audio_info += f" | 节省: {self._colorize(f'{audio_time_skipped:.1f}s', 'cyan')}"
                stats_lines.append(audio_info)
        
        # 设备状态
        if self.device_protector:
            device_status = self.device_protector.get_status()
            is_healthy = device_status.get('is_healthy', False)
            is_streaming = device_status.get('is_streaming', False)
            recovery_count = device_status.get('recovery_count', 0)
            device_name = device_status.get('device_name', '未知')
            
            # 设备状态颜色
            if is_healthy and is_streaming:
                status_color = 'green'
                status_text = '正常'
            elif is_streaming:
                status_color = 'yellow'
                status_text = '警告'
            else:
                status_color = 'red'
                status_text = '断开'
            
            device_display = f"设备: {self._colorize(status_text, status_color)}"
            if recovery_count > 0:
                device_display += f" (恢复:{recovery_count})"
            stats_lines.append(device_display)
        
        # 显示统计信息
        if stats_lines:
            self._clear_line()
            # 优化显示格式：使用更清晰的分隔符和布局
            stats_text = "  │  ".join(stats_lines)
            print(f"\r📊 {stats_text}", end='', flush=True)
    
    def display_device_status(self, force: bool = False):
        """
        显示设备状态（独立显示）
        
        Args:
            force: 是否强制更新
        """
        if not self.device_protector:
            return
        
        current_time = time.time()
        if not force and (current_time - self.last_update_time) < self.update_interval:
            return
        
        device_status = self.device_protector.get_status()
        is_healthy = device_status.get('is_healthy', False)
        is_streaming = device_status.get('is_streaming', False)
        device_name = device_status.get('device_name', '未知')
        
        if is_healthy and is_streaming:
            status_icon = "✓"
            status_color = 'green'
            status_text = '正常'
        elif is_streaming:
            status_icon = "⚠"
            status_color = 'yellow'
            status_text = '警告'
        else:
            status_icon = "✗"
            status_color = 'red'
            status_text = '断开'
        
        status_display = f"{status_icon} 设备: {self._colorize(status_text, status_color)} ({device_name})"
        print(f"\r{status_display}", end='', flush=True)
    
    def display_error(self, error_type: str, message: str, suggestion: Optional[str] = None):
        """
        显示友好的错误信息
        
        Args:
            error_type: 错误类型
            message: 错误消息
            suggestion: 建议（可选）
        """
        error_icon = "✗"
        error_color = 'red'
        
        print(f"\n{self._colorize(f'{error_icon} 错误', error_color)}: {error_type}")
        print(f"  {message}")
        
        if suggestion:
            print(f"  {self._colorize('💡 建议', 'yellow')}: {suggestion}")
    
    def display_warning(self, message: str):
        """显示警告信息"""
        warning_icon = "⚠"
        warning_color = 'yellow'
        print(f"{self._colorize(f'{warning_icon} 警告', warning_color)}: {message}")
    
    def display_success(self, message: str):
        """显示成功信息"""
        success_icon = "✓"
        success_color = 'green'
        print(f"{self._colorize(f'{success_icon} 成功', success_color)}: {message}")
    
    def display_info(self, message: str):
        """显示信息"""
        info_icon = "ℹ"
        info_color = 'cyan'
        print(f"{self._colorize(f'{info_icon} 信息', info_color)}: {message}")
    
    def display_progress(self, message: str, end: str = '\r'):
        """显示进度信息"""
        print(f"⏳ {message}", end=end, flush=True)
    
    def clear(self):
        """清除显示"""
        self._clear_line()
    
    def newline(self):
        """换行"""
        print()

