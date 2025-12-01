#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志管理模块
负责记录所有关键信息用于分析
"""

import os
import csv
import json
import logging
import sys
from datetime import datetime


class LogManager:
    """日志管理器，记录所有关键信息用于分析"""
    def __init__(self, log_dir="logs"):
        """
        Args:
            log_dir: 日志文件目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建时间戳用于文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV日志文件（用于数据分析）
        self.csv_log_path = os.path.join(log_dir, f"session_{timestamp}.csv")
        self.csv_file = open(self.csv_log_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        # 写入CSV表头
        self.csv_writer.writerow([
            'timestamp', 'event_type', 'original_text', 'translated_text', 'optimized_text',
            'api_model', 'api_success', 'api_delay', 'skip_reason',
            'success_rate', 'avg_delay', 'pending_count', 'session_id'
        ])
        
        # 文本日志文件（详细日志）
        self.text_log_path = os.path.join(log_dir, f"session_{timestamp}.log")
        
        # 根据配置决定是否输出到控制台
        handlers = [logging.FileHandler(self.text_log_path, encoding='utf-8')]
        console_log_enabled = True  # 默认启用控制台输出
        
        # 尝试从配置管理器获取设置（如果可用）
        try:
            from config_manager import ConfigManager
            config_manager = ConfigManager()
            console_log_enabled = config_manager.get("logging.console_log_enabled", True)
        except:
            pass
        
        if console_log_enabled:
            handlers.append(logging.StreamHandler(sys.stdout))
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        self.logger = logging.getLogger(__name__)
        
        self.session_id = timestamp
        self.logger.info(f"日志系统初始化完成，会话ID: {self.session_id}")
        self.logger.info(f"CSV日志文件: {self.csv_log_path}")
        self.logger.info(f"文本日志文件: {self.text_log_path}")
    
    def log_recognition(self, text, beg_time=None, end_time=None):
        """记录语音识别结果"""
        timestamp = datetime.now().isoformat()
        self.csv_writer.writerow([
            timestamp, 'recognition', text, '', '',
            '', True, 0, '',
            '', '', '', self.session_id
        ])
        self.csv_file.flush()
        self.logger.info(f"[识别] {text} (时间: {beg_time:.2f}-{end_time:.2f}s)" if beg_time else f"[识别] {text}")
    
    def log_translation(self, original_text, translated_text, model_name, success, delay, skip_reason=None, silent=False):
        """记录翻译结果"""
        timestamp = datetime.now().isoformat()
        self.csv_writer.writerow([
            timestamp, 'translation', original_text, translated_text, '',
            model_name, success, delay, skip_reason or '',
            '', '', '', self.session_id
        ])
        self.csv_file.flush()
        if skip_reason:
            # 如果silent=True，只写入文件，不输出到控制台
            if silent:
                # 只写入文件，不输出到控制台
                file_handler = logging.FileHandler(self.text_log_path, encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                temp_logger = logging.getLogger(f"{__name__}_file_only")
                temp_logger.addHandler(file_handler)
                temp_logger.setLevel(logging.INFO)
                temp_logger.info(f"[翻译-跳过] {original_text} | 原因: {skip_reason}")
                temp_logger.removeHandler(file_handler)
            else:
                self.logger.info(f"[翻译-跳过] {original_text} | 原因: {skip_reason}")
        elif success:
            self.logger.info(f"[翻译] {original_text} -> {translated_text} (延迟: {delay:.2f}s)")
        else:
            self.logger.warning(f"[翻译-失败] {original_text} (延迟: {delay:.2f}s)")
    
    def log_optimization(self, translated_text, optimized_text, model_name, success, delay, skip_reason=None, silent=False):
        """记录优化结果"""
        timestamp = datetime.now().isoformat()
        self.csv_writer.writerow([
            timestamp, 'optimization', '', translated_text, optimized_text,
            model_name, success, delay, skip_reason or '',
            '', '', '', self.session_id
        ])
        self.csv_file.flush()
        if skip_reason:
            # 如果silent=True，只写入文件，不输出到控制台
            if silent:
                # 直接写入文件，不通过logger输出到控制台
                with open(self.text_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{timestamp} - INFO - [优化-跳过] {translated_text} | 原因: {skip_reason}\n")
            else:
                self.logger.info(f"[优化-跳过] {translated_text} | 原因: {skip_reason}")
        elif success:
            self.logger.info(f"[优化] {translated_text} -> {optimized_text} (延迟: {delay:.2f}s)")
        else:
            self.logger.warning(f"[优化-失败] {translated_text} (延迟: {delay:.2f}s)")
    
    def log_performance(self, success_rate, avg_delay, pending_count):
        """记录性能指标"""
        timestamp = datetime.now().isoformat()
        self.csv_writer.writerow([
            timestamp, 'performance', '', '', '',
            '', True, avg_delay, '',
            success_rate, avg_delay, pending_count, self.session_id
        ])
        self.csv_file.flush()
        self.logger.debug(f"[性能] 成功率: {success_rate*100:.1f}% | 平均延迟: {avg_delay:.2f}s | 待处理: {pending_count}")
    
    def log_context_update(self, index, old_text, new_text):
        """记录上下文更新"""
        timestamp = datetime.now().isoformat()
        self.logger.info(f"[上下文更新] 第{index+1}句: {old_text} -> {new_text}")
    
    def log_error(self, error_type, error_message, context=None):
        """记录错误"""
        timestamp = datetime.now().isoformat()
        self.logger.error(f"[错误-{error_type}] {error_message}")
        if context:
            self.logger.error(f"上下文: {context}")
    
    def log_config(self, config_dict):
        """记录配置信息"""
        self.logger.info(f"[配置] {json.dumps(config_dict, ensure_ascii=False, indent=2)}")
    
    def close(self):
        """关闭日志文件"""
        self.csv_file.close()
        self.logger.info(f"日志文件已关闭: {self.csv_log_path}")

