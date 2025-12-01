#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻译管理器模块
负责定期翻译未翻译的识别结果
"""

import requests
import json
import time
import threading
import os
from typing import Dict, List, Optional, Callable
from collections import deque
from datetime import datetime


class TranslationManager:
    """翻译管理器，定期翻译未翻译的识别结果"""
    
    # 翻译API配置
    MODEL_CONFIG = {
        "vendor_name": "siliconflow",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": os.getenv("SILICONFLOW_API_KEY", ""),  # 从环境变量读取，如果不存在则使用空字符串
        "model": "tencent/Hunyuan-MT-7B"
    }
    
    TRANSLATE_PROMPT_TEMPLATE = "请将以下文本精准、流畅地翻译成中文，保持原文的语气和专业度，不要增减语义：\n{text}"
    TIMEOUT_SECONDS = 90
    
    def __init__(self, translate_interval: float = 10.0, 
                 output_callback: Optional[Callable[[str, str], None]] = None):
        """
        初始化翻译管理器
        
        Args:
            translate_interval: 翻译间隔（秒）
            output_callback: 翻译结果输出回调函数，参数为(原文, 翻译)
        """
        self.translate_interval = translate_interval
        self.output_callback = output_callback
        
        # 未翻译的文本队列（线程安全）
        self._lock = threading.Lock()
        self._pending_texts = deque()  # [(text, timestamp, retry_count), ...]
        self._translated_texts = set()  # 已翻译的文本（用于去重）
        
        # 控制标志
        self._running = False
        self._thread = None
        
        # 统计信息
        self._stats = {
            "total_added": 0,
            "total_translated": 0,
            "total_failed": 0,
            "total_retried": 0
        }
    
    def add_text(self, text: str):
        """
        添加待翻译的文本
        
        Args:
            text: 待翻译的文本
        """
        if not text or not text.strip():
            return
        
        text_clean = text.strip()
        
        with self._lock:
            # 检查是否已翻译过（去重）
            if text_clean in self._translated_texts:
                return
            
            # 检查是否已在待翻译队列中
            for pending_text, _, _ in self._pending_texts:
                if pending_text == text_clean:
                    return
            
            # 添加到待翻译队列
            self._pending_texts.append((text_clean, time.time(), 0))
            self._stats["total_added"] += 1
    
    def _translate_text(self, text: str) -> Dict:
        """
        调用翻译API翻译文本
        
        Args:
            text: 待翻译的文本
            
        Returns:
            包含翻译结果的字典
        """
        url = self.MODEL_CONFIG["url"]
        api_key = self.MODEL_CONFIG["api_key"]
        model = self.MODEL_CONFIG["model"]
        
        # 检查 API key 是否配置
        if not api_key:
            return {
                "status": "failed",
                "error": "API key 未配置，请设置环境变量 SILICONFLOW_API_KEY",
                "elapsed": 0
            }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        prompt = self.TRANSLATE_PROMPT_TEMPLATE.format(text=text)
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(data),
                timeout=self.TIMEOUT_SECONDS
            )
            response.raise_for_status()
            elapsed = time.time() - start_time
            
            resp_json = response.json()
            translation = resp_json["choices"][0]["message"]["content"].strip()
            
            return {
                "status": "success",
                "translation": translation,
                "elapsed": elapsed
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "status": "failed",
                "error": str(e),
                "elapsed": elapsed
            }
    
    def _process_pending_texts(self):
        """处理待翻译的文本"""
        texts_to_translate = []
        texts_to_retry = []
        
        with self._lock:
            # 收集待翻译的文本（合并失败重试的）
            current_time = time.time()
            remaining_texts = deque()
            
            while self._pending_texts:
                text, timestamp, retry_count = self._pending_texts.popleft()
                
                # 如果失败过（retry_count > 0），且还未超过重试次数（retry_count == 0表示第一次失败，可以重试一次）
                # retry_count == 0: 新文本，未失败过
                # retry_count == 1: 失败过一次，可以跟随重发一次
                # retry_count >= 2: 已超过重试次数，放弃
                if retry_count == 1:
                    # 失败过一次，可以跟随重发一次
                    texts_to_retry.append((text, retry_count))
                elif retry_count >= 2:
                    # 已超过重试次数，放弃
                    self._stats["total_failed"] += 1
                    print(f"⚠ 翻译失败，已超过重试次数: {text[:50]}...")
                else:
                    # 新文本（retry_count == 0），加入待翻译列表
                    remaining_texts.append((text, timestamp, retry_count))
            
            # 合并重试文本到待翻译列表（最多跟随重发一次）
            if texts_to_retry:
                # 将重试文本合并到新文本中一起发送
                retry_texts = [text for text, _ in texts_to_retry]
                if remaining_texts:
                    # 如果有新文本，将重试文本合并到第一个新文本中
                    first_text, first_timestamp, first_retry = remaining_texts[0]
                    combined_text = "\n".join([first_text] + retry_texts)
                    remaining_texts[0] = (combined_text, first_timestamp, 0)
                    self._stats["total_retried"] += len(retry_texts)
                else:
                    # 如果没有新文本，将重试文本合并成一个文本发送
                    if retry_texts:
                        combined_retry_text = "\n".join(retry_texts)
                        remaining_texts.append((combined_retry_text, current_time, 1))
                        self._stats["total_retried"] += len(retry_texts)
            
            # 收集要翻译的文本
            while remaining_texts:
                text, _, retry_count = remaining_texts.popleft()
                texts_to_translate.append((text, retry_count))
            
            # 清空待翻译队列
            self._pending_texts = remaining_texts
        
        # 批量翻译（合并多个文本）
        if texts_to_translate:
            # 合并所有文本，用换行分隔
            combined_text = "\n".join([text for text, _ in texts_to_translate])
            
            result = self._translate_text(combined_text)
            
            with self._lock:
                if result["status"] == "success":
                    translation = result["translation"]
                    
                    # 标记所有文本为已翻译
                    for text, retry_count in texts_to_translate:
                        self._translated_texts.add(text)
                        self._stats["total_translated"] += 1
                    
                    # 调用输出回调
                    if self.output_callback:
                        # 如果合并了多个文本，翻译结果可能也是多行
                        # 简单处理：如果翻译结果包含换行，按行分割；否则整体显示
                        if "\n" in translation:
                            translations = translation.split("\n")
                            for i, (text, _) in enumerate(texts_to_translate):
                                if i < len(translations):
                                    self.output_callback(text, translations[i].strip())
                                else:
                                    self.output_callback(text, translation)
                        else:
                            # 单个翻译结果，为每个原文显示
                            for text, _ in texts_to_translate:
                                self.output_callback(text, translation)
                else:
                    # 翻译失败，标记为需要重试（最多重试一次）
                    for text, retry_count in texts_to_translate:
                        if retry_count == 0:
                            # 第一次失败，加入重试队列（跟随下一次内容一起发送）
                            with self._lock:
                                self._pending_texts.append((text, time.time(), 1))
                            self._stats["total_failed"] += 1
                        elif retry_count == 1:
                            # 已经重试过一次，超过重试次数，放弃
                            self._stats["total_failed"] += 1
                            print(f"⚠ 翻译失败，已超过重试次数: {text[:50]}...")
                        else:
                            # 其他情况，放弃
                            self._stats["total_failed"] += 1
    
    def _translation_worker(self):
        """翻译工作线程"""
        while self._running:
            try:
                self._process_pending_texts()
            except Exception as e:
                print(f"⚠ 翻译处理错误: {e}")
            
            # 等待指定间隔
            time.sleep(self.translate_interval)
    
    def start(self):
        """启动翻译管理器"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._translation_worker, daemon=True)
        self._thread.start()
    
    def stop(self):
        """停止翻译管理器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self._lock:
            return {
                "total_added": self._stats["total_added"],
                "total_translated": self._stats["total_translated"],
                "total_failed": self._stats["total_failed"],
                "total_retried": self._stats["total_retried"],
                "pending_count": len(self._pending_texts)
            }

