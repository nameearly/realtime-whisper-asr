#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的跳句检测模块
使用编辑距离和相似度算法，提供更智能的去重功能
"""

import time
from typing import Optional, Tuple, Dict
from difflib import SequenceMatcher


class ImprovedSkipDetector:
    """改进的跳句检测器，使用编辑距离和相似度算法"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 time_window: float = 3.0,
                 min_length: int = 2,
                 use_edit_distance: bool = True):
        """
        Args:
            similarity_threshold: 相似度阈值（0-1），超过此值认为是重复
            time_window: 时间窗口（秒），只在此时间窗口内进行去重
            min_length: 最小文本长度，短于此长度的文本不进行去重
            use_edit_distance: 是否使用编辑距离算法（更准确但稍慢）
        """
        self.similarity_threshold = similarity_threshold
        self.time_window = time_window
        self.min_length = min_length
        self.use_edit_distance = use_edit_distance
        
        # 历史记录：[(文本, 时间戳), ...]
        self.history = []
        self.last_recognized_text = ""
        self.last_recognized_time = None
        
        # 统计信息
        self.stats = {
            'total_checked': 0,
            'skipped_duplicate': 0,
            'skipped_partial': 0,
            'skipped_similar': 0,
            'skipped_repetition': 0,  # 重复模式
            'skipped_time_window': 0,
            'passed': 0
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度（0-1）
        """
        if not text1 or not text2:
            return 0.0
        
        # 使用 SequenceMatcher（基于最长公共子序列）
        # 这比简单的字符串包含更准确
        similarity = SequenceMatcher(None, text1, text2).ratio()
        return similarity
    
    def _calculate_edit_distance_ratio(self, text1: str, text2: str) -> float:
        """
        计算编辑距离比率（归一化的编辑距离）
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度（0-1），1表示完全相同，0表示完全不同
        """
        if not text1 or not text2:
            return 0.0
        
        # 计算编辑距离
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        
        if max_len == 0:
            return 1.0
        
        # 转换为相似度（0-1）
        similarity = 1.0 - (distance / max_len)
        return similarity
    
    def _detect_repetition_pattern(self, text: str) -> Tuple[bool, str]:
        """
        检测文本中的重复模式（如 "ABCABCABC" 或 "ABC" 重复多次）
        
        Args:
            text: 要检查的文本
            
        Returns:
            (是否检测到重复模式, 重复模式描述)
        """
        if len(text) < 6:  # 太短的文本不检查
            return False, ""
        
        # 检查是否有明显的重复子串
        # 方法：检查文本是否由某个子串重复多次组成
        for pattern_len in range(1, min(len(text) // 2, 20) + 1):  # 检查1-20字符的模式
            pattern = text[:pattern_len]
            # 检查这个模式是否重复多次
            repetitions = len(text) // pattern_len
            if repetitions >= 3:  # 至少重复3次
                expected_text = pattern * repetitions
                if text.startswith(expected_text):
                    return True, f"重复模式: '{pattern}' 重复 {repetitions} 次"
        
        # 检查是否有长重复子串（如 "ABC" 在文本中出现多次）
        # 如果某个子串在文本中出现3次以上，且总长度超过文本的60%，可能是重复
        for check_len in range(3, min(len(text) // 3, 15) + 1):
            for i in range(len(text) - check_len * 2):
                substring = text[i:i+check_len]
                count = text.count(substring)
                if count >= 3 and count * check_len > len(text) * 0.6:
                    return True, f"重复子串: '{substring}' 出现 {count} 次"
        
        return False, ""
    
    def _is_partial_match(self, text1: str, text2: str) -> Tuple[bool, str]:
        """
        检查是否是部分匹配（一个文本是另一个文本的一部分）
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            (是否是部分匹配, 匹配类型)
        """
        if not text1 or not text2:
            return False, ""
        
        # 完全重复
        if text1 == text2:
            return True, "duplicate"
        
        # 检查是否有重复模式（异常文本）
        has_repetition, repetition_info = self._detect_repetition_pattern(text1)
        if has_repetition:
            return True, f"repetition_pattern ({repetition_info})"
        
        has_repetition, repetition_info = self._detect_repetition_pattern(text2)
        if has_repetition:
            return True, f"repetition_pattern ({repetition_info})"
        
        # text1 是 text2 的一部分（且长度明显更短）
        if len(text1) < len(text2) * 0.8 and text1 in text2:
            return True, "partial_contained"
        
        # text2 是 text1 的一部分（且长度明显更短）
        # 注意：如果新文本包含旧文本，但新文本明显更长，可能是新内容，不应该跳过
        if len(text2) < len(text1) * 0.8 and text2 in text1:
            # 如果新文本明显更长（> 旧文本的1.5倍），说明新文本是完整内容，不应该跳过
            # 只有当新文本是异常重复（> 旧文本的2倍）时才跳过
            # 但这里需要更严格的判断：检查新文本是否只是旧文本的简单重复
            if len(text1) > len(text2) * 2:
                # 检查是否是异常重复模式（如 "你好你好你好"）
                # 如果新文本只是旧文本的重复，应该跳过
                # 但如果新文本是旧文本的扩展（如 "你好" -> "你好，今天天气不错"），不应该跳过
                # 判断方法：检查新文本中是否包含明显的扩展内容（如标点、新词等）
                # 简单判断：如果新文本长度 > 旧文本的2倍，且新文本不是简单的重复，可能是扩展内容
                # 为了安全，这里只跳过明显的异常重复，其他情况不跳过
                # 检查是否是简单重复（如 "你好你好你好"）
                repetition_count = text1.count(text2)
                if repetition_count >= 3:  # 重复3次以上，认为是异常重复
                    return True, "partial_contains"
                # 否则，新文本可能是扩展内容，不应该跳过
                return False, ""
        
        return False, ""
    
    def _clean_history(self, current_time: float):
        """清理超出时间窗口的历史记录"""
        cutoff_time = current_time - self.time_window
        self.history = [(text, timestamp) for text, timestamp in self.history 
                       if timestamp > cutoff_time]
    
    def should_skip(self, text: str, current_time: Optional[float] = None) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        判断是否应该跳过此文本
        
        Args:
            text: 要检查的文本
            current_time: 当前时间戳（如果为None则使用当前时间）
            
        Returns:
            (是否跳过, 跳句原因, 详细信息)
        """
        if current_time is None:
            current_time = time.time()
        
        self.stats['total_checked'] += 1
        
        # 清理历史记录
        self._clean_history(current_time)
        
        # 文本预处理
        text_clean = text.strip()
        
        # 如果文本太短，不进行去重
        if len(text_clean) < self.min_length:
            self.stats['passed'] += 1
            return False, None, None
        
        # 检查完全重复
        if text_clean == self.last_recognized_text:
            self.stats['skipped_duplicate'] += 1
            details = {
                'type': 'duplicate',
                'text': text_clean,
                'last_text': self.last_recognized_text,
                'time_since_last': current_time - self.last_recognized_time if self.last_recognized_time else None
            }
            return True, "duplicate", details
        
        # 检查部分匹配
        is_partial, partial_type = self._is_partial_match(text_clean, self.last_recognized_text)
        if is_partial:
            # 特殊情况：如果新文本明显更长（> 旧文本的1.5倍）且包含旧文本，
            # 说明新文本是完整内容，应该保留新文本，而不是跳过
            # 这可以避免"反向跳句"问题（当识别速度快时，先识别到部分内容，后识别到完整内容）
            if (partial_type == "partial_contains" and 
                len(text_clean) > len(self.last_recognized_text) * 1.5 and
                self.last_recognized_text in text_clean):
                # 新文本是旧文本的扩展，应该保留新文本
                # 更新历史记录，使用新文本替换旧文本
                self.last_recognized_text = text_clean
                self.last_recognized_time = current_time
                # 更新历史记录中的旧文本（如果存在）
                for i, (hist_text, hist_time) in enumerate(self.history):
                    if hist_text == self.last_recognized_text:
                        self.history[i] = (text_clean, current_time)
                        break
                else:
                    # 如果历史记录中没有，添加新文本
                    self.history.append((text_clean, current_time))
                self.stats['passed'] += 1
                return False, None, None
            
            # 如果是重复模式，记录为特殊类型
            if "repetition_pattern" in partial_type:
                self.stats['skipped_repetition'] += 1
            elif partial_type == "duplicate":
                self.stats['skipped_duplicate'] += 1
            else:
                self.stats['skipped_partial'] += 1
            details = {
                'type': partial_type,
                'text': text_clean[:100] if len(text_clean) > 100 else text_clean,  # 限制文本长度，避免日志过长
                'last_text': self.last_recognized_text[:100] if len(self.last_recognized_text) > 100 else self.last_recognized_text,
                'time_since_last': current_time - self.last_recognized_time if self.last_recognized_time else None
            }
            return True, partial_type, details
        
        # 检查历史记录中的相似文本（在时间窗口内）
        if self.history:
            for hist_text, hist_time in self.history:
                # 检查时间窗口
                if current_time - hist_time > self.time_window:
                    continue
                
                # 检查部分匹配
                is_partial, partial_type = self._is_partial_match(text_clean, hist_text)
                if is_partial:
                    # 特殊情况：如果新文本明显更长（> 历史文本的1.5倍）且包含历史文本，
                    # 说明新文本是完整内容，应该保留新文本，而不是跳过
                    if (partial_type == "partial_contains" and 
                        len(text_clean) > len(hist_text) * 1.5 and
                        hist_text in text_clean):
                        # 新文本是历史文本的扩展，应该保留新文本
                        # 不跳过，继续处理（会在后面更新历史记录）
                        continue  # 跳过这个历史记录，继续检查其他历史记录
                    
                    # 如果是重复模式，记录为特殊类型
                    if "repetition_pattern" in partial_type:
                        self.stats['skipped_repetition'] += 1
                    else:
                        self.stats['skipped_partial'] += 1
                    details = {
                        'type': partial_type,
                        'text': text_clean[:100] if len(text_clean) > 100 else text_clean,  # 限制文本长度
                        'hist_text': hist_text[:100] if len(hist_text) > 100 else hist_text,
                        'time_since_hist': current_time - hist_time
                    }
                    return True, partial_type, details
                
                # 计算相似度
                if self.use_edit_distance:
                    similarity = self._calculate_edit_distance_ratio(text_clean, hist_text)
                else:
                    similarity = self._calculate_similarity(text_clean, hist_text)
                
                if similarity >= self.similarity_threshold:
                    self.stats['skipped_similar'] += 1
                    details = {
                        'type': 'similar',
                        'text': text_clean[:100] if len(text_clean) > 100 else text_clean,  # 限制文本长度
                        'hist_text': hist_text[:100] if len(hist_text) > 100 else hist_text,
                        'similarity': similarity,
                        'time_since_hist': current_time - hist_time
                    }
                    return True, "similar", details
        
        # 不跳过，更新历史记录
        self.last_recognized_text = text_clean
        self.last_recognized_time = current_time
        self.history.append((text_clean, current_time))
        self.stats['passed'] += 1
        
        return False, None, None
    
    def reset(self):
        """重置检测器状态"""
        self.history = []
        self.last_recognized_text = ""
        self.last_recognized_time = None
        self.stats = {
            'total_checked': 0,
            'skipped_duplicate': 0,
            'skipped_partial': 0,
            'skipped_similar': 0,
            'skipped_repetition': 0,  # 重复模式
            'skipped_time_window': 0,
            'passed': 0
        }
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_skipped = (self.stats['skipped_duplicate'] + 
                        self.stats['skipped_partial'] + 
                        self.stats['skipped_similar'])
        skip_rate = (total_skipped / self.stats['total_checked'] * 100) if self.stats['total_checked'] > 0 else 0
        
        return {
            **self.stats,
            'total_skipped': total_skipped,
            'skip_rate': skip_rate,
            'history_size': len(self.history)
        }

