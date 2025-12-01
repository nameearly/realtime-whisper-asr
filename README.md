# Realtime Whisper ASR

基于 Whisper 的实时语音识别系统，支持多语言识别、自动翻译、语音活动检测等功能。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ 功能特性

### 核心功能
- 🎤 **实时语音识别**：支持从麦克风实时捕获并识别语音
- 🌍 **多语言支持**：支持 100+ 种语言，自动检测或手动指定
- ⚡ **双后端支持**：支持 faster-whisper（推荐，速度快）和原始 whisper（安装简单）
- 🔄 **流式处理**：基于 whisper-streaming 实现低延迟流式识别
- 🌐 **自动翻译**：支持将识别结果自动翻译成中文（API 翻译）

### 智能优化
- 🎯 **VAD 语音活动检测**：使用 Silero VAD 自动检测语音开始和结束
- 🔄 **动态静音检测**：根据说话密集程度自动调整静音检测时间
- 📝 **智能去重**：多级去重机制（文本级 + 音频级），自动过滤重复结果
- 🎚️ **语速自适应**：根据语速自动调整识别参数
- 🎵 **人声分离**：支持从音频中分离人声和背景音乐（可选）

### 高级功能
- 📊 **性能监控**：实时显示识别速度、延迟等性能指标
- 📝 **完整日志**：记录所有识别结果、跳句信息、性能数据
- 🔧 **语言特定配置**：为 16+ 种常用语言提供优化配置
- 🛡️ **设备保护**：自动处理音频设备中断和恢复
- ⚙️ **灵活配置**：通过 `config.json` 统一管理所有配置

## 🚀 快速开始

### 系统要求

- Python 3.8+
- Windows 10/11（已测试）或 Linux/macOS
- 麦克风设备
- 足够的系统内存（推荐 8GB+）
- 可选：NVIDIA GPU（用于加速，需要 CUDA 支持）

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/yourusername/realtime-whisper-asr.git
cd realtime-whisper-asr
```

#### 2. 安装依赖

**基础依赖（必需）：**
```bash
pip install faster-whisper sounddevice numpy
```

**完整依赖（推荐）：**
```bash
pip install faster-whisper sounddevice numpy torch torchaudio silero-vad requests psutil
```

**可选依赖：**
- 使用原始 whisper：`pip install openai-whisper whisper-timestamped`
- 使用人声分离：`pip install demucs` 或 `pip install spleeter`

#### 3. 配置环境变量（可选）

如果使用翻译功能，需要配置 API key：

**方法 1：使用环境变量（推荐）**

复制 `.env.example` 为 `.env` 并填入你的 API key：
```bash
# Windows
copy .env.example .env
# 然后编辑 .env 文件，将 your_api_key_here 替换为你的实际 API key

# Linux/macOS
cp .env.example .env
# 然后编辑 .env 文件，将 your_api_key_here 替换为你的实际 API key
```

或者直接创建 `.env` 文件（项目根目录）：
```bash
# .env
SILICONFLOW_API_KEY=your_api_key_here
```

或者在系统中设置环境变量：
- **Windows:**
  ```cmd
  set SILICONFLOW_API_KEY=your_api_key_here
  ```
- **Linux/macOS:**
  ```bash
  export SILICONFLOW_API_KEY=your_api_key_here
  ```

**方法 2：直接在代码中配置**

如果不想使用环境变量，可以直接修改 `translation_manager.py` 中的配置（不推荐，安全性较低）。

> **注意**：`.env` 文件已添加到 `.gitignore`，不会被上传到 GitHub。

#### 4. 下载模型

首次运行时会自动下载模型，也可以手动下载：

- **faster-whisper 模型**：保存到 `models_fast/` 目录
- **原始 whisper 模型**：保存到 `models/` 目录

支持的模型大小：
- `tiny`：最快，准确度较低
- `base`：平衡选择（推荐）
- `small`：较好准确度
- `medium`：高准确度（较慢）
- `large-v2`/`large-v3`：最高准确度（最慢）

### 使用方法

#### 基本使用

直接运行主程序：

```bash
python 一键实时识别麦克风.py
```

程序会引导你完成以下配置：
1. 选择运行模式（GPU/CPU）
2. 选择识别后端（faster-whisper/whisper）
3. 选择模型大小
4. 选择识别语言
5. 选择任务类型（transcribe/translate）
6. 选择音频设备

#### 配置说明

所有配置通过 `config.json` 文件管理，主要配置项包括：

- **`translate_interval`**：翻译间隔（秒），默认 10 秒
- **`skip_detector`**：跳句检测配置（相似度阈值、时间窗口等）
- **`speech_rate_adaptive`**：语速自适应配置
- **`asr_optimization`**：ASR 优化配置（agreement_n、VAD 阈值等）
- **`vocal_separation`**：人声分离配置
- **`audio_deduplication`**：音频级别去重配置
- **`language_specific`**：语言特定配置（16+ 种语言）

详细配置说明请参考 `config.json` 文件中的注释。

## 📖 功能详解

### 1. 实时语音识别

支持从麦克风实时捕获音频并进行识别，识别结果实时显示。

**显示格式：**
- 识别结果：`💬 {原文}`
- 翻译结果：`🌐 {翻译}`（如果启用翻译）

### 2. 任务类型选择

程序支持两种任务类型：

- **`transcribe`**（转录模式）：
  - 输出与输入相同的语言
  - 如果启用翻译，会通过 API 将结果翻译成中文
  - 适合：需要看到原语言和翻译的用户

- **`translate`**（翻译模式）：
  - Whisper 直接翻译成英文
  - 不需要 API 翻译
  - 适合：懂英语，只需要英文翻译的用户

### 3. 自动翻译功能

如果选择 `transcribe` 模式，系统会：
- 每隔 `translate_interval` 秒自动翻译未翻译的识别结果
- 翻译失败时，最多跟随下一次内容重发一次
- 使用 SiliconFlow API（tencent/Hunyuan-MT-7B 模型）

**配置翻译间隔：**
在 `config.json` 中修改 `translate_interval` 值（单位：秒）

### 4. VAC 模式（推荐）

使用语音活动检测（VAD），自动检测语音开始和结束：

**特点：**
- 自动检测语音边界
- 动态调整静音检测时间（根据说话密集程度）
- 更自然的识别节奏
- 减少延迟

**配置：**
- 在 `config.json` 中配置 `speech_rate_adaptive` 参数
- 支持语言特定配置，自动根据识别语言选择最优参数

### 5. 智能去重机制

系统提供多级去重：

**文本级去重：**
- 使用编辑距离和相似度算法
- 检测完全重复、部分重复、相似文本
- 可配置相似度阈值和时间窗口

**音频级去重：**
- 在音频进入 ASR 模型之前检测重复
- 使用音频特征（RMS、频谱质心、过零率）进行相似度比较
- 减少模型计算量，提升速度

### 6. 语言特定优化

系统为以下语言提供了优化配置：

- 中文（zh）、英文（en）、日语（ja）、韩语（ko）
- 西班牙语（es）、法语（fr）、德语（de）、俄语（ru）
- 意大利语（it）、葡萄牙语（pt）、阿拉伯语（ar）、印地语（hi）
- 泰语（th）、越南语（vi）、印尼语（id）、荷兰语（nl）
- 波兰语（pl）、土耳其语（tr）

每种语言的配置包括：
- 跳句检测参数（相似度阈值、最小长度等）
- 语速自适应参数（静音检测时间、语速阈值等）
- ASR 优化参数（agreement_n、VAC 块大小、VAD 阈值等）

## 📁 项目结构

```
realtime-whisper-asr/
├── 一键实时识别麦克风.py    # 主程序
├── config.json              # 配置文件
├── README.md                # 项目说明
│
├── 核心模块/
│   ├── asr_components.py           # ASR 组件（VAD迭代器、ASR处理器）
│   ├── translation_manager.py      # 翻译管理器
│   ├── config_manager.py           # 配置管理器
│   └── enhanced_asr_processor.py   # 增强的 ASR 处理器
│
├── 功能模块/
│   ├── improved_skip_detector.py   # 改进的跳句检测器
│   ├── audio_deduplicator.py       # 音频级别去重
│   ├── audio_device_protector.py   # 音频设备保护器
│   ├── performance_monitor.py      # 性能监控
│   ├── performance_display.py      # 性能显示
│   ├── log_manager.py              # 日志管理
│   └── vocal_separation.py        # 人声分离
│
├── 辅助模块/
│   ├── speech_rate_adaptive.py     # 语速自适应
│   ├── speech_rate_audio_processor.py  # 语速音频处理器
│   └── time_utils.py                # 时间工具
│
├── 文档/
│   ├── 使用说明-新功能.md
│   ├── 人声分离使用说明.md
│   ├── 快速语速优化说明.md
│   ├── 语速自适应说明.md
│   ├── 识别准确率优化方案.md
│   └── 改进说明.md
│
├── models/                  # Whisper 模型目录（原始 whisper）
├── models_fast/             # Faster-Whisper 模型目录
├── logs/                    # 日志文件目录
└── whisper_streaming-main/  # whisper-streaming 库
```

## 🔧 配置说明

### 主要配置项

#### 翻译配置
```json
{
  "translate_interval": 10  // 翻译间隔（秒）
}
```

#### 跳句检测配置
```json
{
  "skip_detector": {
    "similarity_threshold": 0.85,  // 相似度阈值（0-1）
    "time_window": 3.0,            // 时间窗口（秒）
    "min_length": 2,               // 最小文本长度
    "use_edit_distance": true      // 使用编辑距离算法
  }
}
```

#### 语速自适应配置
```json
{
  "speech_rate_adaptive": {
    "enable": true,
    "initial_silence_ms": 1000,     // 初始静音检测时间
    "min_silence_ms": 500,          // 最小静音检测时间
    "max_silence_ms": 1500,         // 最大静音检测时间
    "rate_threshold_slow": 5.0,     // 慢语速阈值
    "rate_threshold_fast": 15.0     // 快语速阈值
  }
}
```

#### ASR 优化配置
```json
{
  "asr_optimization": {
    "agreement_n": 3,               // Local Agreement-n 的 n 值
    "vac_chunk_size": 0.08,         // VAC 音频块大小（秒）
    "beam_size": 5,                 // Whisper beam_size 参数
    "temperature": 0.0,             // Whisper temperature 参数
    "vad_threshold": 0.6            // VAD 语音检测阈值
  }
}
```

详细配置说明请参考 `config.json` 文件中的注释。

## 📊 日志系统

### 日志文件位置

所有日志文件保存在 `logs/` 目录下：

- `session_YYYYMMDD_HHMMSS.csv`：CSV 格式的结构化日志
- `session_YYYYMMDD_HHMMSS.log`：文本格式的详细日志
- `skip_YYYYMMDD_HHMMSS.log`：跳句专用日志文件

### 日志内容

- **识别结果**：所有语音识别结果
- **翻译结果**：API 翻译结果（如果启用）
- **跳句记录**：跳过的重复或部分重复结果
- **性能指标**：识别速度、延迟等
- **错误信息**：系统错误和异常

## 🎯 性能优化建议

### 模型选择

- **实时性优先**：使用 `base` 或 `small` 模型
- **准确度优先**：使用 `medium` 或 `large-v3` 模型
- **平衡选择**：使用 `medium` 模型（推荐）

### 后端选择

- **faster-whisper**（推荐）：
  - 速度快（约 4 倍加速）
  - 需要安装 `faster-whisper`
  - GPU 需要特定 CUDA 版本

- **whisper**（原始版本）：
  - 安装简单
  - GPU 支持更好
  - 速度较慢

### GPU 加速

如果系统有 NVIDIA GPU，faster-whisper 会自动使用 GPU 加速。

检查 CUDA 支持：
```python
import torch
print(torch.cuda.is_available())
```

### 内存优化

- 使用较小的模型（`base` 或 `small`）
- 关闭不必要的日志输出
- 减少处理缓冲区大小
- 使用音频级别去重（减少重复计算）

## ❓ 常见问题

### 1. 麦克风无法识别

- 检查麦克风是否已连接
- 检查系统音频权限设置
- 尝试手动指定设备索引
- 检查设备是否被其他程序占用

### 2. 识别延迟过高

- 使用较小的模型（`base` 或 `small`）
- 启用 GPU 加速
- 使用 VAC 模式
- 使用 faster-whisper 后端
- 调整 `vac_chunk_size` 参数

### 3. 识别结果重复

- 系统已内置多级去重机制
- 查看日志了解跳句详情
- 调整静音检测时间（VAC 模式）
- 调整 `skip_detector` 配置

### 4. 内存不足

- 使用较小的模型
- 关闭其他占用内存的程序
- 减少处理缓冲区大小
- 关闭人声分离功能

### 5. 翻译功能不工作

- 检查网络连接
- 检查 API 配置（在 `translation_manager.py` 中）
- 查看日志了解错误详情
- 确认选择了 `transcribe` 模式（不是 `translate` 模式）

### 6. GPU 无法使用

- 检查 CUDA 是否正确安装
- 检查 GPU 驱动是否最新
- 尝试使用原始 whisper 后端
- 查看错误日志了解详情

## 🛠️ 技术架构

### 核心组件

1. **Whisper Streaming**：基于 OpenAI Whisper 的流式识别实现
2. **Silero VAD**：语音活动检测，自动识别语音边界
3. **Dynamic VAD Iterator**：动态调整静音检测时间
4. **Enhanced ASR Processor**：增强的语音识别处理器
5. **Translation Manager**：翻译管理器，定期翻译未翻译的结果

### 去重机制

系统提供多级去重：

1. **文本级去重**：
   - 完全重复检测
   - 部分重复检测
   - 相似度检测（编辑距离算法）

2. **音频级去重**：
   - 音频特征提取（RMS、频谱质心、过零率）
   - 特征向量比较（余弦相似度）
   - 滑动窗口缓存

### 语言特定优化

系统为 16+ 种常用语言提供了优化配置，包括：
- 跳句检测参数优化
- 语速自适应参数优化
- ASR 优化参数调整

## 📝 开发说明

### 项目特点

- **模块化设计**：功能模块独立，易于维护和扩展
- **配置驱动**：所有配置通过 `config.json` 统一管理
- **错误处理**：完善的错误处理和日志记录
- **性能监控**：实时性能监控和统计

### 扩展开发

项目采用模块化设计，易于扩展：

- 添加新的语言配置：在 `config.json` 的 `language_specific` 中添加
- 添加新的去重算法：扩展 `improved_skip_detector.py`
- 添加新的翻译后端：修改 `translation_manager.py`
- 添加新的 ASR 后端：参考 `asr_components.py`

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📜 更新日志

### v2.0.0（最新）
- ✨ 新增自动翻译功能（API 翻译）
- ✨ 支持原始 whisper 后端（faster-whisper 和 whisper 双后端）
- ✨ 新增语言特定配置（16+ 种语言）
- ✨ 新增音频级别去重功能
- 🎯 优化跳句检测算法（编辑距离）
- 🎯 优化语速自适应机制
- 🎯 优化性能监控和显示
- 🐛 修复多个已知问题

### v1.0.0
- 初始版本
- 支持实时语音识别
- 支持 VAD 语音活动检测
- 支持动态静音检测
- 支持去重和跳句功能
- 完整的日志系统

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 语音识别模型
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - 快速 Whisper 实现
- [whisper-streaming](https://github.com/ufal/whisper_streaming) - 流式识别实现
- [Silero VAD](https://github.com/snakers4/silero-vad) - 语音活动检测

## 📮 联系方式

如有问题或建议，欢迎提交 Issue。
