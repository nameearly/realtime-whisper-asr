import whisper
import os

# 下载并加载指定模型，该地址已有则不会重复下载
# 我下的 base 141mb, small 461mb， medium 1.42GB
# 使用相对路径，相对于项目根目录
download_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
model = whisper.load_model("medium", download_root=download_dir)

# 音频文件路径（请根据实际情况修改）
audio_path = "your_audio_file.wav"  # 请修改为实际的音频文件路径
result = model.transcribe(audio_path,language="auto")  # 核心转录函数
print(result["text"])  # 提取转录文本

"""transcribe()参数包括：
重要
- `language`：ISO 639-1/2 语言代码，如"zh"，或设为 `"auto"` 以自动检测。
- `task`：设为 `"transcribe"`（默认）输出同语言文本  /   `"translate"` 则翻译成英语。

常用
- `length_penalty`：调整束搜索中对长/短序列的偏好。
- `suppress_tokens`：屏蔽词(token ID)列表。token_id.py
- `word_timestamps`：启用 token/单词级别的时间戳。
- `vad_filter`：启用内置的语音活动检测（VAD）；`vad_parameters` 用于调整其参数。
- `suppress_blank`：禁用空白 token，默认值为 `True`。
- verbose ：是否打印日志，注意无引号

关心模型运行细节
- `model`：仅在将 `whisper.transcribe` 作为模块级函数调用时使用；若已通过 `model = whisper.load_model(...)` 加载模型，则直接调用 `model.transcribe`。
- `temperature`：采样用的浮点数或列表，默认值为 `0`。
- `temperature_increment_on_fallback`：解码失败时提升的温度值，默认值为 `0.2`。
- `best_of`：采样时的候选数量，默认值为 `5`。
- `beam_size`：当 `temperature=0` 时的束搜索宽度。
- `patience`：束搜索的解码器耐心值，默认值为 `1`。
- `initial_prompt`：用于补充上下文的前置文本或 token。
- `prompt_reset_on_temperature`：温度升高时重置提示词的阈值，默认值为 `0.5`。
- `condition_on_previous_text`：是否让每个片段作为下一个片段的输入，默认值为 `True`。
- `compression_ratio_threshold`：过滤 gzip 压缩比过高的片段（可能是无效内容），默认值为 `2.4`。
- `logprob_threshold`：过滤平均对数概率过低的片段，默认值为 `-1.0`。
- `no_speech_threshold`：过滤无语音概率过高的片段，默认值为 `0.6`。
- `prepend_punctuations` / `append_punctuations`：时间戳对应的标点符号处理规则。
- `fp16`：若 GPU 支持则强制使用 FP16 运算，默认值为 `True`。
- `temperature_increment_on_fallback`、`compression_ratio_threshold` 等参数均通过关键字参数暴露。

"""