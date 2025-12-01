import os
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

"""
使用 faster-whisper 转录音频文件的简单示例。
运行前请先安装依赖: pip install faster-whisper soundfile huggingface-hub
"""

# 模型相关配置
# 手动下载地址：
#   https://huggingface.co/models?other=faster-whisper
# 我下的 medium 约 1.53GB
MODEL_SIZE = "medium"                 # 可选: tiny, base, small, medium, large-v2, large-v3
# 使用相对路径，相对于项目根目录
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_fast")

# 设备自动适配：优先尝试 GPU（cuda），失败自动退回 CPU
PREFERRED_DEVICE = "cuda"  # 想强制 CPU 的话改成 "cpu"

# 待转录音频（建议 16kHz 单声道 wav），可使用 ffmpeg 转换:
# ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
AUDIO_PATH = "your_audio_file.wav"  # 请修改为实际的音频文件路径

if not os.path.isfile(AUDIO_PATH):
    raise FileNotFoundError(f"音频文件不存在: {AUDIO_PATH}")

local_model_dir = os.path.join(MODEL_CACHE_DIR, MODEL_SIZE)
repo_id = f"guillaumekln/faster-whisper-{MODEL_SIZE}"

if not os.path.isdir(local_model_dir) or not os.listdir(local_model_dir):
    print(f"模型 {MODEL_SIZE} 尚未下载，将从 {repo_id} 获取（会显示自带进度条）...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_model_dir,
            resume_download=True,
        )
    except Exception as err:
        raise RuntimeError(
            f"下载模型失败：{err}\n"
            "请确认网络可访问 huggingface.co，或手动下载后再次运行。"
        ) from err
else:
    print(f"检测到本地模型缓存：{local_model_dir}")

print(f"正在加载 faster-whisper 模型 ({MODEL_SIZE}) ...")

model = None
load_errors = []

if PREFERRED_DEVICE == "cuda":
    # 先尝试 GPU
    try:
        model = WhisperModel(
            local_model_dir,
            device="cuda",
            compute_type="float16",
        )
        print("已使用 GPU (cuda + float16)。")
    except Exception as e:
        load_errors.append(e)
        print("GPU 加载失败，将自动退回 CPU。错误信息：", e)

if model is None:
    # 回退到 CPU
    model = WhisperModel(
        local_model_dir,
        device="cpu",
        compute_type="int8",
    )
    print("已使用 CPU (int8)。")

print("模型加载完成。开始转录 ...")

segments, info = model.transcribe(
    AUDIO_PATH,
    language=None,          # 自动检测语言，可改为 "zh"、"en" 等
    beam_size=5,
    word_timestamps=True
)

print(f"检测语言: {info.language}, 置信度: {info.language_probability:.2f}")
print("-" * 50)
full_text = []
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    full_text.append(segment.text)

print("-" * 50)
print("完整转录结果:")
print("".join(full_text).strip())


