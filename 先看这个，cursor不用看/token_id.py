import whisper

# 1. 加载Whisper的分词器（和你用的模型对应，比如base模型）
model_name = "base"  # 换成你实际用的模型（small/medium等）
tokenizer = whisper.tokenizer.get_tokenizer(
    whisper.load_model(model_name).is_multilingual,  # 多语言模型设为True
    language="zh",  # 目标语言（如查中文就写zh，英文写en）
    task="transcribe"  # 任务类型（和转写时一致）
)

# 2. 查询「文本/符号」对应的Token ID
# 示例1：查“你好”的Token ID
text = "你好"
token_ids = tokenizer.encode(text)
print(f"“{text}”对应的Token ID：{token_ids}")

# 示例2：查逗号“，”的Token ID
text = "，"
token_ids = tokenizer.encode(text)
print(f"“{text}”对应的Token ID：{token_ids}")

# 3. 反向查询：已知Token ID，查对应的内容
token_id = 1001  # 假设某个ID，替换成你要查的
text = tokenizer.decode([token_id])
print(f"Token ID {token_id}对应的内容：{text}")