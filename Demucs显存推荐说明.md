# Demucs 显存推荐功能说明

## 功能概述

程序现在可以根据 GPU 显存情况，智能推荐合适的 Demucs 模型配置，同时考虑 Whisper 语音识别模型的显存占用。

## 显存需求参考

### Whisper 模型显存占用（float16）

| 模型大小 | 显存占用 | 说明 |
|---------|---------|------|
| tiny | ~0.5GB | 最小模型，速度快但准确度较低 |
| base | ~1GB | 平衡速度和准确度 |
| small | ~2GB | 较好的准确度 |
| medium | ~5GB | 高准确度，推荐用于实时场景 |
| large-v2/v3 | ~10GB | 最高准确度，但延迟较高 |

### Demucs 模型显存占用

| 模型名称 | 显存占用 | 说明 |
|---------|---------|------|
| htdemucs | ~1.5-2GB | 轻量级，适合实时（推荐） |
| htdemucs_ft | ~3-4GB | 更高质量但更慢 |
| htdemucs_6s | ~2-2.5GB | 6种音源分离 |
| hdemucs_mmi | ~2.5-3GB | 混合模型 |
| mdx | ~2-3GB | 高质量，较慢 |
| mdx_extra | ~3-4GB | 最高质量，最慢 |

**注意**：实际显存占用还取决于音频长度和批处理大小。

## 推荐逻辑

程序会根据以下信息进行推荐：

1. **GPU 总显存**：检测到的 GPU 显存容量
2. **当前可用显存**：当前未被占用的显存
3. **Whisper 模型大小**：用户选择的 Whisper 模型
4. **系统开销**：预留 1GB 给系统和其他开销

### 推荐规则

| 可用于 Demucs 的显存 | 推荐模型 | 说明 |
|---------------------|---------|------|
| ≥ 3.5GB | htdemucs_ft | 显存充足，可以使用高质量模型 |
| 2.5GB - 3.5GB | htdemucs | 显存中等，使用标准模型 |
| 1.5GB - 2.5GB | htdemucs | 显存紧张，使用轻量级模型（可能OOM） |
| < 1.5GB | 不推荐 | 显存不足，建议使用 filter 方法或关闭人声分离 |

## 使用示例

### 示例 1：8GB 显存 + medium Whisper 模型

```
GPU总显存: 8GB
当前可用显存: 7.5GB
Whisper模型 (medium) 预计占用: ~5GB
可用于Demucs的显存: ~1.5GB

⚠ 显存紧张，建议使用 htdemucs（最轻量级）
  如果出现OOM错误，考虑：
  1. 降低Whisper模型大小
  2. 使用filter方法替代Demucs
  3. 关闭人声分离
```

### 示例 2：12GB 显存 + small Whisper 模型

```
GPU总显存: 12GB
当前可用显存: 11GB
Whisper模型 (small) 预计占用: ~2GB
可用于Demucs的显存: ~8GB

✓ 推荐: htdemucs_ft（更高质量，显存充足）
  显存占用: ~3-4GB，分离质量最高
```

### 示例 3：6GB 显存 + large-v2 Whisper 模型

```
GPU总显存: 6GB
当前可用显存: 5.5GB
Whisper模型 (large-v2) 预计占用: ~10GB
可用于Demucs的显存: ~-5.5GB

❌ 显存不足，无法同时运行Whisper和Demucs
  需要至少 13.5GB 显存（Whisper + Demucs + 系统开销）
  建议：
  1. 使用filter方法（频域滤波，无需额外显存）
  2. 降低Whisper模型大小
  3. 关闭人声分离
```

## 程序启动时的推荐流程

1. **硬件检测**：检测 GPU 显存情况
2. **Whisper 推荐**：根据显存推荐 Whisper 模型
3. **用户选择**：用户确认或选择 Whisper 模型
4. **Demucs 推荐**：根据选择的 Whisper 模型和剩余显存推荐 Demucs 模型
5. **配置应用**：用户可以选择是否应用推荐配置到 `config.json`

## 配置建议

### 场景 1：显存充足（≥12GB）

```json
{
  "vocal_separation": {
    "enable": true,
    "method": "demucs",
    "demucs_model": "htdemucs_ft"
  }
}
```

**推荐 Whisper 模型**：medium 或 large-v2

### 场景 2：显存中等（6-12GB）

```json
{
  "vocal_separation": {
    "enable": true,
    "method": "demucs",
    "demucs_model": "htdemucs"
  }
}
```

**推荐 Whisper 模型**：small 或 medium

### 场景 3：显存紧张（4-6GB）

```json
{
  "vocal_separation": {
    "enable": true,
    "method": "demucs",
    "demucs_model": "htdemucs"
  }
}
```

**推荐 Whisper 模型**：base 或 small

**注意**：如果出现 OOM（Out of Memory）错误，考虑：
- 降低 Whisper 模型大小
- 使用 filter 方法替代 Demucs
- 关闭人声分离

### 场景 4：显存不足（<4GB）

```json
{
  "vocal_separation": {
    "enable": true,
    "method": "filter"
  }
}
```

**推荐 Whisper 模型**：tiny 或 base

**说明**：filter 方法使用频域滤波，无需额外显存，但效果有限。

## 故障排除

### 问题 1：出现 OOM（Out of Memory）错误

**原因**：显存不足，无法同时加载 Whisper 和 Demucs 模型

**解决方案**：
1. 降低 Whisper 模型大小（如从 medium 改为 small）
2. 使用 filter 方法替代 Demucs
3. 关闭人声分离
4. 如果可能，升级 GPU 显存

### 问题 2：推荐配置与实际不符

**原因**：显存占用估算可能因实际音频长度和处理参数而有所不同

**解决方案**：
1. 根据实际运行情况调整配置
2. 使用 `nvidia-smi` 监控实际显存使用
3. 如果显存充足，可以尝试更高质量的模型

### 问题 3：CPU 模式下无法使用 Demucs

**原因**：Demucs 在 CPU 上处理速度太慢，不适合实时场景

**解决方案**：
- 使用 filter 方法（频域滤波，CPU 上速度较快）
- 或关闭人声分离

## 监控显存使用

可以使用以下命令实时监控 GPU 显存使用：

```bash
# Windows (PowerShell)
nvidia-smi -l 1

# Linux/macOS
watch -n 1 nvidia-smi
```

## 参考信息

- **Demucs 官方文档**: https://github.com/facebookresearch/demucs
- **Whisper 模型大小**: 参考程序中的 `recommend_config()` 函数
- **显存优化技巧**: 
  - 使用 float16 精度（GPU 模式）
  - 减小批处理大小
  - 使用更短的音频片段

## 总结

显存推荐功能帮助您：
1. ✅ 根据硬件情况选择最合适的模型配置
2. ✅ 避免 OOM 错误
3. ✅ 平衡质量和性能
4. ✅ 优化显存使用

建议在首次运行时使用推荐配置，然后根据实际效果进行调整。

