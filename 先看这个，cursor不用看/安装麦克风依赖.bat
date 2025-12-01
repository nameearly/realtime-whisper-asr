@echo off
chcp 65001 >nul
echo ========================================
echo 安装麦克风实时识别所需依赖
echo ========================================
echo.

echo 正在安装必需依赖...
pip install librosa soundfile
if %errorlevel% neq 0 (
    echo 安装失败，请检查 pip 是否可用
    pause
    exit /b 1
)

echo.
echo 正在安装可选依赖 (用于语音活动检测)...
pip install torch torchaudio
if %errorlevel% neq 0 (
    echo torch 安装失败，VAC/VAD 功能将不可用
    echo 你可以稍后手动安装: pip install torch torchaudio
)

echo.
echo 正在安装麦克风录音库 (sounddevice)...
pip install sounddevice
if %errorlevel% neq 0 (
    echo sounddevice 安装失败
    pause
    exit /b 1
)

echo.
echo 正在安装 HTTP 请求库 (requests，用于 LLM 翻译功能)...
pip install requests
if %errorlevel% neq 0 (
    echo requests 安装失败，LLM 翻译功能将不可用
    echo 你可以稍后手动安装: pip install requests
)

echo.
echo 正在安装硬件检测库 (psutil，用于自动配置)...
pip install psutil
if %errorlevel% neq 0 (
    echo psutil 安装失败，硬件检测功能将不可用
    echo 你可以稍后手动安装: pip install psutil
)

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 现在可以运行: python 一键实时识别麦克风.py
echo.
pause

