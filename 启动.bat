@echo off
chcp 65001 >nul
title 实时麦克风语音识别
echo.
echo ========================================
echo   一键实时识别麦克风
echo ========================================
echo.
python 一键实时识别麦克风.py
if %errorlevel% neq 0 (
    echo.
    echo 运行出错，请检查：
    echo 1. 是否已安装依赖（运行 安装麦克风依赖.bat）
    echo 2. 麦克风是否已连接并启用
    echo.
    pause
)

