﻿@echo off
chcp 65001 >nul

echo 正在启动Grounding DINO UI...

:: 设置项目根目录（脚本所在目录）
set "SCRIPT_PATH=%~dp0"
set "PROJECT_ROOT=%SCRIPT_PATH%"

:: 切换到项目根目录
cd /d "%PROJECT_ROOT%" || (
echo 错误：无法切换到项目根目录
pause
exit /b 1
)

echo 当前工作目录: %PROJECT_ROOT%

:: 激活虚拟环境
echo 正在激活虚拟环境...
call "%PROJECT_ROOT%venv_groundingdino\Scripts\activate.bat" || (
echo 错误：无法激活虚拟环境
pause
exit /b 1
)

:: 启动UI程序
echo 正在启动UI程序...
echo 请在界面中选择要使用的设备(CPU/GPU)
python grounding_dino_ui.py

:: 检查启动结果
if %ERRORLEVEL% neq 0 (
echo 错误：UI程序启动失败
pause
)
