#!/bin/bash
# 统一启动Grounding DINO UI脚本
# 支持CPU和GPU模式选择
# Linux版本

echo -e "\e[32m正在启动Grounding DINO UI...\e[0m"

# 切换到项目根目录
script_path=$(readlink -f "$0")
project_root=$(dirname "$script_path")
cd "$project_root" || { echo -e "\e[31m错误：无法切换到项目根目录\e[0m"; exit 1; }
echo -e "\e[32m当前工作目录: $project_root\e[0m"

# 检查并激活虚拟环境
echo -e "\e[32m正在激活虚拟环境...\e[0m"
venv_path="$project_root/venv_groundingdino/bin/activate"
if [ -f "$venv_path" ]; then
    source "$venv_path"
else
    echo -e "\e[31m错误：虚拟环境不存在，请先创建虚拟环境。\e[0m"
    read -p "按Enter键退出"
    exit 1
fi

# 启动UI程序（UI程序已移至项目根目录）
echo -e "\e[32m正在启动UI程序...\e[0m"
echo -e "\e[33m请在界面中选择要使用的设备(CPU/GPU)\e[0m"
python grounding_dino_ui.py

# 检查是否成功启动
if [ $? -ne 0 ]; then
    echo -e "\e[31m错误：UI程序启动失败，请检查错误信息。\e[0m"
    read -p "按Enter键退出"
    exit 1
fi