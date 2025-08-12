#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型路径修复工具
"""

import os
import re

def fix_model_path():
    """修复main.py中的模型路径"""
    print("🔧 修复模型路径")
    print("=" * 30)
    
    # 检查main.py是否存在
    if not os.path.exists("main.py"):
        print("❌ main.py 文件不存在")
        return False
    
    # 读取main.py内容
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # 查找模型路径行
    path_patterns = [
        r'model_path\s*=\s*r?"[^"]*"',
        r'model_path\s*=\s*r?\'[^\']*\''
    ]
    
    found_path = False
    for pattern in path_patterns:
        match = re.search(pattern, content)
        if match:
            old_path = match.group()
            print(f"找到模型路径: {old_path}")
            found_path = True
            break
    
    if not found_path:
        print("❌ 未找到模型路径配置")
        return False
    
    # 检查模型文件是否存在
    model_files = [
        "model/model.tflite",
        "model\\model.tflite",
        "./model/model.tflite",
        ".\\model\\model.tflite"
    ]
    
    working_path = None
    for path in model_files:
        if os.path.exists(path):
            working_path = path
            break
    
    if not working_path:
        print("❌ 未找到模型文件")
        print("请确保 model.tflite 文件在 model 文件夹中")
        return False
    
    print(f"✅ 找到模型文件: {working_path}")
    
    # 替换路径
    new_path_line = f'model_path = "{working_path}"'
    new_content = re.sub(pattern, new_path_line, content)
    
    # 写回文件
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"✅ 已修复模型路径为: {working_path}")
    return True

def check_model_file():
    """检查模型文件"""
    print("\n📁 检查模型文件...")
    
    if os.path.exists("model/model.tflite"):
        size = os.path.getsize("model/model.tflite")
        print(f"✅ 模型文件存在 (大小: {size/1024/1024:.1f} MB)")
        return True
    elif os.path.exists("model\\model.tflite"):
        size = os.path.getsize("model\\model.tflite")
        print(f"✅ 模型文件存在 (大小: {size/1024/1024:.1f} MB)")
        return True
    else:
        print("❌ 模型文件不存在")
        print("请将您的 .tflite 模型文件放在 model 文件夹中")
        return False

def main():
    """主函数"""
    print("🔧 模型路径修复工具")
    print("=" * 50)
    
    # 检查模型文件
    if not check_model_file():
        return
    
    # 修复路径
    if fix_model_path():
        print("\n✅ 修复完成!")
        print("现在可以运行 python main.py 启动应用程序")
    else:
        print("\n❌ 修复失败")

if __name__ == "__main__":
    main()
