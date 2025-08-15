#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型检查工具
用于验证TFLite模型的兼容性和基本信息
"""

import tensorflow as tf
import numpy as np
import os
import sys

def check_model(model_path):
    """检查TFLite模型的基本信息"""
    print(f"正在检查模型: {model_path}")
    print("=" * 50)
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        return False
    
    try:
        # 加载模型
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # 获取输入输出信息
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✅ 模型加载成功!")
        print()
        
        # 显示输入信息
        print("📥 输入信息:")
        for i, detail in enumerate(input_details):
            print(f"  输入 {i}:")
            print(f"    - 名称: {detail['name']}")
            print(f"    - 形状: {detail['shape']}")
            print(f"    - 数据类型: {detail['dtype']}")
            print(f"    - 量化信息: {detail.get('quantization', '无')}")
            print()
        
        # 显示输出信息
        print("📤 输出信息:")
        for i, detail in enumerate(output_details):
            print(f"  输出 {i}:")
            print(f"    - 名称: {detail['name']}")
            print(f"    - 形状: {detail['shape']}")
            print(f"    - 数据类型: {detail['dtype']}")
            print(f"    - 量化信息: {detail.get('quantization', '无')}")
            print()
        
        # 分析模型类型
        analyze_model_type(input_details, output_details)
        
        # 检查兼容性
        check_compatibility(input_details, output_details)
        
        return True
        
    except Exception as e:
        print(f"❌ 模型检查失败: {e}")
        return False

def analyze_model_type(input_details, output_details):
    """分析模型类型"""
    print("🔍 模型类型分析:")
    
    # 检查输入
    input_shape = input_details[0]['shape']
    if len(input_shape) == 4 and input_shape[3] == 3:
        print("   ✅ 输入: 图像分类模型 (4D张量，3通道)")
    else:
        print(f"   ⚠️  输入: 非标准图像输入 {input_shape}")
    
    # 检查输出
    output_shape = output_details[0]['shape']
    if len(output_shape) == 2 and output_shape[0] == 1:
        num_classes = output_shape[1]
        print(f"   ✅ 输出: 分类模型 ({num_classes} 个类别)")
        
        if num_classes == 5:
            print("   💡 提示: 当前代码配置为5个类别，匹配!")
        else:
            print(f"   ⚠️  提示: 当前代码配置为5个类别，但模型有{num_classes}个类别")
            print(f"      需要更新 main.py 中的 self.labels 列表")
    else:
        print(f"   ⚠️  输出: 非标准分类输出 {output_shape}")
    
    print()

def check_compatibility(input_details, output_details):
    """检查与当前代码的兼容性"""
    print("🔧 兼容性检查:")
    
    # 检查输入尺寸
    input_shape = input_details[0]['shape']
    expected_sizes = [224, 299, 512]
    input_size = input_shape[1]  # 假设是正方形输入
    
    if input_size in expected_sizes:
        print(f"   ✅ 输入尺寸 {input_size}x{input_size} 是常见尺寸")
    else:
        print(f"   ⚠️  输入尺寸 {input_size}x{input_size} 不是常见尺寸")
        print(f"      常见尺寸: {expected_sizes}")
    
    # 检查数据类型
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.float32:
        print("   ✅ 输入数据类型: float32 (标准)")
    else:
        print(f"   ⚠️  输入数据类型: {input_dtype} (可能需要特殊处理)")
    
    # 检查量化
    input_quant = input_details[0].get('quantization', None)
    if input_quant:
        print("   ⚠️  模型已量化，可能需要特殊处理")
    else:
        print("   ✅ 模型未量化，使用标准处理")
    
    print()

def suggest_labels(num_classes):
    """根据类别数量建议标签"""
    print("🏷️  标签建议:")
    
    if num_classes == 5:
        print("   当前配置 (5个类别):")
        print("   ['cats', 'chicken', 'cow', 'dogs', 'elephant']")
    elif num_classes == 10:
        print("   10个类别示例:")
        print("   ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']")
    elif num_classes == 20:
        print("   20个类别示例:")
        print("   ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',")
        print("    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow']")
    else:
        print(f"   请根据您的模型训练数据定义 {num_classes} 个类别标签")
    
    print()

def auto_check_all_models():
    """自动检查指定目录下的所有模型"""
    import glob
    
    model_dir = r"C:\Users\AI_LAB_Student\image_classifier - Copy\model"
    
    print("🔍 自动检查所有模型")
    print("=" * 50)
    print(f"搜索目录: {model_dir}")
    print()
    
    # 检查目录是否存在
    if not os.path.exists(model_dir):
        print(f"❌ 目录不存在: {model_dir}")
        return False
    
    # 搜索所有.tflite文件
    tflite_files = glob.glob(os.path.join(model_dir, "*.tflite"))
    
    if not tflite_files:
        print("❌ 在指定目录中未找到.tflite文件")
        return False
    
    print(f"✅ 找到 {len(tflite_files)} 个模型文件:")
    for i, model_file in enumerate(tflite_files, 1):
        print(f"   {i}. {os.path.basename(model_file)}")
    print()
    
    # 检查每个模型
    results = {}
    for i, model_file in enumerate(tflite_files):
        print(f"📋 检查模型 {i+1}/{len(tflite_files)}: {os.path.basename(model_file)}")
        print("-" * 40)
        
        success = check_model(model_file)
        results[model_file] = success
        
        if i < len(tflite_files) - 1:  # 不是最后一个模型
            print("\n" + "=" * 60 + "\n")
    
    # 生成检查报告
    print("📊 检查报告")
    print("=" * 50)
    
    successful_models = [path for path, success in results.items() if success]
    failed_models = [path for path, success in results.items() if not success]
    
    print(f"✅ 成功检查: {len(successful_models)} 个模型")
    for model_path in successful_models:
        print(f"   - {os.path.basename(model_path)}")
    
    if failed_models:
        print(f"❌ 检查失败: {len(failed_models)} 个模型")
        for model_path in failed_models:
            print(f"   - {os.path.basename(model_path)}")
    
    print()
    print("💡 建议:")
    if successful_models:
        print("1. 成功的模型可以直接使用")
        print("2. 建议使用类别数量匹配的模型")
    if failed_models:
        print("3. 失败的模型需要检查文件完整性")
    
    return len(successful_models) > 0

def main():
    """主函数"""
    print("🔍 TFLite模型检查工具")
    print("=" * 50)
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all" or sys.argv[1] == "-a":
            # 自动检查所有模型
            success = auto_check_all_models()
            if success:
                print("✅ 所有模型检查完成!")
            else:
                print("❌ 模型检查过程中发现问题")
            return
        else:
            # 检查指定模型
            model_path = sys.argv[1]
    else:
        # 默认模型路径
        model_path = r"C:\Users\AI_LAB_Student\image_classifier\model\exported_model__animals_40_2_10 _True__20250808_001555__model.tflite"
        
        # 如果默认路径不存在，尝试其他常见路径
        if not os.path.exists(model_path):
            possible_paths = [
                "model/model.tflite",
                "model.tflite",
                "./model.tflite"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            else:
                print("❌ 未找到模型文件")
                print()
                print("💡 使用说明:")
                print("1. 检查单个模型: python check_model.py <模型文件路径>")
                print("2. 检查所有模型: python check_model.py --all")
                print("3. 检查所有模型: python check_model.py -a")
                return
    
    # 检查单个模型
    success = check_model(model_path)
    
    if success:
        print("✅ 模型检查完成!")
        print()
        print("💡 使用建议:")
        print("1. 如果模型兼容，可以直接使用")
        print("2. 如果不兼容，请参考README.md中的'更改模型'部分")
        print("3. 确保更新main.py中的类别标签")
    else:
        print("❌ 模型检查失败，请检查模型文件")

if __name__ == "__main__":
    main()
