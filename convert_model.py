#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型转换工具
用于将其他格式的模型转换为TFLite格式
"""

import tensorflow as tf
import os
import sys

def convert_keras_to_tflite(keras_model_path, output_path):
    """将Keras模型转换为TFLite格式"""
    print(f"正在转换Keras模型: {keras_model_path}")
    
    try:
        # 加载Keras模型
        model = tf.keras.models.load_model(keras_model_path)
        
        # 创建转换器
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # 转换为TFLite
        tflite_model = converter.convert()
        
        # 保存模型
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ 转换成功! 保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

def convert_saved_model_to_tflite(saved_model_path, output_path):
    """将SavedModel转换为TFLite格式"""
    print(f"正在转换SavedModel: {saved_model_path}")
    
    try:
        # 创建转换器
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        # 转换为TFLite
        tflite_model = converter.convert()
        
        # 保存模型
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ 转换成功! 保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

def convert_with_optimization(model_path, output_path, model_type="keras"):
    """转换模型并应用优化"""
    print(f"正在转换并优化模型: {model_path}")
    
    try:
        if model_type == "keras":
            model = tf.keras.models.load_model(model_path)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
        elif model_type == "saved_model":
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        else:
            print("❌ 不支持的模型类型")
            return False
        
        # 启用优化
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 转换为TFLite
        tflite_model = converter.convert()
        
        # 保存模型
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ 优化转换成功! 保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

def convert_with_quantization(model_path, output_path, model_type="keras"):
    """转换模型并应用量化"""
    print(f"正在转换并量化模型: {model_path}")
    
    try:
        if model_type == "keras":
            model = tf.keras.models.load_model(model_path)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
        elif model_type == "saved_model":
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        else:
            print("❌ 不支持的模型类型")
            return False
        
        # 启用量化
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # 转换为TFLite
        tflite_model = converter.convert()
        
        # 保存模型
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ 量化转换成功! 保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

def show_conversion_examples():
    """显示转换示例"""
    print("📋 模型转换示例:")
    print()
    
    print("1. Keras模型转换为TFLite:")
    print("   python convert_model.py keras model.h5 model.tflite")
    print()
    
    print("2. SavedModel转换为TFLite:")
    print("   python convert_model.py saved_model ./saved_model/ model.tflite")
    print()
    
    print("3. 带优化的转换:")
    print("   python convert_model.py optimize model.h5 model_optimized.tflite")
    print()
    
    print("4. 带量化的转换:")
    print("   python convert_model.py quantize model.h5 model_quantized.tflite")
    print()

def main():
    """主函数"""
    print("🔄 TFLite模型转换工具")
    print("=" * 50)
    
    if len(sys.argv) < 3:
        show_conversion_examples()
        return
    
    conversion_type = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "converted_model.tflite"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}")
        return
    
    # 执行转换
    success = False
    
    if conversion_type == "keras":
        success = convert_keras_to_tflite(input_path, output_path)
    elif conversion_type == "saved_model":
        success = convert_saved_model_to_tflite(input_path, output_path)
    elif conversion_type == "optimize":
        success = convert_with_optimization(input_path, output_path, "keras")
    elif conversion_type == "quantize":
        success = convert_with_quantization(input_path, output_path, "keras")
    else:
        print(f"❌ 不支持的转换类型: {conversion_type}")
        show_conversion_examples()
        return
    
    if success:
        print()
        print("💡 转换完成后的步骤:")
        print("1. 使用 check_model.py 检查转换后的模型")
        print("2. 更新 main.py 中的模型路径")
        print("3. 更新类别标签")
        print("4. 测试应用程序")

if __name__ == "__main__":
    main()
