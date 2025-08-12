# 实时图像分类器

这是一个使用TFLite模型进行实时屏幕捕获和图像分类的Windows应用程序。

## 功能特性

- 🖥️ 实时屏幕捕获（全屏或自定义区域）
- 🤖 使用预训练的TFLite模型进行图像分类
- 📊 实时显示分类结果和置信度

## 安装要求

### 系统要求
- Windows 10/11
- Python 3.7+
- 至少4GB RAM

### Python依赖
```bash
# 方法1：使用安装脚本（推荐）
install.bat

# 方法2：手动安装
pip install -r requirements.txt
```

## 快速开始

### 1. 检查您的模型
```bash
# 检查当前模型是否兼容
python check_model.py
```

### 2. 启动应用程序
```bash
# 方法1：使用启动脚本（推荐）
start.bat

# 方法2：直接运行
python main.py
```

### 3. 使用界面
- 点击"开始捕获"按钮开始实时分类
- 选择捕获区域（全屏或自定义区域）
- 查看实时分类结果
- 点击"停止捕获"按钮停止

### 4. 功能说明
- **全屏模式**: 捕获整个屏幕进行实时分类
- **自定义区域**: 可调整X、Y坐标、宽度和高度，实时预览捕获区域
- **分类结果**: 显示top-3预测结果和置信度
- **实时预览**: 显示当前捕获的图像

## 使用方法

### 基本操作
1. **启动应用程序**
   ```bash
   # 方法1：使用启动脚本（推荐）
   start.bat
   
   # 方法2：直接运行
   python main.py
   ```

2. **使用界面**
   - 点击"开始捕获"按钮开始实时分类
   - 选择捕获区域（全屏或自定义区域）
   - 查看实时分类结果
   - 点击"停止捕获"按钮停止

3. **功能说明**
   - **全屏模式**: 捕获整个屏幕进行实时分类
   - **自定义区域**: 可调整X、Y坐标、宽度和高度，实时预览捕获区域
   - **分类结果**: 显示top-3预测结果和置信度
   - **实时预览**: 显示当前捕获的图像

## 模型信息

- **模型文件**: `model.tflite`
- **类别数量**: 5种动物
- **输入尺寸**: 根据模型自动检测
- **输出**: 分类概率分布

## 更改模型

### 1. 更改模型文件位置

#### 方法1：修改代码中的模型路径
在 `main.py` 文件中找到以下代码行：
```python
model_path = r"\image_classifier\model\model.tflite"
```

将其更改为您的新模型路径：
```python
model_path = r"您的模型文件完整路径\您的模型文件名.tflite"
```

#### 方法2：使用相对路径
将模型文件放在项目目录下，使用相对路径：
```python
model_path = "model/your_model.tflite"
```

### 2. 更改模型类型和类别

#### 更新类别标签
在 `main.py` 的 `__init__` 方法中找到：
```python
self.labels = [
    'cats', 'chicken', 'cow', 'dogs', 'elephant'
]
```

根据您的新模型更新类别标签：
```python
# 例如：用于识别不同物体的模型
self.labels = [
    'person', 'car', 'bicycle', 'dog', 'cat', 'bird'
]

# 或者用于识别不同食物的模型
self.labels = [
    'apple', 'banana', 'orange', 'grape', 'strawberry'
]
```

#### 支持的模型类型

**图像分类模型**（推荐）：
- 输入：单张图像
- 输出：类别概率分布
- 格式：TFLite (.tflite)
- 输入尺寸：通常为 224x224, 299x299, 或 512x512


### 3. 模型兼容性检查

#### 检查模型输入输出

**使用模型检查工具**
```bash
# 检查默认模型
python check_model.py

# 检查指定模型
python check_model.py path/to/your/model.tflite
```


#### 常见模型格式
- **MobileNet**: 输入尺寸 224x224x3
- **EfficientNet**: 输入尺寸 224x224x3 或 299x299x3
- **ResNet**: 输入尺寸 224x224x3
- **Inception**: 输入尺寸 299x299x3

### 4. 模型预处理要求

#### 图像预处理
当前代码使用以下预处理：
```python
# 调整图像大小
resized = cv2.resize(image, target_size)

# 标准化到 [0,1]
normalized = resized.astype(np.float32) / 255.0
```

如果您的模型需要不同的预处理，请修改 `preprocess_image` 方法：

```python
def preprocess_image(self, image):
    # 调整图像大小
    target_size = (self.input_shape[1], self.input_shape[2])
    resized = cv2.resize(image, target_size)
    
    # 根据模型要求进行预处理
    if self.model_type == "mobilenet":
        # MobileNet预处理
        normalized = resized.astype(np.float32) / 255.0
    elif self.model_type == "efficientnet":
        # EfficientNet预处理
        normalized = resized.astype(np.float32) / 255.0
    elif self.model_type == "custom":
        # 自定义预处理
        normalized = (resized.astype(np.float32) - 127.5) / 127.5
    
    # 添加batch维度
    batched = np.expand_dims(normalized, axis=0)
    return batched
```

### 5. 模型性能优化

#### 启用GPU加速（如果可用）
```python
# 在模型加载时启用GPU
interpreter = tf.lite.Interpreter(
    model_path=model_path,
    experimental_delegates=[tf.lite.load_delegate('libedgetpu.so.1')]
)
```

#### 量化模型
使用量化模型可以提高推理速度：
```python
# 加载量化模型
interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
```

### 6. 模型转换

#### 转换其他格式的模型
如果您有其他格式的模型（如Keras .h5文件），可以使用转换工具：

```bash
# 转换Keras模型
python convert_model.py keras model.h5 model.tflite

# 转换SavedModel
python convert_model.py saved_model ./saved_model/ model.tflite

# 带优化的转换
python convert_model.py optimize model.h5 model_optimized.tflite

# 带量化的转换
python convert_model.py quantize model.h5 model_quantized.tflite
```

#### 支持的输入格式
- **Keras模型** (.h5, .keras)
- **SavedModel** (目录格式)
- **TensorFlow Hub模型**

### 7. 故障排除

#### 模型加载错误
- 检查文件路径是否正确
- 确保模型文件未损坏
- 验证模型格式是否为TFLite

#### 推理错误
- 检查输入尺寸是否匹配
- 验证预处理步骤是否正确
- 确认类别数量与模型输出匹配

#### 性能问题
- 使用更小的模型
- 降低输入分辨率
- 启用模型量化
- 使用GPU加速

#### 转换错误
- 确保输入模型格式正确
- 检查TensorFlow版本兼容性
- 验证模型结构是否支持转换

## 支持的动物类别

当前模型支持以下5种动物类别：
- cats (猫)
- chicken (鸡)
- cow (牛)
- dogs (狗)
- elephant (大象)

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确保模型文件存在且可访问
   - 验证模型文件是否为有效的TFLite格式

2. **依赖安装问题**
   - 使用管理员权限运行pip
   - 尝试升级pip: `python -m pip install --upgrade pip`

3. **性能问题**
   - 降低捕获帧率（修改代码中的`time.sleep(0.1)`）
   - 使用自定义区域而非全屏捕获
   - 使用更小的模型或量化模型

4. **TFLite运行时错误**
   - 确保安装了正确的tensorflow版本
   - 检查Python版本兼容性

5. **模型推理错误**
   - 检查模型输入尺寸是否匹配
   - 验证类别标签数量与模型输出匹配
   - 确认图像预处理步骤正确

6. **类别标签不匹配**
   - 确保`self.labels`列表与模型输出类别数量一致
   - 检查类别标签顺序是否正确

7. **自定义区域设置问题**
   - 确保X、Y坐标在屏幕范围内
   - 检查宽度和高度设置是否合理
   - 验证自定义区域不会超出屏幕边界

## 自定义配置

### 修改捕获区域
现在您可以通过界面直接调整捕获区域：
- **X坐标**: 距离屏幕左侧的像素数 (0-1920)
- **Y坐标**: 距离屏幕顶部的像素数 (0-1080)  
- **宽度**: 捕获区域的宽度 (100-800像素)
- **高度**: 捕获区域的高度 (100-600像素)

或者通过代码修改默认值：
```python
self.x_var = tk.IntVar(value=400)      # X坐标
self.y_var = tk.IntVar(value=200)      # Y坐标
self.width_var = tk.IntVar(value=400)  # 宽度
self.height_var = tk.IntVar(value=300) # 高度
```

### 调整帧率
修改`time.sleep(0.1)`中的数值：
- `0.1` = 10 FPS
- `0.05` = 20 FPS
- `0.2` = 5 FPS

### 修改类别标签
在`__init__`方法中更新`self.labels`列表。

## 许可证

本项目仅供学习和研究使用。

## 技术支持

如遇到问题，请检查：
1. Python版本兼容性
2. 依赖包版本
3. 模型文件完整性
4. 系统权限设置
#
