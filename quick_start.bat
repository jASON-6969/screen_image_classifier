@echo off
echo 实时图像分类器 - 快速启动
echo ================================

echo.
echo 1. 检查模型文件...
if exist "model\model.tflite" (
    echo    ✅ 模型文件存在
) else (
    echo    ❌ 模型文件不存在
    echo    请确保 model.tflite 文件在 model 文件夹中
    pause
    exit /b 1
)

echo.
echo 2. 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo    ❌ Python未安装或不在PATH中
    pause
    exit /b 1
) else (
    echo    ✅ Python环境正常
)

echo.
echo 3. 检查依赖...
python -c "import tensorflow, cv2, PIL" >nul 2>&1
if %errorlevel% neq 0 (
    echo    ⚠️  依赖可能未完全安装
    echo    正在尝试安装依赖...
    pip install -r requirements.txt
) else (
    echo    ✅ 依赖检查通过
)

echo.
echo 4. 启动应用程序...
echo    正在启动实时图像分类器...
python main.py

echo.
echo 应用程序已退出
pause
