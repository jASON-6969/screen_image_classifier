@echo off
echo 实时图像分类器启动器
echo ========================
echo.

echo 1. 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo    ❌ Python未安装或不在PATH中
    echo    请先安装Python并确保在PATH中
    pause
    exit /b 1
) else (
    echo    ✅ Python环境正常
)

echo.
echo 2. 检查模型文件...
if exist "model\*.tflite" (
    echo    ✅ 模型文件存在
) else (
    echo    ⚠️  未找到模型文件
    echo    请确保model文件夹中有.tflite文件
)

echo.
echo 3. 启动应用程序...
echo    正在启动实时图像分类器...
echo.
python main.py

echo.
echo 应用程序已退出
pause
