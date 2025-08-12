@echo off
echo 正在安装实时图像分类器所需的Python包...
echo.

echo 升级pip...
python -m pip install --upgrade pip

echo.
echo 安装依赖包...
pip install -r requirements.txt

echo.
echo 安装完成！
echo.
echo 现在您可以运行以下命令启动应用程序：
echo python main.py
echo.
pause
