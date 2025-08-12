@echo off
echo 正在启动实时图像分类器（稳定版）...
echo.

:start_app
echo 启动应用程序...
python main.py

if %errorlevel% neq 0 (
    echo.
    echo 应用程序异常退出，错误代码: %errorlevel%
    echo.
    echo 是否要重新启动？ (Y/N)
    set /p choice=
    if /i "%choice%"=="Y" (
        echo 重新启动中...
        goto start_app
    ) else (
        echo 按任意键退出...
        pause >nul
    )
) else (
    echo.
    echo 应用程序正常退出
    pause
)
