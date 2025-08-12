import numpy as np
import cv2
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageGrab
import threading
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

class RealTimeImageClassifier:
    def __init__(self):
        try:
            ######## 加载TFLite模型#############
            model_path = "model/model.tflite"
            ######## 加载TFLite模型#############    
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
        except Exception as e:
            print(f"模型加载失败: {e}")
            # 创建一个简单的错误提示窗口
            error_root = tk.Tk()
            error_root.title("错误")
            error_root.geometry("400x200")
            ttk.Label(error_root, text=f"模型加载失败:\n{e}", font=("Arial", 12)).pack(pady=20)
            ttk.Button(error_root, text="确定", command=error_root.destroy).pack()
            error_root.mainloop()
            raise e
        
        # 获取模型输入输出信息
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        
        # 动物类别标签（根据您的模型调整）
        self.labels = [
            'cats', 'chicken', 'cow', 'dogs', 'elephant'
        ]
        
        # 创建GUI窗口
        self.root = tk.Tk()
        self.root.title("实时图像分类器")
        self.root.geometry("800x600")
        
        # 设置窗口属性，防止意外关闭
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.resizable(True, True)
        
        # 尝试将窗口置顶（可选）
        try:
            self.root.attributes('-topmost', True)
            self.root.after(1000, lambda: self.root.attributes('-topmost', False))
        except:
            pass
        
        # 创建控件
        self.setup_gui()
        
        # 控制变量
        self.is_running = False
        self.capture_thread = None
        
        # 屏幕捕获器 - 使用PIL的ImageGrab替代mss
        self.sct = None
        
    def setup_gui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 控制按钮
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        self.start_btn = ttk.Button(control_frame, text="开始捕获", command=self.start_capture)
        self.start_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_btn = ttk.Button(control_frame, text="停止捕获", command=self.stop_capture, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=(0, 10))
        
        # 捕获区域选择
        ttk.Label(control_frame, text="捕获区域:").grid(row=0, column=2, padx=(20, 5))
        self.area_var = tk.StringVar(value="全屏")
        area_combo = ttk.Combobox(control_frame, textvariable=self.area_var, 
                                 values=["全屏", "自定义区域"], state="readonly", width=15)
        area_combo.grid(row=0, column=3)
        
        # 自定义区域调整控件 - 移动到右侧
        self.custom_frame = ttk.Frame(main_frame)
        self.custom_frame.grid(row=0, column=4, columnspan=1, pady=(0, 10), sticky=(tk.E))
        
        # X坐标调整
        ttk.Label(self.custom_frame, text="X:").grid(row=0, column=0, padx=(0, 2))
        self.x_var = tk.IntVar(value=400)
        x_spinbox = ttk.Spinbox(self.custom_frame, from_=0, to=1920, textvariable=self.x_var, width=6)
        x_spinbox.grid(row=0, column=1, padx=(0, 5))
        
        # Y坐标调整
        ttk.Label(self.custom_frame, text="Y:").grid(row=0, column=2, padx=(0, 2))
        self.y_var = tk.IntVar(value=200)
        y_spinbox = ttk.Spinbox(self.custom_frame, from_=0, to=1080, textvariable=self.y_var, width=6)
        y_spinbox.grid(row=0, column=3, padx=(0, 5))
        
        # 宽度调整
        ttk.Label(self.custom_frame, text="宽:").grid(row=0, column=4, padx=(0, 2))
        self.width_var = tk.IntVar(value=400)
        width_spinbox = ttk.Spinbox(self.custom_frame, from_=100, to=800, textvariable=self.width_var, width=6)
        width_spinbox.grid(row=0, column=5, padx=(0, 5))
        
        # 高度调整
        ttk.Label(self.custom_frame, text="高:").grid(row=0, column=6, padx=(0, 2))
        self.height_var = tk.IntVar(value=300)
        height_spinbox = ttk.Spinbox(self.custom_frame, from_=100, to=600, textvariable=self.height_var, width=6)
        height_spinbox.grid(row=0, column=7, padx=(0, 5))
        
        # 重置按钮
        reset_btn = ttk.Button(self.custom_frame, text="重置", command=self.reset_custom_area)
        reset_btn.grid(row=0, column=8, padx=(5, 0))
        

        
        # 绑定区域选择变化事件
        area_combo.bind('<<ComboboxSelected>>', self.on_area_change)
        
        # 绑定自定义区域值变化事件
        self.x_var.trace('w', self.on_custom_area_change)
        self.y_var.trace('w', self.on_custom_area_change)
        self.width_var.trace('w', self.on_custom_area_change)
        self.height_var.trace('w', self.on_custom_area_change)
        
        # 初始状态：隐藏自定义区域控件
        self.custom_frame.grid_remove()
        
        # 图像显示区域
        self.image_label = ttk.Label(main_frame, text="等待开始捕获...")
        self.image_label.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # 分类结果显示
        result_frame = ttk.LabelFrame(main_frame, text="分类结果", padding="10")
        result_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        
        self.result_text = tk.Text(result_frame, height=8, width=60)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # 滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken")
        status_bar.grid(row=4, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        result_frame.columnconfigure(0, weight=1)
        
    def on_area_change(self, event=None):
        """当捕获区域选择改变时调用"""
        selected_area = self.area_var.get()
        
        if selected_area == "自定义区域":
            self.custom_frame.grid()  # 显示自定义区域控件
            # 立即更新状态栏
            self.on_custom_area_change()
        else:
            self.custom_frame.grid_remove()  # 隐藏自定义区域控件
            self.status_var.set("就绪")
        
    def on_custom_area_change(self, *args):
        """当自定义区域参数改变时调用"""
        if self.area_var.get() == "自定义区域":
            try:
                # 更新状态栏显示当前选择的区域
                x = self.x_var.get()
                y = self.y_var.get()
                width = self.width_var.get()
                height = self.height_var.get()
                status_text = f"自定义区域: X={x}, Y={y}, 宽={width}, 高={height}"
                self.status_var.set(status_text)
            except (ValueError, tk.TclError):
                # 如果值无效，不更新状态栏
                pass
        
    def reset_custom_area(self):
        """重置自定义区域为默认值"""
        self.x_var.set(400)
        self.y_var.set(200)
        self.width_var.set(400)
        self.height_var.set(300)
        self.status_var.set("已重置为默认区域设置")
        
    def on_closing(self):
        """窗口关闭时的处理"""
        if self.is_running:
            # 如果正在捕获，先停止
            self.stop_capture()
        
        # 确认是否真的要关闭
        if tk.messagebox.askokcancel("退出", "确定要退出应用程序吗？"):
            self.root.destroy()
        
    def start_capture(self):
        self.is_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("正在捕获...")
        
        # 清除之前的显示内容
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "正在捕获，请稍候...")
        
        # 启动捕获线程
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        
    def stop_capture(self):
        self.is_running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("已停止")
        
        # 清除图像显示区域，确保状态栏可见
        self.clear_image_display()
        
    def capture_loop(self):
        error_count = 0
        while self.is_running:
            try:
                # 捕获屏幕
                if self.area_var.get() == "全屏":
                    # 全屏捕获
                    screenshot = ImageGrab.grab()
                else:
                    # 使用用户自定义的区域设置
                    try:
                        x = self.x_var.get()
                        y = self.y_var.get()
                        width = self.width_var.get()
                        height = self.height_var.get()
                        bbox = (x, y, x + width, y + height)  # (left, top, right, bottom)
                        screenshot = ImageGrab.grab(bbox=bbox)
                    except (ValueError, tk.TclError):
                        # 如果自定义区域值无效，回退到全屏捕获
                        screenshot = ImageGrab.grab()
                
                # 转换为numpy数组
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # 预处理图像
                processed_img = self.preprocess_image(img)
                
                # 进行推理
                prediction = self.classify_image(processed_img)
                
                # 更新GUI（在主线程中）
                self.root.after(0, self.update_gui, img, prediction)
                
                # 重置错误计数
                error_count = 0
                
                # 控制帧率
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                error_count += 1
                print(f"捕获错误 #{error_count}: {e}")
                
                # 如果连续错误超过5次，尝试恢复
                if error_count >= 5:
                    print("连续错误过多，尝试恢复...")
                    self.root.after(0, lambda: self.status_var.set(f"捕获错误，正在恢复... (错误#{error_count})"))
                    time.sleep(2)
                    error_count = 0
                else:
                    time.sleep(1)
    
    def preprocess_image(self, image):
        # 调整图像大小以匹配模型输入
        target_size = (self.input_shape[1], self.input_shape[2])
        resized = cv2.resize(image, target_size)
        
        # 标准化
        normalized = resized.astype(np.float32) / 255.0
        
        # 添加batch维度
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def classify_image(self, image):
        try:
            # 设置输入张量
            self.interpreter.set_tensor(self.input_details[0]['index'], image)
            
            # 运行推理
            self.interpreter.invoke()
            
            # 获取输出
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # 获取top-3预测结果
            top_indices = np.argsort(output_data[0])[-3:][::-1]
            predictions = []
            
            for idx in top_indices:
                confidence = float(output_data[0][idx])
                label = self.labels[idx] if idx < len(self.labels) else f"类别{idx}"
                predictions.append((label, confidence))
            
            return predictions
            
        except Exception as e:
            print(f"推理错误: {e}")
            return [("错误", 0.0)]
    
    def update_gui(self, image, prediction):
        try:
            # 更新图像显示
            # 调整图像大小以适应显示
            display_img = cv2.resize(image, (400, 300))
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL图像
            pil_img = Image.fromarray(display_img)
            photo = ImageTk.PhotoImage(pil_img)
            
            # 更新标签
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # 保持引用
            
            # 更新分类结果
            self.result_text.delete(1.0, tk.END)
            timestamp = time.strftime("%H:%M:%S")
            result_str = f"[{timestamp}] 分类结果:\n\n"
            
            for i, (label, confidence) in enumerate(prediction):
                result_str += f"{i+1}. {label}: {confidence:.2%}\n"
            
            self.result_text.insert(tk.END, result_str)
            
            # 更新状态
            self.status_var.set(f"最后更新: {timestamp}")
            
        except Exception as e:
            print(f"GUI更新错误: {e}")
    
    def clear_image_display(self):
        """清除图像显示区域"""
        try:
            # 清除图像显示
            self.image_label.configure(image="", text="等待开始捕获...")
            self.image_label.image = None
            
            # 清除分类结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "等待开始捕获...")
            
        except Exception as e:
            print(f"清除显示错误: {e}")
    
    def run(self):
        try:
            self.root.mainloop()
        finally:
            pass

def main():
    try:
        app = RealTimeImageClassifier()
        app.run()
    except Exception as e:
        print(f"应用程序启动错误: {e}")
        input("按回车键退出...")

if __name__ == "__main__":
    main()