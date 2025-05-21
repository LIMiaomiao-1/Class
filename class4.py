
import gradio as gr
import numpy as np

# 定义浓度和吸光度的列表
c = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
a = np.array([0.14, 0.31, 0.46, 0.60, 0.73])

# 使用最小二乘法拟合一条直线，计算斜率和截距
slope, intercept = np.polyfit(c, a, 1)

def predict_concentration(absorbance):
    au = float(absorbance)
    cu = (au - intercept) / slope
    result = f"浓度: {cu:.2f} mg/L"
    print(result)
    return result

# 创建接口
demo = gr.Interface(fn=predict_concentration, inputs="text", outputs="text")

# 启动接口
demo.launch(share=True)
