import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from numpy import array, expand_dims, argmax
from keras.models import load_model

model = load_model('../model.h5')
labels = ['爱迪奥特曼', '戴拿奥特曼', '迪迦奥特曼', '盖亚奥特曼', '雷欧奥特曼', '梦比优斯奥特曼', '赛罗奥特曼',
          '赛文奥特曼', '泰罗奥特曼']
ultraman_images = {
    0: 'img/aidi.jpg',
    1: 'img/daina.jpg',
    2: 'img/dijia.jpg',
    3: 'img/gaiya.jpg',
    4: 'img/leiou.jpg',
    5: 'img/mengbiyousi.jpg',
    6: 'img/sailuo.jpg',
    7: 'img/saiwen.png',
    8: 'img/tailuo.jpg',
}


def select_image():
    # 使用文件对话框让用户选择一张图片
    file_path = filedialog.askopenfilename()
    # 使用PIL库打开图片并缩放到合适的大小
    img = Image.open(file_path)

    # 对图片进行预处理并使用模型进行预测
    img = img.resize((200, 200))
    img = array(img) / 255.0
    img = expand_dims(img, axis=0)
    prediction = model.predict(img)

    # 获取预测结果中概率最大的类别
    predicted_class = argmax(prediction[0])
    # 打印预测的类别
    result_label.config(text="这是它，" + labels[predicted_class] + "!", font=("黑体", 15, "bold"))
    # 转换格式并显示图片
    select_img = Image.open(ultraman_images[predicted_class])
    select_img = select_img.resize((250, 250), Image.BICUBIC)
    select_img = ImageTk.PhotoImage(select_img)
    label.config(image=select_img)
    label.image = select_img


root = tk.Tk()

# 设置窗口的大小
window_width = 300
window_height = 320

# 获取屏幕的宽度和高度
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# 计算窗口的初始位置
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)

# 设置窗口的大小和位置
root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

# 创建一个按钮，点击后会触发select_image函数
button = tk.Button(root, text="选择图片", command=select_image)
button.pack()

# 创建一个标签，用于显示选择的图片
label = tk.Label(root)
label.pack()

# 创建一个标签，用于显示预测结果
result_label = tk.Label(root)
result_label.pack()

root.mainloop()