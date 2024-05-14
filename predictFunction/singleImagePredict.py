from keras_preprocessing import image
import numpy as np
from keras.models import load_model

model = load_model('../model.h5')

# 图像路径
img_path = '../ultraman/test/sailuo/0037.png'
img = image.load_img(img_path, target_size=(128, 128))  # 调整图像大小以匹配模型的输入大小

# 将图像转换为数组
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# 进行归一化
img_array /= 255.0

# 使用模型进行预测
predictions = model.predict(img_array)

# 获取预测结果中概率最大的类别
predicted_class = np.argmax(predictions[0])

# 标签
labels = ['aidi', 'daina', 'dijia', 'gaiya', 'leiou', 'mengbiyousi', 'sailuo', 'saiwen', 'tailuo']

# 打印预测的类别
print(f"实际标签: {img_path} 预测标签: {labels[predicted_class]}")