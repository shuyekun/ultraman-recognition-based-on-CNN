import os
import random

from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import load_model
import numpy as np

# 加载测试数据
root = "../ultraman"
test_dir = os.path.join(root, "test")
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,  # 测试图片的位置
    batch_size=1,
    class_mode='categorical',
    target_size=(128, 128))

# 加载模型
model = load_model('../model.h5')

# 评估模型在所有测试集上的正确率
loss, acc = model.evaluate(test_generator)
print('Test loss:', loss)
print('Test accuracy:', acc)

# 测试文件夹路径
main_folder = '../ultraman/test'

# 获取所有子文件夹
folders = [os.path.join(main_folder, f) for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]


# 预处理图像
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


labels = ['aidi', 'daina', 'dijia', 'gaiya', 'leiou', 'mengbiyousi', 'sailuo', 'saiwen', 'tailuo']

# 在每个文件夹里随机选择 k 张图像进行预测
for folder in folders:
    images = os.listdir(folder)
    selected_images = random.sample(images, 5)
    for image in selected_images:
        image_path = os.path.join(folder, image)
        img_array = preprocess_image(image_path)
        # verbose=0 表示不显示进度条
        prediction = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(prediction)
        print(f'预测图像:{os.path.basename(image_path)} 实际标签: {os.path.basename(folder)}, 预测标签: {labels[predicted_index]}')
