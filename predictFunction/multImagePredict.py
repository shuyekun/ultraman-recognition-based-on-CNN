import os
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# 加载测试数据
root = "../ultraman"
test_dir = os.path.join(root, "test")
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,  # 测试图片的位置
    batch_size=1,
    class_mode='categorical',
    target_size=(200, 200))

# 加载模型
model = load_model('../model.h5')

# 评估模型在所有测试集上的正确率
loss, acc = model.evaluate(test_generator)
print('Test loss:', loss)
print('Test accuracy:', acc)


labels = ['aidi', 'daina', 'dijia', 'gaiya', 'leiou', 'mengbiyousi', 'sailuo', 'saiwen', 'tailuo']

# 使用ImageDataGenerator来读取测试数据
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(200, 200),
    batch_size=20,
    class_mode='categorical',
    shuffle=False)

# 预测标签
Y_pred = model.predict_generator(test_generator, len(test_generator))
y_pred = np.argmax(Y_pred, axis=1)

# 打印每个标签的分类报告
print('分类情况')
print(classification_report(test_generator.classes, y_pred, target_names=labels))

# 打印混淆矩阵
print('混淆矩阵')
print(confusion_matrix(test_generator.classes, y_pred))