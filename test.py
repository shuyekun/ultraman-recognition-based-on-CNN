from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np

# 加载测试数据
root = "ultraman"
test_dir = root + "/test"
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,  # 测试图片的位置
    batch_size=1,
    class_mode='categorical',
    target_size=(256, 256))

model = load_model('model.h5')

# # 评估正确值模型
# loss, acc = model.evaluate(test_generator)
# print('Test accuracy:', acc)

# 预测
predictions = model.predict(test_generator)

# 获取预测结果中概率最大的类别
predicted_classes = np.argmax(predictions, axis=1)

# 获取类别和对应索引的字典
class_indices = test_generator.class_indices

# 反转字典，得到索引到类别名称的映射
index_to_class = {v: k for k, v in class_indices.items()}

# 使用映射将预测的索引转换为类别名称
predicted_class_names = [index_to_class[i] for i in predicted_classes]

# 打印预测的类别名称
print("predicted:")
print(predicted_class_names)

# 获取实际的类别索引
actual_classes = test_generator.classes

# 使用映射将实际的索引转换为类别名称
actual_class_names = [index_to_class[i] for i in actual_classes]

# 打印实际的类别名称
print("actual:")
print(actual_class_names)
