import os
#构建模型
import tensorflow as tf
from keras.optimizers import RMSprop

#图像预处理
from keras.preprocessing.image import ImageDataGenerator
#画图
import matplotlib.pyplot as plt

# 数据集根目录
base_dir = 'ultraman'

# 指定每一种数据的位置
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# 卷积神经网络模型例子
# # 模型处添加了dropout随机失效，也就是说有时候可能不用到某些神经元，失效率为0.5
# # 下面是模型的整体结构，可以观察到每一层卷积之后，都会使用一个最大池化层对提取的数据进行降维，减少计算量，后续实验修改网络结构主要修改下面部分
# model = tf.keras.models.Sequential([
#     # 我们的数据是150x150而且是三通道的，所以我们的输入应该设置为这样的格式。
#     tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(150, 150, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2), # 最大池化层
#     tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(128, (5, 5), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Dropout(0.5), # dropout 层通通过忽略一般数量特征，可以减少过拟合现象
#     tf.keras.layers.Flatten(), # 全链接层，将多维输入一维化
#     tf.keras.layers.Dense(512, activation='relu'),
#     # 二分类只需要一个输出
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#
# # 进行优化方法选择和一些超参数设置
# # 因为只有两个分类。所以用2分类的交叉熵，使用RMSprop，学习率为0.0001.优化指标为accuracy
# model.compile(loss='binary_crossentropy', # 损失函数使用交叉熵
# optimizer=RMSprop(lr=1e-4), # 优化器，学习率设置为0.0001
# metrics=['acc'])


#对图片进行预处理，将像素调整至[0，1]之间
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # 可能的其余变换处理，待测试
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1. / 255)


# 生成训练集，会自动将文件夹名字设置为标签
train_generator = train_datagen.flow_from_directory(train_dir, # 训练图片的位置
                                                    batch_size=5, # 每一个投入多少张图片训练
                                                    class_mode='categorical', # 设置我们需要的标签类型
                                                    target_size=(250, 250)) # 将图片统一大小

# 验证集
validation_generator = test_datagen.flow_from_directory(validation_dir, # 验证图片的位置
                                                        batch_size=5, # 每一个投入多少张图片训练
                                                        class_mode='categorical', # 设置我们需要的标签类型
                                                        target_size=(250, 250)) # 将图片统一大小
