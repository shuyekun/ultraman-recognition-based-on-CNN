import os
# 构建模型
import tensorflow as tf
from keras.optimizers import RMSprop, Adam
# 图像预处理
from keras.preprocessing.image import ImageDataGenerator

# 数据集根目录
base_dir = '../ultraman'

# 指定每一种数据的位置
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

model = tf.keras.models.Sequential([
    # input_shape为图像大小+彩色(RGB)3通道，32为滤波器数量，(5,5)为滤波器大小,使用relu作为激活函数
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    # 池化层，窗口大小为(2,2),在每个窗口中只保留像素的最大值
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    # 全连接层
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    # 9个输出类别
    tf.keras.layers.Dense(9, activation='softmax')
])

# 进行优化方法选择和一些超参数设置
# 损失函数使用交叉熵
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0005),
              metrics=['acc'])

# 对图片进行预处理，将像素调整至[0，1]之间
# 同时进行数据增强
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# 生成训练集，会自动将文件夹名字设置为标签
train_generator = train_datagen.flow_from_directory(train_dir,  # 训练图片的位置
                                                    batch_size=20,  # 每一个投入多少张图片训练
                                                    class_mode='categorical',  # 设置我们需要的标签类型
                                                    target_size=(200, 200))  # 将图片统一大小

# 验证集
validation_generator = test_datagen.flow_from_directory(validation_dir,  # 验证图片的位置
                                                        batch_size=20,  # 每一个投入多少张图片训练
                                                        class_mode='categorical',  # 设置我们需要的标签类型
                                                        target_size=(200, 200))  # 将图片统一大小

# 进行训练
history = model.fit_generator(
    train_generator,  # 训练集数据
    steps_per_epoch=190,  # 每个epoch训练多少次
    epochs=20,  # 训练轮数，建议在[10,50]如果电脑训练速度快，可以大于50
    validation_data=validation_generator,  # 验证集数据
    validation_steps=50,
    verbose=1)  # 训练进度显示方式，可取值0，1（显示训练进度条），2（一个epoch输出一条信息）

# 保存训练的模型到当前目录
model.save('../model.h5')
