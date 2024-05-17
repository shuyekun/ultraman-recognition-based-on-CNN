from keras.models import load_model
import os

# 图像预处理
from keras.preprocessing.image import ImageDataGenerator



# 加载模型
model = load_model('../model.h5')

# 数据集根目录
base_dir = '../ultraman'

# 指定每一种数据的位置
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# 对图片进行预处理，将像素调整至[0，1]之间
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


# 继续训练
history = model.fit(
    train_generator,  # 训练集数据
    steps_per_epoch=190,  # 每个epoch训练多少次
    epochs=5,  # 训练轮数，建议在[10,50]如果电脑训练速度快，可以大于50
    validation_data=validation_generator,  # 验证集数据
    validation_steps=50,
    verbose=1)  # 训练进度显示方式，可取值0，1（显示训练进度条），2（一个epoch输出一条信息）

# 保存训练的模型
model.save('../model.h5')
