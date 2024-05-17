from keras.models import load_model
from keras.optimizers import Adam

# 加载模型
model = load_model('../model.h5')

# 更改学习率
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['acc'])


# 保存训练的模型
model.save('../model.h5')