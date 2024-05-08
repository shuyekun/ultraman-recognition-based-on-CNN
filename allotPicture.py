import os
import shutil
import numpy as np


def split_dataset_into_train_val_test(dataset_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.2):
    # 获取所有子目录
    sub_dirs = [sub_dir for sub_dir in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, sub_dir))]

    # 对每个子目录进行处理
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(dataset_dir, sub_dir)
        files = os.listdir(sub_dir_path)

        # 打乱文件顺序
        np.random.shuffle(files)

        # 计算训练、验证和测试集的大小
        train_size = int(len(files) * train_ratio)
        val_size = int(len(files) * val_ratio)

        # 分配文件到对应的集合
        for i, file in enumerate(files):
            src_file_path = os.path.join(sub_dir_path, file)

            if i < train_size:
                dst_dir = os.path.join(train_dir, sub_dir)
            elif i < train_size + val_size:
                dst_dir = os.path.join(val_dir, sub_dir)
            else:
                dst_dir = os.path.join(test_dir, sub_dir)

            # 创建目标目录
            os.makedirs(dst_dir, exist_ok=True)

            # 移动文件
            shutil.move(src_file_path, os.path.join(dst_dir, file))


# 根目录路径
# 自动将文件夹中的图片按比例分为train validation 和test
# 默认比例为 7:2:1
root = "ultraman"
split_dataset_into_train_val_test(root, root + '/train', root + '/validation', root + '/test')
