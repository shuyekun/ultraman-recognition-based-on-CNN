import os
from PIL import Image

def convert_webp_to_jpg(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img = Image.open(os.path.join(directory, filename)).convert('RGB')
            jpg_filename = os.path.splitext(filename)[0] + '.jpg'
            img.save(os.path.join(directory, jpg_filename), 'JPEG')

# 调用函数，将你的目录路径替换为 'your_directory'
convert_webp_to_jpg('C:\\Users\\shuye\\Desktop\\fix2')
