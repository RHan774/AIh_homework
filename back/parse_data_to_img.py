"""该解析方法是在网上查找的
"""

import numpy as np
import struct
 
from PIL import Image
import os

# 解析图片数据文件
data_file = r'E:\桌面\学习\专业课\人工智能\lab\Week3-BP\Week3-BP\train_data\train-images.idx3-ubyte'
data_file_size = 47040016   # 通过“属性”查看文件大小
data_file_size = str(data_file_size - 16) + 'B' # 前16个字节为描述性内容
data_buf = open(data_file, 'rb').read()

magic, numImages, numRows, numColumns = struct.unpack_from(
    '>IIII', data_buf, 0)
datas = struct.unpack_from(
    '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
datas = np.array(datas).astype(np.uint8).reshape(
    numImages, 1, numRows, numColumns)

# 解析标签文件
label_file = r'E:\桌面\学习\专业课\人工智能\lab\Week3-BP\Week3-BP\train_data\train-labels.idx1-ubyte'
label_file_size = 60008 # 通过“属性”查看文件大小
label_file_size = str(label_file_size - 8) + 'B'    # 前8个字节为描述性内容
label_buf = open(label_file, 'rb').read()
 
magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from(
    '>' + label_file_size, label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)

datas_root = 'train_img'
if not os.path.exists(datas_root):
    os.mkdir(datas_root)

for i in range(10):
    file_name = datas_root + os.sep + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)
 
for ii in range(numLabels):
    img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
    label = labels[ii]
    file_name = datas_root + os.sep + str(label) + os.sep + \
                'train_img_' + str(ii) + '.png'
    img.save(file_name)