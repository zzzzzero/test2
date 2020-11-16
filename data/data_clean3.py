
#step3: 修改和再测试
import os
import piexif
import warnings
from PIL import Image
warnings.filterwarnings('error')

files = ['ExifError.txt', 'pExifError.txt']

for file in files:
    with open(file, 'r')as f:
        for i in f.readlines():
            i = i.strip()
            print(i.strip())
            piexif.remove(i.strip())
            try:
               img = Image.open(i)
            except Exception as e:
                print('Error:', str(e))
                print(i)

