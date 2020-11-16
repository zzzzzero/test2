# Refer: https://blog.csdn.net/a19990412/article/details/105940446
# pip install piexif -i https://pypi.tuna.tsinghua.edu.cn/simple/
#step1: 遍历所有图片，筛选有问题的
import os
from PIL import Image
import cv2
import warnings

warnings.filterwarnings('error')

root = './train'

f1 = open('pExifError.txt', 'w')
f2 = open('rgbaError.txt', 'w')
f3 = open('ExifError.txt', 'w')
f4 = open('4chImg.txt', 'w')
f5 = open('WebpError.txt', 'w')
f6 = open('UnknownError.txt', 'w')

idx = 0
for r, d, files in os.walk(root):
    if files != []:
        for i in files:
            fp = os.path.join(r, i)
            try:
                img = Image.open(fp)
                if (len(img.split()) != 3):
                    # print('4CH:', fp)
                    f4.write('{}\n'.format(fp))

            except Exception as e:
                print('Error:', str(e))
                print(fp)
                if 'Possibly corrupt EXIF data' in str(e):
                    print('Exif error')
                    f1.write('{}\n'.format(fp))
                elif 'Palette images with Transparency' in str(e):
                    print('rgba error')
                    f2.write('{}\n'.format(fp))
                elif 'Corrupt EXIF data' in str(e):
                    print('pExif error')
                    f3.write('{}\n'.format(fp))
                elif 'image file could not be identified because WEBP' in str(e):
                    print('Webp error')
                    f5.write('{}\n'.format(fp))
                else:
                    print('Unknown error')
                    f6.write('{}\n'.format(fp))

            if idx % 5000 == 0:
                print('=' * 20, idx)

            idx += 1

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()

