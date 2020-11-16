import os
from PIL import Image
import cv2
import shutil

root = './train'
save_path = './thumbnail'
for r, d, files in os.walk(root):
    if files != []:
        for i in files:
            fp = os.path.join(r, i)
            label = i.split('_')[0]
            dst = os.path.join(save_path, label)
            if not os.path.exists(dst):
                os.makedirs(dst)

            img = Image.open(fp).convert('RGB')
            w, h = img.size
            if max(w, h) > 256:
                img.thumbnail((256, 256), Image.ANTIALIAS)
                img.save(os.path.join(dst, i), quality=95, subsampling=0)
            else:
                shutil.copy(fp, os.path.join(dst, i))



#原数据由于尺寸不一，多数是高清图片，训练时resize会很耗时，因此先resize到一个小尺寸保存起来。
# Image.thumbnail()可以起到过滤的作用，如果hw在范围内就不会resize，超过就会按比例放缩。
#处理前数据集大小为114G，处理后为86G。在 Tesla V100 32GB*2 硬件环境下，训练Baseline，处理前训练时间一个epoch约为2400s(40min)，
# 处理后一个epoch约1400s(23min)，极大缩小了训练时间，精度应该没有什么影响，调小判别尺寸应该还能更快，毕竟训练数据尺寸是224x224。