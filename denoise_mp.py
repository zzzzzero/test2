import cv2
import os
import numpy as np
import shutil
from multiprocessing import Pool
def calculate(image1, image2):
    # 灰度直方图算法
    # 计算单通道的直方图的相似值
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                     (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

def classify_hist_with_split(image1, image2):
    # RGB每个通道的直方图相似度
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data

def cal_transfer_sim(img):
    test=cv2.imread(img)
    gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    gray=gray[:,:,np.newaxis]
    image =gray.repeat([3],axis=2)
    sim = classify_hist_with_split(test, image)
    return sim

# fpath=[]
# for mode in ['train_256使用时删去','val_256使用时删去']:
#     txt = './data/%s.txt' % mode
#     with open(txt, 'r')as f:
#         for i in f.readlines():
#             fp, label = i.strip().split(',')
#             fpath.append(fp)
dataroot='./data'
def worker(fpath):
    for i in range(len(fpath)):
        try:
            fp = os.path.join(dataroot, fpath[i])
            sim=cal_transfer_sim(fp)
            if sim>0.915:
                shutil.move(fp,'./data/noise256_0.915/noise/')

        except:
            print(fp)


def main():
    fpath = []

    for mode in ['train_256使用时删去', 'val_256使用时删去']:
        txt = './data/%s.txt' % mode
        with open(txt, 'r')as f:
            for i in f.readlines():
                fp, label = i.strip().split(',')
                fpath.append(fp)
    length=len(fpath)

    print("主进程开始执行>>> pid={}".format(os.getpid()))
    ps =Pool(6)
    step=length//6
    for i in range(6):
        st=i*step
        end=(i+1)*step
        ps.apply_async(worker, args=(fpath[st:end],))  # 异步执行

    # 关闭进程池，停止接受其它进程
    ps.close()
    # 阻塞进程
    ps.join()
    print("主进程终止")

if __name__ == '__main__':
    main()