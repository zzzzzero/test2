
#个人习惯把路径存到txt再在dataset加载。
from sklearn.model_selection import train_test_split
import os

if __name__ == '__main__':
    root = './train448'

    fpath = []
    labels = []
    for d in os.listdir(root):
        fd = os.path.join(root, d)
        label = int(d)
        for i in os.listdir(fd):
            fp = os.path.join(fd, i)
            fpath.append(fp)
            labels.append(label)

    print(len(fpath), len(labels))

    x_train, x_val, y_train, y_val = train_test_split(fpath, labels, random_state=2020, test_size=0.1)
    print(len(x_train), len(x_val))

    with open('train_noise.txt', 'w')as f:
        for fn, l in zip(x_train, y_train):
            f.write('{},{}\n'.format(fn, l))

    with open('val_noise.txt', 'w')as f:
        for fn, l in zip(x_val, y_val):
            f.write('{},{}\n'.format(fn, l))
