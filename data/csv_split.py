import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    root = './Image_Classification/'

    df = pd.read_csv('./Image_Classification/train.csv')

    fnames = df['filename'].values
    labels = df['label'].values

    x_train, x_val, y_train, y_val = train_test_split(fnames, labels, test_size=0.2, random_state=2020)

    f = open('train0.txt', 'w')
    for fname, label in zip(x_train, y_train):
        # f.write('{}/{}.jpg,{}\n'.format(root, fname, label))
        f.write('{}/{},{}\n'.format(root, fname, label))
    f.close()

    f = open('val0.txt', 'w')
    for fname, label in zip(x_val, y_val):
        # f.write('{}/{}.jpg,{}\n'.format(root, fname, label))
        f.write('{}/{},{}\n'.format(root, fname, label))

    f.close()
