import torch
from models.model import BaseModel
import os
import pandas as pd
import numpy as np
import argparse
from dataload import Testset
from torch.utils.data import DataLoader
import time


def get_setting(path):
    args = {}
    with open(os.path.join(path, 'setting.txt'), 'r')as f:
        for i in f.readlines():
            k, v = i.strip().split(':')
            args[k] = v
    return args


def load_pretrained_model(path, model, mode='best'):
    print('load pretrained model...')
    state = torch.load(os.path.join(path, '%s.pth' % mode))
    print('best_epoch:{}, best_acc:{}'.format(state['epoch'], state['acc']))
    model.load_state_dict(state['net'])


if __name__ == '__main__':
    mode = 'best'

    parser = argparse.ArgumentParser()
    parser.add_argument('--savepath', default='./Base224L2/eff-b3', type=str)
    parser.add_argument('--last', action='store_true')
    args = parser.parse_args()

    path = args.savepath
    if args.last:
        mode = 'last'

    args = get_setting(path)
    # print(args)

    # model
    model = BaseModel(model_name=args['model_name'], num_classes=int(args['num_classes']), \
                    pretrained=int(args['pretrained']), pool_type=args['pool_type'], down=int(args['down']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = model.to(device)
    load_pretrained_model(path, model, mode=mode)

    # data
    testset = Testset(root='./data/test')
    testloader = DataLoader(dataset=testset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    submit = {}
    TTA_times = 10

    fnames = []

    model.eval()
    with torch.no_grad():
        results = np.zeros((len(testset), 9691))
        for n in range(TTA_times):
            print('{:>3}/{:>3}'.format(n, TTA_times))
            t1 = time.time()
            preds = []
            for idx, (data, fname) in enumerate(testloader):
                if n == 0:
                    fnames.extend(fname)
                print(idx, end=',')
                data = data.to(device)

                out = model(data)
                _, pred = torch.max(out, dim=1)
                preds.extend(pred.cpu().tolist())
            t2 = time.time()
            print('Take {:.2f}s'.format(t2 - t1))
            for i, j in enumerate(preds):
                results[i, j] += 1

    submit['name'] = fnames
    submit['class'] = np.argmax(results, axis=1)

    df = pd.DataFrame(submit)
    df.to_csv(os.path.join(path, 'submit.csv'), encoding='utf-8', index=False)