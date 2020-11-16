import sys
import warnings
from PIL.Image import DecompressionBombWarning
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DecompressionBombWarning)
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import *
from dataload import Dataset,Testset
import time
from models.mymodel import BaseModel,BaseModel1
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms



def test_model():
    model.eval()
    num_steps = len(eval_loader)
    print(f'total batches: {num_steps}')
    preds = []
    image_names = []
    confident=[]
    with torch.no_grad():
        for i, (XI, image_name) in enumerate(tqdm(eval_loader)):
            # if i % 50 == 0:
            #     print(i, i/len(eval_loader))

            x = XI.cuda(device_id)
            output = model(x)
            output = nn.Softmax(dim=1)(output)

            confs, predicts = torch.max(output.detach(), dim=1)
            confident+=list(confs.cpu().numpy() > 0.7)
            preds += list(predicts.cpu().numpy())
            image_names += list(image_name)


    print(len(preds), len(image_names))


    with open(csv_path, 'w') as f:
        f.write('{0},{1}\n'.format('image_name', 'class'))
        for i in tqdm(range(len(preds))):
            # f.write('{0},{1}\n'.format(image_names[i], preds[i]))
            if confident[i]:
                f.write('{0},{1}\n'.format(image_names[i], preds[i]))


if __name__ == '__main__':
    csv_path = './output/efficientnet-b1_cat_1+e2/46.18_up0.7.csv'
    test_batch_size =4
    num_class = 5000
    device_id = 0
    model_name = 'efficientnet-b1'
    # model_path = None
    model_path = 'F:\ACCV_WebFG5000_log\BaseCode.Pytorch-master\output\efficientnet-b1_cat_1+e2\\best_44_acc_0.5592.pth'
    # model = get_efficientnet(model_name=model_name)
    # model, *_ = model_selection(modelname=model_name, num_out_classes=5000, dropout=None)
    model = BaseModel(model_name=model_name, num_classes=num_class, pretrained=1,
                       pool_type='cat', down=1)
    if model_path is not None:
        my_model = torch.load(model_path, map_location='cpu')
        model.load_state_dict(my_model['net'])
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    model = model.cuda(device_id)

    start = time.time()
    epoch_start = 1
    num_epochs = 1

    testset = Testset(root='./data/test_448')
    # testset1 = Dataset('./data/test_448')
    # eval_loader = DataLoader(xdl_test, batch_size=test_batch_size, shuffle=False, num_workers=4)
    eval_loader = DataLoader(dataset=testset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    test_dataset_len = len(testset)
    print('test_dataset_len:', test_dataset_len)
    test_model()
    print('Total time:', time.time() - start)







