import os
import shutil

path='./data/noise256_0.915'

with open('noise.txt','a') as n:
    for i in os.listdir(path):
        src='./data/train256/' + i[:4] + '/' + i
        dst='./data/noise256/' + i
        n.writelines('./data/train256/'+i[:4]+'/'+i+'\n')
        if not os.path.exists(src):
            print(src)
            continue
        shutil.move(src,dst)




