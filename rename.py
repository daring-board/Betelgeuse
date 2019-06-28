import os

base_dir = 'train'
for d in os.listdir(base_dir):
    if 'empty' == d: continue
    count = 0
    prefix = d.split('_')[1]
    tmp = base_dir+'/'+d+'/'
    for f in os.listdir(tmp):
        count += 1
        ext = f.split('.')[-1]
        os.rename(tmp+f, tmp+prefix+str(count)+'.'+ext)
        # print(tmp+prefix+str(count)+'.'+ext)