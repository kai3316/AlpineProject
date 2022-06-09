import os
data_dir = '/home/kai/Desktop/RGB_noflip/'
NewDepth_dir = '/home/kai/Desktop/depth_noflip/'
NewLabel_dir = '/home/kai/Desktop/label_noflip/'
depth_dir = '/home/kai/Desktop/Depth/'
label_dir = '/home/kai/Desktop/yoliclabel/'
alist = os.listdir(data_dir)
# create depth_dir and label_dir
if not os.path.exists(NewDepth_dir):
    os.mkdir(NewDepth_dir)
if not os.path.exists(NewLabel_dir):
    os.mkdir(NewLabel_dir)
for i in alist:
    src = os.path.join(depth_dir, i[:-4] + '.png')
    dst = os.path.join(NewDepth_dir, i[:-4] + '.png')
    os.system('cp ' + src + ' ' + dst)
    # for label
    src = os.path.join(label_dir, i[:-4] + '.txt')
    dst = os.path.join(NewLabel_dir, i[:-4] + '.txt')
    os.system('cp ' + src + ' ' + dst)
print('done')