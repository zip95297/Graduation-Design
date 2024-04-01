# Description: This file contains the utils to get test_pair from WebFace Dataset
import os
import os.path as osp
import random

train_root="/home/zjb/workbench/data/CASIA-WebFace"
test_pair_txt="/home/zjb/workbench/data/webface_test_pair.txt"

same_pairs_num = 8000
diff_pairs_num = 8000
same_pairs = []
diff_pairs = []


folder_names = [name for name in os.listdir(train_root) if osp.isdir(osp.join(train_root, name))]
#print(folder_names)

for same_count in range(same_pairs_num):
    folder_name = random.choice(folder_names)
    img_names = os.listdir(osp.join(train_root, folder_name))
    img_name = random.sample(img_names,2)
    same_pairs.append(osp.join( folder_name, img_name[0])+" "+osp.join(folder_name, img_name[1])+" 1")

for diff_count in range(diff_pairs_num):
    folder_name1 = random.choice(folder_names)
    folder_name2 = random.choice(folder_names)
    while folder_name1 == folder_name2:
        folder_name2 = random.choice(folder_names)
    img_name1 = random.choice(os.listdir(osp.join(train_root, folder_name1)))
    img_name2 = random.choice(os.listdir(osp.join(train_root, folder_name2)))
    diff_pairs.append(osp.join( folder_name1, img_name1)+" "+osp.join(folder_name2, img_name2)+" 0")

with open(test_pair_txt, 'w') as f:
    for pair in same_pairs:
        f.write(pair+'\n')
    for pair in diff_pairs:
        f.write(pair+'\n')
    f.close()



if __name__ == "__main__":
    
    # 获取当前目录下的所有文件夹名
    folders = [folder for folder in os.listdir('/home/zjb/workbench/data/lfw_as_train')]

    # 对文件夹名进行重新命名
    for i, folder in enumerate(folders):
        files = os.listdir(f"/home/zjb/workbench/data/lfw_as_train/{folder}")
        file_count=1
        for file in files:
            os.rename(f"/home/zjb/workbench/data/lfw_as_train/{folder}/{file}",f"/home/zjb/workbench/data/lfw_as_train/{folder}/{file_count:03d}.jpg" )
            file_count+=1
        folder_new_name = str(f"/home/zjb/workbench/data/lfw_as_train/{i:05d}")
        os.rename(f"/home/zjb/workbench/data/lfw_as_train/{folder}", folder_new_name)

        