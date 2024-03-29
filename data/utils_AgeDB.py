# 将AgeDB——pair.txt转换为webface_test_pair.txt和lfw_test_pair.txt相同的格式
import os
import os.path as osp
import random

origin_path="./AgeDB_pair.txt"
target_path="./AgeDB_test_pair.txt"


def generateFromPair():
    with open(origin_path, 'r') as fd:
        origin_pairs = fd.readlines()
        for pair in origin_pairs :
            temp_pair=pair.split()
            if len(temp_pair)==3 :
                with open(target_path, 'a') as fd:
                    fd.write(f"{temp_pair[0]}/{temp_pair[0]}_{temp_pair[1].zfill(4)}.jpg {temp_pair[0]}/{temp_pair[0]}_{temp_pair[2].zfill(4)}.jpg 1\n")
            if len(temp_pair)==4 :
                with open(target_path, 'a') as fd:
                    fd.write(f"{temp_pair[0]}/{temp_pair[0]}_{temp_pair[1].zfill(4)}.jpg {temp_pair[2]}/{temp_pair[2]}_{temp_pair[3].zfill(4)}.jpg 0\n")                                        


if __name__ == '__main__':

        same_pairs_num = 8000
        diff_pairs_num = 8000
        same_pairs = []
        diff_pairs = []
        train_root="/home/zjb/workbench/data/AgeDB"
        test_pair_txt="/home/zjb/workbench/data/AgeDB_test_pair.txt"

        folder_names = [name for name in os.listdir(train_root) if osp.isdir(osp.join(train_root, name))]
        #print(folder_names)

        for same_count in range(same_pairs_num):
                folder_name = random.choice(folder_names)
                img_names = os.listdir(osp.join(train_root, folder_name))
                if len(img_names) < 2:
                        continue
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
