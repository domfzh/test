import random
import glob
import cv2
from cv2 import imshow
from cv2 import imwrite
import numpy as np
import shutil
from shutil import copy2
import os
import cv2 


class SPLIT():
    def __init__(self,img_set,is_train=True,batch_size=10):
        self.img_set=img_set

    def split_datasset(self,target_path,train_scale=0.8,val_scale=0.1,test_scale=0.1):
        print('-----开始划分-----')
        class_names=os.listdir(self.img_set)
        spilt_names=['train','val','test']
        for spilt_name in spilt_names:
            spilt_path=os.path.join(target_path,spilt_name)
            if os.path.isdir(spilt_path):
                pass
            else:
                os.makedirs(spilt_path)
            

        for class_name in class_names:
            current_all_data=glob.glob(os.path.join(self.img_set,class_name)+'/*.jpg')
            current_dirs_len=len(current_all_data)
            current_index_list = list(range(current_dirs_len))
            random.shuffle(current_index_list)

            train_folder_HR=os.path.join(target_path,'train','HR')
            val_folder_HR=os.path.join(target_path,'val','HR')
            test_folder=os.path.join(target_path,'test')

            train_folder_LR_X4=os.path.join(target_path,'train','train_LR','X4')
            train_folder_LR_X3=os.path.join(target_path,'train','train_LR','X3')
            train_folder_LR_X2=os.path.join(target_path,'train','train_LR','X2')

            val_folder_LR_X4=os.path.join(target_path,'val','LR','X4')
            val_folder_LR_X3=os.path.join(target_path,'val','train_LR','X3')
            val_folder_LR_X2=os.path.join(target_path,'val','train_LR','X2')

            if os.path.isdir(train_folder_HR):
                pass
            else:
                os.makedirs(train_folder_HR)

            if os.path.isdir(train_folder_LR_X4):
                pass
            else:
                os.makedirs(train_folder_LR_X4)

            if os.path.isdir(train_folder_LR_X3):
                pass
            else:
                os.makedirs(train_folder_LR_X3)

            if os.path.isdir(train_folder_LR_X2):
                pass
            else:
                os.makedirs(train_folder_LR_X2)

            if os.path.isdir(val_folder_HR):
                pass
            else:
                os.makedirs(val_folder_HR)

            if os.path.isdir(val_folder_LR_X4):
                pass
            else:
                os.makedirs(val_folder_LR_X4)



            if os.path.isdir(val_folder_LR_X3):
                pass
            else:
                os.makedirs(val_folder_LR_X3)

            if os.path.isdir(val_folder_LR_X2):
                pass
            else:
                os.makedirs(val_folder_LR_X2)

            train_stop_flag=current_dirs_len*train_scale
            val_stop_flag=current_dirs_len*(train_scale+val_scale)

            current_idx=0
            train_num=0
            val_num=0
            test_num=0
            for i in current_index_list:
                src_img_path=current_all_data[i]
                img=cv2.imread(src_img_path)

                img_HR=cv2.resize(img,(240,240))
                H,W=img_HR.shape[:2]
                img_LR_X4=cv2.resize(img,(int(H/4),int(W/4)))
                img_LR_X3=cv2.resize(img,(int(H/3),int(W/3)))
                img_LR_X2=cv2.resize(img,(int(H/2),int(W/2)))


                if current_idx<=train_stop_flag:

                    cv2.imwrite(train_folder_HR+'\\'+str(train_num)+'.jpg',img_HR)
                    cv2.imwrite(train_folder_LR_X4+'\\'+str(train_num)+'.jpg',img_LR_X4)
                    cv2.imwrite(train_folder_LR_X3+'\\'+str(train_num)+'.jpg',img_LR_X3)
                    cv2.imwrite(train_folder_LR_X2+'\\'+str(train_num)+'.jpg',img_LR_X2)
                    train_num+=1

                elif (current_idx>train_stop_flag) and (current_idx<=val_stop_flag):

                    cv2.imwrite(val_folder_HR+'\\'+str(val_num)+'.jpg',img_HR)
                    cv2.imwrite(val_folder_LR_X4+'\\'+str(val_num)+'.jpg',img_LR_X4)
                    cv2.imwrite(val_folder_LR_X3+'\\'+str(val_num)+'.jpg',img_LR_X3)
                    cv2.imwrite(val_folder_LR_X2+'\\'+str(val_num)+'.jpg',img_LR_X2)
                    val_num+=1
                    
                else:
                    copy2(src_img_path,test_folder)
                    test_num+=1
                
                current_idx+=1
                
        
        print("{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale, current_dirs_len))
        print("训练集{}：{}张".format(train_folder_HR, train_num))
        print("验证集{}：{}张".format(val_folder_HR, val_num))
        print("测试集{}：{}张".format(test_folder, test_num))
        print('-----结束划分-----')
    
target_path='F:\KEY\FINAL\example\MY-SR'

split=SPLIT('F:\KEY\FINAL\example\MY-SR\CASIA-WebFace')
split.split_datasset(target_path)
