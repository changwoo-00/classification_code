

import os
import sys
import tensorflow as tf
import argparse
import train
import inference
from sklearn.model_selection import train_test_split
import shutil

class  Generate_dataset(object):
    def __init__(self, data_path, dataset_name, val_split):
        print('hello')
        self.data_path = data_path
        self.class_list = os.listdir(self.data_path)
        self.class_num = len(self.class_list)
        self.class_path_list = [data_path + i + '\\' for i in self.class_list]
        self.dataset_name = dataset_name
        self.val_split = val_split
        print('hello')

    def Makedir(self):
        if not os.path.exists('D:\\HDL\\Project\\Classification\\'+self.dataset_name):
            os.mkdir('D:\\HDL\\Project\\Classification\\'+self.dataset_name)
            os.mkdir('D:\\HDL\\Project\\Classification\\'+self.dataset_name+'\\DataSet')
            os.mkdir('D:\\HDL\\Project\\Classification\\'+self.dataset_name+'\\DataSet\\Test')
            os.mkdir('D:\\HDL\\Project\\Classification\\'+self.dataset_name+'\\DataSet\\Train')
            os.mkdir('D:\\HDL\\Project\\Classification\\'+self.dataset_name+'\\DataSet\\Validation')
        else:
            print('Project exists!')

    def generate_dataset(self):
        f = open('D:\\HDL\\Project\\Classification\\'+self.dataset_name+"\\DataSet\\ClassList.txt","w")
        for i in range(len(self.class_list)):
            f.write(self.class_list[i] + '\t 11111'+str(i)+'\n')
        f.close()
        

        for i in range(self.class_num):     # 각 class에 대해 작업
            img_list = os.listdir(self.class_path_list[i])
            
            # Train Validation Split
            train_list, val_list = train_test_split(img_list, test_size=self.val_split, random_state=42)

            #이미지를 PJ\Dataset 폴더로 이동
            train_list_path = [self.class_path_list[i] + k for k in train_list]
            val_list_path = [self.class_path_list[i] + k for k in val_list]
            for j in range(len(train_list_path)):
                shutil.copyfile(train_list_path[j], 'D:\\HDL\\Project\\Classification\\'+self.dataset_name+'\\DataSet\\Train\\'+train_list[j])
            for j in range(len(val_list_path)):
                shutil.copyfile(val_list_path[j], 'D:\\HDL\\Project\\Classification\\'+self.dataset_name+'\\DataSet\\Validation\\'+val_list[j])

            # txt file update
            f = open('D:\\HDL\\Project\\Classification\\'+self.dataset_name+"\\DataSet\\Dataset_Train.txt","a")
            for j in range(len(train_list)):
                f.write(train_list[j]+",1,"+str(i)+",\n")
            f.close()
            f = open('D:\\HDL\\Project\\Classification\\'+self.dataset_name+"\\DataSet\\Dataset_Valid.txt","a")
            for j in range(len(val_list)):
                f.write(val_list[j]+",1,"+str(i)+",\n")

    
