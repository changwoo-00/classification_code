import os
import sys
import tensorflow as tf
import argparse
import train
import inference
import dataset_generator

class  ConfigInference(object):

    if __name__ == '__main__':
        print("parser")
        parser = argparse.ArgumentParser(
            description='Classification Network')

        parser.add_argument('--mode',
                            default='training',
                            help='training or Inference')
        parser.add_argument('--dataset_name', # "New Dataset Generation"으로 생성한 Dataset 폴더 이름.
                            default='TEST', 
                            type=str)
        parser.add_argument('--pj_name', # 동일 Data set 하
                            default='128_train_set_test_2', 
                            type=str)
        parser.add_argument('--cutmix',
                            default=False,
                            type=bool)
        #=============== Args For New Dataset Generation ================
        # Data set folder(--data_path) => Train folder, Validation folder, Dataset_Train.txt, Dataset_Valid.txt 생성
        # -> --dataset_name 
        parser.add_argument('--generate_dataset',
                            default=None)
        parser.add_argument('--data_path',  # Data set folder (이미지가 저장된 폴더)
                            default="D:\\Dataset\\[Project] HKC H5 SAMPLE TEST IMAGE\\Dataset_v1\\", 
                            type=str)
        parser.add_argument('--val_split',
                            default=0.25,
                            type=float)
        #==================================================================
        args = parser.parse_args()
        

        if args.generate_dataset != None:
            g_dataset = dataset_generator.Generate_dataset(args.data_path, args.dataset_name, args.val_split)
            g_dataset.Makedir()
            g_dataset.generate_dataset()

        # Configurations
        if args.mode == 'training':
            model = train.ModelTraining(
                mode=6,
                crop_size=224,
                batch_size=8,
                epoch=200,
                checkpoint_continue=0,
                learningRate=0.001,
                learningRateDecayFactor=0.9,
                args=args);

            model.setDataAugmentationRotation(False, False, False, False)
            model.setDataAugmentationGaussianBlur(False, 5, 0.5, 1.5)
            model.setDataAugmentationGaussianNoise(False, 0, 1)
            #model.setConfig(2)
            model.train()
    

    
