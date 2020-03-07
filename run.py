

import os
import sys
import tensorflow as tf
import argparse
import train
import inference
import dataset

class  ConfigInference(object):

    if __name__ == '__main__':
        print("parser")
        parser = argparse.ArgumentParser(
            description='Image Enhance Network(Denoise, Deblur, Super Resolution)')

        parser.add_argument('--mode',
                            default='training',
                            help='training or Inference')
        parser.add_argument('--pj_name', 
                            default='HKC_3class_dataset_v1', 
                            type=str)
        parser.add_argument('--data_path',
                            default="D:\\Dataset\\[Project] HKC H5 SAMPLE TEST IMAGE\\Dataset_v1\\", 
                            type=str)
        parser.add_argument('--generate_dataset',
                            default=None)
        parser.add_argument('--val_split',
                            default=0.25,
                            type=float)
        parser.add_argument('--cutmix',
                            default=False,
                            type=bool)
        args = parser.parse_args()
        

    # DEFAULT = 0,
	# DENOISE = 1,
	# DEBLUR = 2,
	# SR = 3,
	# COLORIZATION = 4,
	# PAD = 5,
	# CLASSIFICATION = 6,
	# SEGMENTATION = 7
        if args.generate_dataset != None:
            g_dataset = dataset.Generate_dataset(args.data_path, args.pj_name, args.val_split)
            g_dataset.Makedir()
            g_dataset.generate_dataset()

        # Configurations
        if args.mode == 'training':
            model = train.ModelTraining(
                mode=6,
                layer_size=3,
                network_feature_size=32,
                crop_size=299,
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
    

    
