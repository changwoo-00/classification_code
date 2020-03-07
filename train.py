
import tensorflow as tf
import data as dataset
import build_model
import utils
import math
import numpy as np
import cv2
import os
import datetime

import sys
import numpy as np
import queue
import time
from random import shuffle


class ModelTraining(object):
    def __init__(self, mode=6, crop_size=256, batch_size=16, epoch=100, start_epoch=0, learningRate=0.0001, learningRateDecayFactor=0.9, checkpoint_continue=0, transfer_learning=0, args=None):
       
        self.IS_MODEL               = {'CLASSIFICATION':False}
        self.NETWORK_MODEL          = {'CLASSIFICATION':'mobilenet_v2'} # resnet_v2_50, inception_resnet_v2, mobilenet_v2


        self.PATH_PROJECT            = 'D:/HDL/Project/Classification/'+args.dataset_name
        self.PATH_DATASET_IMAGE      = self.PATH_PROJECT + '/DataSet/Train'
        self.PATH_DATASET_VALIDATION = self.PATH_PROJECT + '/DataSet/Validation'
        self.PATH_CHECKPOINT         = self.PATH_PROJECT + '/Checkpoint/'
        self.PATH_CHECKPOINT_SAVE    = self.PATH_PROJECT + '/Checkpoint/' + args.pj_name + '/'
        self.PATH_RESULT             = 'D:/Dataset/Result/'

        # only classification
        self.FILE_DATASET_IMAGE_LIST = self.PATH_PROJECT + '/DataSet/Dataset_Train.txt'
        self.FILE_DATASET_VALID_LIST = self.PATH_PROJECT + '/DataSet/Dataset_Valid.txt'
        self.FILE_DATASET_CLASS_LIST = self.PATH_PROJECT + '/DataSet/ClassList.txt'
        self.PATH_PRETRAIN           = 'D:/HDL/System/'

        # HYPER-PARAMETER
        self.HYPERP_LEARNING_RATE    = learningRate
        self.HYPERP_DECAY_FACTOR     = learningRateDecayFactor
        self.HYPERP_ADAM_BETA        = 0.9
        self.HYPERP_EPOCH            = epoch
        self.HYPERP_EPOCH_START      = start_epoch
        self.BATCH_SIZE              = batch_size

        # ETC 
        self.CHECKPOINT_CONTINUE     = checkpoint_continue
        self.TRANSFER_LEARNING       = transfer_learning
        self.TRAINING_STOP           = False


        # Dataset
        self.DATASET_CROP_SIZE       = crop_size
        self.DATASET_CROP            = True
        self.DATASET_CROP_COUNT      = 10  
        self.DATASET_SAVE_EXTENSION  = 'bmp'

        self.IS_MODEL['CLASSIFICATION'] = True

        self.cutmix = args.cutmix


    def setDataAugmentationRotation(self, enable_CW90, enable_CCW90, enable_h_flip, enable_v_flip):
        self.AUG_ROTATION            = dict(enable_CW90=enable_CW90, enable_CCW90=enable_CCW90, enable_h_flip=enable_h_flip, enable_v_flip=enable_v_flip)

    def setDataAugmentationGaussianBlur(self, enable, max_kernel_size, min_sigma, max_sigma):
        self.AUG_BLUR_GAUSSIAN       = dict(enable=enable, max_kernel_size=max_kernel_size, min_sigma=min_sigma, max_sigma=max_sigma)

    def setDataAugmentationGaussianNoise(self, enable, mean, var):
        self.AUG_NOISE_GAUSSIAN      = dict(enable=enable, mean=mean, var=var)


    def _loadDataSetClassification(self):
        class_names = dataset.read_class_txt(self.FILE_DATASET_CLASS_LIST)
        class_num = len(class_names)

        image_list_path, image_list_train_labels = dataset.read_image_path_txt(self.FILE_DATASET_IMAGE_LIST, class_num)
        image_train_list = dataset.read_list_data(self.PATH_DATASET_IMAGE,   # image folder path
                                                    image_list_path,         # image name list
                                                    (self.DATASET_CROP_SIZE, self.DATASET_CROP_SIZE),
                                                    self.DATASET_CROP,
                                                    True)
        train_datset_count = len(image_train_list)
        if train_datset_count == 0:
            print('Load fail a dataset : Count = {}'.format(train_datset_count))

        channel = 1
        
        if len(image_train_list[0].shape) == 3:
            train_dataset_height, train_dataset_width, channel = image_train_list[0].shape
        else:
            train_dataset_height, train_dataset_width = image_train_list[0].shape[:2]
        print('Read dataset(input) - train : count = {}, width = {}, height = {} / class number = {}'.format(train_datset_count, train_dataset_width, train_dataset_height, class_num))

        image_list_path, image_list_valid_labels = dataset.read_image_path_txt(self.FILE_DATASET_VALID_LIST, class_num)
        image_valid_list = dataset.read_list_data(self.PATH_DATASET_VALIDATION,
                                                    image_list_path,
                                                    (self.DATASET_CROP_SIZE, self.DATASET_CROP_SIZE),
                                                    self.DATASET_CROP,
                                                    True)

        valid_dataset_height = 0
        valid_dataset_width = 0
        valid_datset_count = len(image_valid_list)
        if valid_datset_count != 0:
            valid_dataset_height, valid_dataset_width = image_valid_list[0].shape[:2]
        print('Read dataset(input) - validation : count = {}, width = {}, height = {}'.format(valid_datset_count, valid_dataset_width, valid_dataset_height))

        return image_train_list, image_list_train_labels, image_valid_list, image_list_valid_labels, class_names

    def _loadCheckpoint(self, session, saver):
        if not os.path.exists(self.PATH_CHECKPOINT):
            os.makedirs(self.PATH_CHECKPOINT)
        if not os.path.exists(self.PATH_CHECKPOINT_SAVE):
            os.makedirs(self.PATH_CHECKPOINT_SAVE)
        print('Checkpoint path : {}'.format(self.PATH_CHECKPOINT))

        if self.CHECKPOINT_CONTINUE:
            saver.restore(session, tf.train.latest_checkpoint(self.PATH_CHECKPOINT_SAVE))
            print('Loaded latest model checkpoint.')
        elif self.TRANSFER_LEARNING:
            ret = tf.train.latest_checkpoint(self.PATH_CHECKPOINT_SAVE)
            if not ret == None:
                saver.restore(session, ret)
                print('Loaded transfer learning model checkpoint.')


    def _print_train_info(self, cur_epoch, total_epoch, cur_iterator, total_iterator, **dict):
        text = ''
        for key, value in dict.items():
            answer = key[0]
            for idx, char in enumerate(key[1:]):
                if key[idx].islower() and char.isupper():
                    answer += ' '
                answer += char
            if not value == None:
                text += '{0:} = {1:3.2f} '.format(answer, value)
        print('Epoch [{0:2d}/{1:2d}], Batch [{2:4d}/{3:4d}] : {4:}'.format((cur_epoch + 1), total_epoch, (cur_iterator+1), total_iterator, text))

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def cutmix(self, batch_img, batch_label, class_num, c_size):
        #print('cutmix')
        batch_img_s, batch_label_s = batch_img.copy(), batch_label.copy()
        c = list(zip(batch_img_s, batch_label_s))
        shuffle(c)
        batch_img_s, batch_label_s = zip(*c)
        img_W = np.shape(batch_img_s)[1]
        img_H = np.shape(batch_img_s)[2]
        #r_w = np.random.randint(np.floor(img_W/3),img_W-1)
        #r_h = np.random.randint(np.floor(img_H/3),img_H-1)
        c_size_min = 30
        c_size = np.max([c_size_min,c_size])
        r_w = np.random.randint(c_size_min,np.min([c_size,127]))
        r_h = np.random.randint(c_size_min,np.min([c_size,127]))
        r_x = np.random.randint(0,img_W-r_w-1)
        r_y = np.random.randint(0,img_H-r_h-1)
        ratio_lambda = 1 - (r_w*r_h)/(img_W*img_H)
        for i in range(len(batch_label)):
            batch_img[i][r_x:r_x+r_w,r_y:r_y+r_h,:] = batch_img_s[i][r_x:r_x+r_w,r_y:r_y+r_h,:]
        
        batch_label = [i*ratio_lambda for i in batch_label]
        batch_label_s = [i*(1-ratio_lambda) for i in batch_label_s]
        batch_label = [sum(i) for i in zip(batch_label,batch_label_s)]
        return batch_img, batch_label
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # train #
    def train(self):
        start_time = time.time()
        mode = [key for (key, value) in self.IS_MODEL.items() if value == True]
        if len(mode) > 1 or len(mode) == 0:
            print("You can't select more than one mode.")
            return

        x_train, y_train, x_valid, y_valid, input_size = 0, 0, 0, 0, (0,0)
        channel = 3
        class_names = []

        mode = mode[0]
        network_model = self.NETWORK_MODEL[mode]
        
        # load dataset(Train, Validation)
        
        x_train, y_train, x_valid, y_valid, class_names = self._loadDataSetClassification()
        self.DATASET_CROP_COUNT = 1
        

        class_num = len(class_names)
        train_image_num = len(x_train)
        
        #__________________________________________________________________________ create model __________________________________________________________________________#
        model = build_model.build_model(mode,
                            network_model,
                            (self.DATASET_CROP_SIZE, self.DATASET_CROP_SIZE),
                            self.BATCH_SIZE, 
                            train_image_num,
                            class_num,
                            self.PATH_PRETRAIN,
                            self.HYPERP_LEARNING_RATE,
                            self.HYPERP_DECAY_FACTOR,
                            channel,
                            channel)


        #__________________________________________________________________________ Training __________________________________________________________________________#
        # Train
        print('Starting at Epoch = {}, learning rate = {}'.format(self.HYPERP_EPOCH, self.HYPERP_LEARNING_RATE))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        
        deviceList = ['/gpu:0','/gpu:1']
        GPU_index = 0
        
        with tf.Session(config=config) as sess:
            with tf.device(deviceList[GPU_index]):
                # Initialize all variables
                #tf.global_variables_initializer().run()
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                best_loss = 999999.0
                totallCnt = 0
                loss_list = []

                # checkpoint & pretrain checkpoit load
                if not os.path.exists(self.PATH_CHECKPOINT):
                    os.makedirs(self.PATH_CHECKPOINT)
                if not os.path.exists(self.PATH_CHECKPOINT_SAVE):
                    os.makedirs(self.PATH_CHECKPOINT_SAVE)

                if mode == 'CLASSIFICATION':
                    model.restore_fn(sess)
                elif self.CHECKPOINT_CONTINUE == True:
                    model.restore_fn(sess, self.PATH_CHECKPOINT_SAVE)

                # Data generate - Validation
                input_valid_images, label_valid_images = dataset.dataset_generator(x_valid, self.DATASET_CROP_SIZE, self.DATASET_CROP_COUNT)
                input_valid_count = len(input_valid_images)

                for e in range(self.HYPERP_EPOCH_START, self.HYPERP_EPOCH):
                    if self.TRAINING_STOP:
                        break;


                    input_images = x_train.copy()
                    label_images = y_train
                    input_count = len(input_images)
                    label_count = 0
                    label_valid_images = y_valid


                    start = time.time() 
                    # Data Augmentation
                    dataset.dataAugmentation_run_thread(mode=mode,
                                                    dict_rotation=self.AUG_ROTATION,
                                                    dict_blur_gaussian=self.AUG_BLUR_GAUSSIAN,
                                                    dict_noise_gaussian=self.AUG_NOISE_GAUSSIAN,
                                                    input_images=input_images,
                                                    label_images=label_images)
                    print("time :", time.time() - start)

                    print('Data generator : Input image count = {}, Label image count = {}'.format(input_count, label_count))

                    if input_count < self.BATCH_SIZE:
                        self.BATCH_SIZE = input_count

                    train_data = dataset.DataSet(input_images, label_images)
                    if input_valid_count > 0:
                        valid_data = dataset.DataSet(input_valid_images, label_valid_images)

                    batch_total = math.ceil(input_count / self.BATCH_SIZE)
                    print('Total Batch = {}, Batch size = {}'.format(batch_total, self.BATCH_SIZE))

                    for b in range(batch_total):
                        if self.TRAINING_STOP:
                            print('Train Stop...!!!')
                            break

                        batch_images, batch_labels = train_data.next_batch(self.BATCH_SIZE)

                        batch_labels_num = batch_labels.copy()

                        ##### cutmix #####

                        ## labels to one hot encoding ##
                        batch_one_hot_list = []
                        for i in range(len(batch_labels)):
                            a = np.zeros(class_num)
                            a[batch_labels[i]] = 1
                            batch_one_hot_list.append(list(a))
                        batch_labels = batch_one_hot_list
                        #################################
                        #c_size = np.ceil(96*(e+1)/self.HYPERP_EPOCH) increasing window
                        c_size = 35 + e
                        if self.cutmix == True:
                            if np.random.uniform(0,1) <= 1:
                                batch_images, batch_labels = self.cutmix(batch_images, batch_labels, class_num, c_size)
                            print('batch_labels')
                            print(batch_labels)    
                        
                        ##################

                        loss, predicted_class, cur_learning_rate, acc = model.train_step(sess, batch_images, batch_labels)

                        acc = acc[1]

                        if best_loss > loss:
                            best_loss = loss

                        if b % (batch_total // 5) == 0 :
                            self._print_train_info(e, self.HYPERP_EPOCH, b, batch_total, Loss=loss, BestLoss=best_loss, CurrentLearningRate=cur_learning_rate)

                            print('chart=0,{0:},{1:}'.format(totallCnt,loss))
                            loss_list.append(loss)
                            totallCnt += 1
                            #acc = model.train_step_accuracy(sess, batch_images, batch_labels)
                            #print('accuracy:{}'.format(acc))
                        
                        if b == batch_total - 1:
                            # save model checkpoint
                            checkpoint_save_path = "%s/%04d-%04d"%(self.PATH_CHECKPOINT_SAVE,e,b)
                            if (e+1)%10 == 0:
                                model.save_checkpoint(sess, checkpoint_save_path, mode)
                                model.save_checkpoint(sess, self.PATH_CHECKPOINT_SAVE, mode)
                                print('Saved the checkpoint.(path:{})'.format(checkpoint_save_path))

                            ## input image save
                            #model.save_image(utils, checkpoint_save_path, batch_images, batch_labels_num, output_image, class_names, predicted_class, 'Train') #hcw, cutmix
                            #print('[Weight] Update.')
                            #loss_list.append(loss)

                            #self._SaveTrainInfoTextFile(checkpoint_save_path, e, loss, best_loss) @ hcw, cutmix

                    ## validation  # hcw, cutmix 
                    #valid_acc_list = []
                    #batch_valid_total = math.ceil(input_valid_count / self.BATCH_SIZE);
                    #for v in range(batch_valid_total):
                    #    if self.TRAINING_STOP:
                    #        break;

                    #    batch_valid_images, batch_valid_labels = valid_data.next_batch(self.BATCH_SIZE)

                    #    ###################################
                    #    batch_one_hot_list = []
                    #    for i in range(len(batch_valid_labels)):
                    #        a = np.zeros(class_num)
                    #        a[batch_labels[i]] = 1
                    #        batch_one_hot_list.append(a)
                    #    batch_valid_labels = batch_one_hot_list
                    #    ######################################

                    #    loss, predicted_valid_class, acc = model.validation_step(sess, batch_valid_images, batch_valid_labels)

                    #    print(acc[1])
                    #    valid_acc_list.append(acc[1])

                    #    if v == batch_valid_total - 1:
                    #        valid_acc_avg = np.mean(valid_acc_list)
                    #        #model.save_image(utils, checkpoint_save_path, batch_valid_images, batch_valid_labels, output_valid_image, class_names, predicted_valid_class, 'Validation') #hcw, cutmix
                    #        print('[Weight] Update.')
                    #        print("Validation Accuracy : ")
                    #        print(valid_acc_avg)

        # txt file save
        #self._SaveTrainInfoTextFile(self.PATH_CHECKPOINT_SAVE, e, 0.0, best_loss) # hcw, cutmix
        #self._SaveLossTextFile(self.PATH_CHECKPOINT_SAVE, loss_list) # hcw, cutmix
        end_time = time.time()
        print('Train Time :', end_time-start_time)
        time.sleep(1)
        print('\n')

        ##print_EOT()
        ##self.flush_and_exit()
        ##self.join()
