import tensorflow as tf
import data as dataset
import utils
import numpy as np
import time

def inference(config):

    # Load test dataset
    x_test, _, nameList = dataset.read_data(config.PATH_DATASET_TEST,
                                None,
                                (0,0),
                                True)

    N, h, w, c = x_test.shape
    
    if not h == w and N == 0:
        return
    
    modes = [key for (key, value) in config.IS_MODEL.items() if value == True]

    mode_dict = {}
    saver_dict = {}

    tf.reset_default_graph()

    # Create model
    for mode in modes:
        network_model = config.NETWORK_MODEL[mode]
        network_layer_size = config.NETWORK_LAYER_SIZE[mode]

        mode_dict[str(mode)]  = utils.build_model(mode,
                                                network_model,
                                                (h,w),
                                                network_layer_size,
                                                config.NETWORK_FEATURE_SIZE,    
                                                c)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=mode)
        saver_dict[str(mode)] = tf.train.Saver(var_list)
    
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True))) 
    init = tf.global_variables_initializer()
    sess.run(init)

    # Load weights
    for mode in modes:
        checkpoint_path = config.PATH_WEIGHT + '\\%s' % (mode)
        saver_dict[str(mode)].restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        print('>%s Restored!' % (mode))

    # run
    with sess as se:

        for i in range(N):
            image = x_test[i].reshape(1, h, w, c)
            time_start = time.time()

            if modes.count('denoise'):
                image = se.run([mode_dict['denoise'].out], feed_dict = {mode_dict['denoise'].input:image})
                image = image[0]
            if modes.count('deblur'):
                image = se.run([mode_dict['deblur'].out], feed_dict = {mode_dict['deblur'].input:image})
                image = image[0]
            if modes.count('SR'):
                crop_diff = (h//2//2, w//2//2) 
                image = image[:,crop_diff[0]:crop_diff[0] + h//2,crop_diff[1]:crop_diff[1] + w//2, :]
                image = se.run([mode_dict['SR'].out], feed_dict = {mode_dict['SR'].input:image})            
            
            out_images = np.array(image, dtype=np.uint8)
            #_, res_h, res_w, res_c = out_images[0].shape
            #out_images = out_images.reshape(res_h,res_w,res_c)
            out_images = out_images.reshape(h,w,c)

            time_spand = time.time() - time_start
            
            file_name = '{}_Result'.format(nameList[i])
            path = config.PATH_RESULT + '/%s' % "_".join([str(x) for x in modes])
            utils.saveimage(out_images, path, file_name)
            print('> Inference image : %s(%dx%dx%d) [%0.4f]' % (file_name, h, w, c, time_spand))