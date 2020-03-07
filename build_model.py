
import tensorflow as tf


import classification_model_collection as CMC

def build_model(mode_name, model_name, image_size, batch_size, image_num, class_num, root_pretrain, learning_rate, learning_rate_decay_factor, input_channels, output_channels):
    
    tf.reset_default_graph()

    #if mode_name == 'DM' and model_name == 'DMCNN':
    #    model = DN.ResidualNet_DN(image_size,
    #                                layers_num,
    #                                features_size,
    #                                input_channels,
    #                                output_channels,
    #                                mode_name)

    #elif mode_name == 'SR' and model_name == 'EDSR':
    #    model = SR.ResidualNet_SR(image_size,
    #                                layers_num,
    #                                features_size,
    #                                input_channels,
    #                                output_channels,
    #                                sr_scale,
    #                                mode_name)
    if mode_name == 'CLASSIFICATION':
        model = CMC.ClassificationModelCollection(model_name,
                                                  image_size,
                                                  image_num,
                                                  class_num,
                                                  batch_size,
                                                  root_pretrain,
                                                  learning_rate,
                                                  learning_rate_decay_factor)
    else:
        raise ValueError('Error: the model is not available.')

    print('build model : mode={}, model={}'.format(mode_name, model_name))
    return model