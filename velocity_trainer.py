from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
from keras.optimizers import Adam
from pdb import set_trace as st
import numpy as np
import tensorflow as tf
import cv2
#tf.python.control_flow_ops = tf

class VelocityTrainer:
    def preprocess_train(self,ground_truth):
        print("Preprocess data for velocity trainer...")
        image_train, velocity_train = [], []
        [im_h, im_w, im_c] = ground_truth[0]['velocity_im_diff'].shape
        for file in ground_truth:
            image_train.append(file['velocity_im_diff'])
            velocity_train.append(file['true_velocity'])
        image_train = np.reshape(image_train,(len(ground_truth),im_h,im_w,im_c))
        velocity_train = np.reshape(velocity_train,(len(ground_truth),1))

        return image_train, velocity_train

    def preprocess_test(self,im_diff):
        '''Input: a (60,220,3) numpy array'''
        return np.reshape(im_diff,(1,im_diff.shape[0],im_diff.shape[1],im_diff.shape[2]))

    def nvidia_model(self,N_img_height, N_img_width, N_img_channels):
        inputShape = (N_img_height, N_img_width, N_img_channels)
        model = Sequential()
        # normalization
        model.add(Lambda(lambda x: x/ 127.5 - 1, input_shape = inputShape))
        model.add(Convolution2D(24, 5, 5, 
                                subsample=(2,2), 
                                border_mode = 'valid',
                                init = 'he_normal',
                                name = 'conv1'))        
        model.add(ELU())    
        model.add(Convolution2D(36, 5, 5, 
                                subsample=(2,2), 
                                border_mode = 'valid',
                                init = 'he_normal',
                                name = 'conv2'))        
        model.add(ELU())    
        model.add(Convolution2D(48, 5, 5, 
                                subsample=(2,2), 
                                border_mode = 'valid',
                                init = 'he_normal',
                                name = 'conv3'))
        model.add(ELU())
        model.add(Dropout(0.5))
        model.add(Convolution2D(64, 3, 3, 
                                subsample = (1,1), 
                                border_mode = 'valid',
                                init = 'he_normal',
                                name = 'conv4'))
        model.add(ELU())              
        model.add(Convolution2D(64, 3, 3, 
                                subsample= (1,1), 
                                border_mode = 'valid',
                                init = 'he_normal',
                                name = 'conv5'))                 
        model.add(Flatten(name = 'flatten'))
        model.add(ELU())
        model.add(Dense(100, init = 'he_normal', name = 'fc1'))
        model.add(ELU())
        model.add(Dense(50, init = 'he_normal', name = 'fc2'))
        model.add(ELU())
        model.add(Dense(10, init = 'he_normal', name = 'fc3'))
        model.add(ELU())
        # do not put activation at the end because we want to exact output, not a class identifier
        model.add(Dense(1, name = 'output', init = 'he_normal'))
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer = adam, loss = 'mse')
        return model

    def train(self, train_name, ground_truth,model_weights):
        N_img_height = 66
        N_img_width = 220
        N_img_channels = 3
        self.model = self.nvidia_model(N_img_height, N_img_width, N_img_channels)
        self.model.load_weights(model_weights)
        image_train, velocity_train = self.preprocess_train(ground_truth)
        print('Training velocity...')
        self.model.fit(image_train,velocity_train, nb_epoch=50)
        self.model.save('velocity_model_train_{}.h5'.format(train_name))

    def predict(self,train_name, test_data):
        N_img_height = 66
        N_img_width = 220
        N_img_channels = 3
        self.model = self.nvidia_model(N_img_height, N_img_width, N_img_channels)
        self.model.load_weights('velocity_model_train_{}.h5'.format(train_name))
        print('Deploying velocity...')
        for file in test_data:
            test_X = self.preprocess_test(file['velocity_im_diff'])
            file['pred_velocity'] = float(self.model.predict(test_X))
        return test_data


