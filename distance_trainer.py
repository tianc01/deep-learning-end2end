import tensorflow as tf
import pprint
import re
import os
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import random
import cv2
from pdb import set_trace as st
from PIL import Image

class DistanceTrainer:
    def preprocess_train(self,ground_truth,im_w,im_h,im_c):
        print("Preprocess data for distance trainer...")
        # extract train data
        data = {}
        data['X'],data['distance'] = [],[]
        for file in ground_truth:
            im = file['distance_im'][:,:,:im_c]
            if any([im_w != im.shape[1],im_h != im.shape[0]]):im = cv2.resize(im,(im_w,im_h))
            data['X'].append(np.rollaxis(np.reshape(im,(im_h,im_w,im_c)),1,0))    
            data['distance'].append(file['true_distance'])
        data['X'] = np.reshape(data['X'],(len(ground_truth),im_w,im_h,im_c))
        data['distance'] = np.reshape(data['distance'],(len(ground_truth),1))

        return data

    def preprocess_test(self,im,im_w,im_h,im_c):
        im = im[:,:,:im_c]
        if any([im_w != im.shape[1],im_h != im.shape[0]]):im = cv2.resize(im,(im_w,im_h))
        im = np.reshape(np.rollaxis(np.reshape(im,(im_h,im_w,im_c)),1,0),(1,im_w,im_h,im_c))
        return im

    def next_batch(self,num, X, y):
        ''' input: numpy arrays '''
        idx = np.random.choice(X.shape[0], num, replace=False)
        X_shuffle = X[idx,:,:,:]
        y_shuffle = y[idx,:]
        
        return X_shuffle, y_shuffle

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def deepnn(self,x,im_w,im_h,im_c):
        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since images are
        # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
        x_image = tf.reshape(x, [-1, im_w, im_h, im_c])

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        W_conv1 = self.weight_variable([5, 5, im_c, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        h_pool1 = self.max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        h_pool2 = self.max_pool_2x2(h_conv2)

        # the image size after 2 conv and 2 pool layers
        im_w_fc1 = int(math.ceil(math.ceil(im_w/2.0)/2.0))
        im_h_fc1 = int(math.ceil(math.ceil(im_h/2.0)/2.0))

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        W_fc1 = self.weight_variable([im_w_fc1*im_h_fc1*64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, im_w_fc1*im_h_fc1*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        W_fc2 = self.weight_variable([1024, 1])
        b_fc2 = self.bias_variable([1])

        y_conv = tf.identity(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name = 'y_conv')
        return y_conv, keep_prob

    def deep_train(self,train_name,method_name,data,im_w,im_h,im_c,restore_sess,total_iterations):
        # Create the model
        x = tf.placeholder(tf.float32, [None,im_w,im_h,im_c], name = 'x')
        y_ = tf.placeholder(tf.float32, [None, 1])
        # Build the graph for the deep net
        y_conv, keep_prob = self.deepnn(x,im_w,im_h,im_c)
        cost = tf.reduce_mean(tf.square(y_conv-y_))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

        tf.add_to_collection('x', x)
        tf.add_to_collection('y_conv', y_conv)
        tf.add_to_collection('keep_prob', keep_prob)

        saver = tf.train.Saver()
        sess = tf.Session()
        if restore_sess:
            saver = tf.train.import_meta_graph("distance_model_{}_{}.meta".format(train_name,method_name))
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            print("Model restored.")
            max_iters = 2
        else:       
            sess.run(tf.global_variables_initializer())
            max_iters = total_iterations

        for i in range(max_iters):
            batch_X, batch_y = self.next_batch(50, data['X'], data['distance'])
            if i % 100 == 0:
                train_mse = cost.eval(session=sess,feed_dict={x: batch_X, y_: batch_y, keep_prob: 1.0})
                print('step {}, training mse {}'.format(i, train_mse))
            train_step.run(session=sess,feed_dict={x: batch_X, y_: batch_y, keep_prob: 0.5})
            if (i+1) % 100 == 0:
                save_path = saver.save(sess, "distance_model_{}_{}".format(train_name,method_name))
            
    def train(self, train_name, ground_truth, total_iterations, restore_sess):
        ## -------------------------- Setting -----------------------------------
        method_name = 'LeNet_gray'
        im_w, im_h, im_c = 100,100,1

        train_data = self.preprocess_train(ground_truth,im_w,im_h,im_c)
        print('Training distance...')
        self.deep_train(train_name,method_name,train_data,im_w,im_h,im_c,restore_sess,total_iterations)

    def predict(self,train_name,test_data):
        method_name = 'LeNet_gray'
        im_w, im_h, im_c = 100,100,1

        print('Deploying distance...')
        sess = tf.Session()
        # Load meta graph and restore weights
        saver = tf.train.import_meta_graph("distance_model_{}_{}.meta".format(train_name,method_name))
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y_conv = graph.get_tensor_by_name("y_conv:0")
        keep_prob = graph.get_tensor_by_name('keep_prob:0')

        for file in test_data:
            test_X = self.preprocess_test(file['distance_im'],im_w,im_h,im_c)
            file['pred_distance'] = float(sess.run(y_conv, feed_dict={x:test_X, keep_prob: 1.0}))
        return test_data