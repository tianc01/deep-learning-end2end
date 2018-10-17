import os,pickle,json
import datetime
import cv2
import numpy as np
from pdb import set_trace as st
from sklearn.model_selection import train_test_split

from config import Config
from distance_trainer import DistanceTrainer
from velocity_trainer import VelocityTrainer
from opticalHelpers import opticalFlowDenseDim3
from plot_distribution import PlotDistribution

class Experiment():
    def __init__(self,json_file_name):
        self.config = Config(json.load(open(json_file_name,'r')))
        if self.config.steps.extract_frames:
            self.extract_frames()
        # --------------------- Extract and save data ---------------------
        self.extract_data(load_data = self.config.steps.load_data)
        # --------------------- Obtain train and test data ---------------------
        self.get_train_test_data()
        # --------------------- Train --------------------- 
        self.train_distance_velocity(train_distance = self.config.steps.train_distance, 
            train_velocity = self.config.steps.train_velocity)
        # --------------------- Deploy --------------------- 
        if self.config.steps.predict:
            self.predict_distance_velocity()
        # --------------------- Load prediction and Visualize results --------------------- 
        if self.config.steps.plot:
            self.load_prediction()
            self.plot_save_prediction()

    def extract_data(self, load_data = False):
        if load_data:
            self.data = self.load_data()[:self.config.data_size.max_data_size]
        else:
            self.data = self.preprocess_ground_truth()
            self.save_data()

    def save_data(self):
        save_path = os.path.join(self.config.path.data,'data_{}.pkl'.format(self.config.data_name))
        print('Saving data: {}'.format(save_path))
        with open(save_path, 'wb') as f:
            pickle.dump(self.data, f, protocol=2)

    def load_data(self):
        load_path = os.path.join(self.config.path.data,'data_{}.pkl'.format(self.config.data_name))
        print('Loading data: {}'.format(load_path))
        data = pickle.load(open(load_path, 'rb'))
        return data

    def get_train_test_data(self):
        self.train_name = self.config.data_name
        if self.config.test.externel_source:
            print("Obtaining train and test...")
            self.train_data = self.data
            test_size = self.config.test.max_test_size
            print('Loading test data: {}'.format(self.config.test.load_path))
            self.test_data = pickle.load(open(self.config.test.load_path, 'rb'))[:test_size]
            self.test_name = self.config.test.test_name
        else:
            self.split_data()
            self.test_name = self.config.data_name
        print('Train size of {}: {}'.format(self.train_name,len(self.train_data)))
        print('Test size of {}: {}'.format(self.test_name,len(self.test_data)))

    def split_data(self):
        print("Splitting data to train and test...")
        self.train_data,self.test_data = train_test_split(self.data,test_size = 0.2, random_state=42)
        self.save_test_data()

    def save_test_data(self):
        save_path = os.path.join(self.config.path.data,'data_test_{}.pkl'.format(self.config.data_name))
        print('Saving data: {}'.format(save_path))
        with open(save_path, 'wb') as f:
            pickle.dump(self.test_data, f, protocol=2)

    def train_distance_velocity(self, train_distance = True, train_velocity = True):
        if train_distance:
            self.dt = DistanceTrainer()
            self.dt.train(self.config.data_name, self.train_data, self.config.distance_trainer.max_iteration, 
        	    self.config.distance_trainer.restore_sess)
        if train_velocity:
            self.vt = VelocityTrainer()
            self.vt.train(self.config.data_name, self.train_data, self.config.velocity_trainer.model_weights)

    def predict_distance_velocity(self):
        self.dt = DistanceTrainer()
        self.test_data = self.dt.predict(self.config.data_name,self.test_data)
        self.vt = VelocityTrainer()
        self.test_data = self.vt.predict(self.config.data_name,self.test_data)
        self.save_prediction()

    def save_prediction(self):
        save_path = os.path.join(self.config.path.results,
            'prediction_train_{}_test_{}.pkl'.format(self.train_name, self.test_name))
        print('Saving prediction results: {}'.format(save_path))
        with open(save_path, 'wb') as f:
            pickle.dump(self.test_data, f, protocol=2)

    def load_prediction(self):
        load_path = os.path.join(self.config.path.results,
            'prediction_train_{}_test_{}.pkl'.format(self.train_name, self.test_name))
        print('Loading prediction results: {}'.format(load_path))
        self.prediction = pickle.load(open(load_path, 'rb'))

    ## --------------------------Preprocess Data --------------------------
    def get_next_image(self,cur_image_name,len_prefix,len_postfix,grid):
        next_idx = int(cur_image_name[len_prefix:-len_postfix]) + 1
        next_file_name = self.config.image_name_convention.prefix + str(next_idx) + self.config.image_name_convention.postfix
        next_im = cv2.imread(os.path.join(self.config.path.images,next_file_name))[grid[1]:grid[3],grid[0]:grid[2],:]
        next_velocity_im = cv2.cvtColor(next_im, cv2.COLOR_BGR2RGB)
        next_velocity_im = cv2.resize(next_velocity_im, (220, 66), interpolation = cv2.INTER_AREA)
        return next_idx, next_file_name, next_velocity_im

    def preprocess_ground_truth(self):
        print('Extracting features for data {}'.format(self.config.data_name))
        data = []
        self.ground_truth = pickle.load(open(self.config.path.ground_truth, 'rb'))[:self.config.data_size.max_data_size]
        if self.config.data_type == 'synthetic':
            grid = [0,0,1248,384]
        elif self.config.data_type == 'naturalistic':
            grid = [570,442,1350,682]
        '''Examples: the consecutive frames are 69,70,71,72,95,96,97,98
        Before the loop, next_idx gives 99
        The loop is in reverse order: 98,97,96,95,72,71,70,69
        '''
        # ---------------- Initialize data for calculating image difference ----------------
        # Obtain the length of the prefix
        len_prefix = len(self.config.image_name_convention.prefix)
        len_postfix = len(self.config.image_name_convention.postfix)
        next_idx, next_file_name, next_velocity_im = self.get_next_image(self.ground_truth[-1]['image_name'],len_prefix,len_postfix,grid)
        # ---------------- Loop through data in reverse order ----------------
        for file in reversed(self.ground_truth):
            # read and crop the image
            im = cv2.imread(os.path.join(self.config.path.images,file['image_name']))[grid[1]:grid[3],grid[0]:grid[2],:]
            # ---------------- obtain image array for distance ----------------
            distance_im = cv2.resize(im,(1248,384))
            distance_im = distance_im[0:384,432:816,:] # im_size: H*W*C
            #  ---------obtain image array (diff between two consecutive images) for velocity ----------------
            velocity_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            velocity_im = cv2.resize(velocity_im, (220, 66), interpolation = cv2.INTER_AREA)
            # obtain next image's index, image name and image array
            # Example: for indice 98,97,96,95,72,71,70,69
            # 72+1 != 95 = next_idx, the following command resets next_idx = 72+1 = 73
            # otherwise, no change made to next_idx, next_file_name, next_velocity_im
            cur_idx = int(file['image_name'][len_prefix:-len_postfix])
            if cur_idx + 1 != next_idx:
                next_idx, next_file_name, next_velocity_im = self.get_next_image(file['image_name'],len_prefix,len_postfix,grid)
            im_diff = opticalFlowDenseDim3(velocity_im, next_velocity_im)
            # ----------------------- Append all data -----------------------
            data.append({'cur_frame':file['image_name'],'next_frame':next_file_name,
                'true_distance':file['true_distance'],'true_velocity': file['true_velocity'],
                'distance_im':distance_im,'velocity_im_diff':im_diff})
            next_idx, next_file_name, next_velocity_im = cur_idx, file['image_name'], velocity_im
        return data
 
    # For neutralistic video, extract frames and save them
    def extract_frames(self,start_sec=0,fin_sec=61):
        self.vidcap = cv2.VideoCapture(self.config.path.video)
        success,image = self.vidcap.read()
        if not success: 
            print('Could not open '+self.config.path.video)
            exit(1)
        if not os.path.exists(self.config.path.images): os.mkdir(self.config.path.images)
        count = 0
        success = True
        print('Saving images under {}'.format(self.config.path.images))
        while success:
            success,image = self.vidcap.read()
            print('Read frame {}: {}'.format(count,success))
            start_count = start_sec*30
            fin_count = fin_sec*30
            if success and count >= start_count and count <= fin_count:
                print('Writing...')
                cv2.imwrite(self.config.path.images+"/frame_{}_{}.jpg".format(count,self.config.video_name), image)     # save frame as JPEG file
            count += 1

    ## -------------------------- Prediction error plots --------------------------
    def plot_save_prediction(self):
        pd = PlotDistribution(self.prediction,
            postfix = 'train_{}_test_{}'.format(self.train_name, self.test_name),
            plot_params = self.config.plot_params)
        pd.plot_ground_truth(path = self.config.path.results)
        pd.plot_prediction(path = self.config.path.results)
        pd.plot_error(path = self.config.path.results)


if __name__ == '__main__':
    # # ------------------------------------------------------------------------------------ #
    # #                              Train on Naturalistic                                   #
    # #                              Deploy on Naturalistic                                  #
    # # ------------------------------------------------------------------------------------ #
    e = Experiment('/data/temp_clear_e2e/experiment_naturalistic.json')
  

    # ------------------------------------------------------------------------------------ #
    #                                    Train on Synthetic                                #
    #                               Deploy on Synthetic/Naturalistic                       #
    # ------------------------------------------------------------------------------------ #
    #e = Experiment('/data/temp_clear_e2e/experiment_synthetic.json')
