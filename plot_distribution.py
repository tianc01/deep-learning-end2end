import os
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import random
import numpy

from pdb import set_trace as st
from PIL import Image

class PlotDistribution():
    def __init__(self,prediction,postfix,plot_params):
        '''
        prediction: list of dictionaries
        prediction[0].keys():
        'velocity_im_diff', 'pred_velocity', 'distance_im', 'true_velocity',
        'pred_distance', 'next_frame', 'true_distance', 'cur_frame'
        '''
        self.postfix = postfix
        self.tc_params = plot_params.temp_clear
        self.distance_params = plot_params.distance
        self.velocity_params = plot_params.velocity
        self.tc_error_params = plot_params.temp_clear_error
        self.distance_error_params = plot_params.distance_error
        self.velocity_error_params = plot_params.velocity_error

        self.true_temp_clear = []
        self.pred_temp_clear = []
        self.error_temp_clear,self.error_distance,self.error_velocity = [],[],[]
        for file in prediction:
            if file['true_velocity'] > 0 and file['pred_velocity'] > 0:
                cur_true_tc = file['true_distance']/file['true_velocity']
                cur_pred_tc = file['pred_distance']/file['pred_velocity']
                # if (cur_true_tc > 3 or 
                #     cur_true_tc < 0 or
                #     cur_pred_tc > 3 or 
                #     cur_pred_tc < 0 or
                #     abs(cur_pred_tc-cur_true_tc) > 1.5 or
                #     file['true_distance'] < 0 or
                #     file['true_distance'] > 60 or
                #     file['pred_distance'] < 0 or 
                #     file['pred_distance'] > 60 or 
                #     abs(file['pred_distance']- file['true_distance']) > 40 or
                #     file['true_velocity'] < 0 or
                #     file['true_velocity'] > 60 or
                #     file['pred_velocity'] < 0 or
                #     file['pred_velocity'] > 60 or
                #     abs(file['pred_velocity']-file['true_velocity']) > 50): 
                #     print('{} true temporal clearance: {}, pred temporal clearance: {}'.format(file['cur_frame'],cur_true_tc,cur_pred_tc))
                #     print('{} true distance: {}, pred distance: {}'.format(file['cur_frame'],file['true_distance'],file['pred_distance']))
                #     print('{} true velocity: {}, pred velocity: {}'.format(file['cur_frame'],file['true_velocity'],file['pred_velocity']))
                #     st()
                self.true_temp_clear.append(cur_true_tc)
                self.pred_temp_clear.append(cur_pred_tc)
                self.error_temp_clear.append(cur_pred_tc-cur_true_tc)
                self.error_distance.append(file['pred_distance']-file['true_distance'])
                self.error_velocity.append(file['pred_velocity']-file['true_velocity'])

        self.true_distance = {file['cur_frame']:file['true_distance'] for file in prediction}
        self.pred_distance = {file['cur_frame']:file['pred_distance'] for file in prediction}
        self.true_velocity = {file['cur_frame']:file['true_velocity'] for file in prediction}
        self.pred_velocity = {file['cur_frame']:file['pred_velocity'] for file in prediction}

    def plot_ground_truth(self, path):
        self.plot_tc_truth(path)
        self.plot_distance_truth(path)
        self.plot_velocity_truth(path)

    def plot_prediction(self, path):
        self.plot_tc_prediction(path)
        self.plot_distance_prediction(path)
        self.plot_velocity_prediction(path)

    def plot_error(self, path):
        self.plot_tc_error(path)
        self.plot_distance_error(path)
        self.plot_velocity_error(path)

    def plot_tc_truth(self, path):
        ## Plot true distribution of temporal clearance
        fig, ax = plt.subplots(nrows=1,ncols=1)
        bin_width = float(self.tc_params.xlim[1] - self.tc_params.xlim[0])/self.tc_params.num_bins
        ax.hist(self.true_temp_clear,
            bins=np.arange(self.tc_params.xlim[0], 
                self.tc_params.xlim[1]+bin_width, bin_width), 
            normed = self.tc_params.normed)
        ax.set_xlim(self.tc_params.xlim)
        ax.set_ylim(self.tc_params.ylim)
        ax.set_xlabel('temporal clearance (second)')
        ax.set_title('True Temporal Clearance Distribution (n={})'.format(len(self.true_temp_clear)))
        save_path = os.path.join(path,'true_tc_distribution_{}.png'.format(self.postfix))
        print('Saving figure of true temporal clearance distribution: {}'.format(save_path))
        fig.savefig(save_path)
        plt.close(fig)

    def plot_distance_truth(self, path):
        ## Plot true distribution of distance
        fig, ax = plt.subplots(nrows=1,ncols=1) # 100: 10000 pixel
        bin_width = float(self.distance_params.xlim[1] - self.distance_params.xlim[0])/self.distance_params.num_bins
        ax.hist(self.true_distance.values(),
            bins=np.arange(self.distance_params.xlim[0], 
                self.distance_params.xlim[1]+bin_width,bin_width), 
            normed = self.distance_params.normed)
        ax.set_xlim(self.distance_params.xlim)
        ax.set_ylim(self.distance_params.ylim)
        ax.set_xlabel('distance (meter)')
        ax.set_title('True Distance Distribution (n={})'.format(len(self.true_distance)))
        save_path = os.path.join(path,'true_distance_distribution_{}.png'.format(self.postfix))
        print('Saving figure of true distance distribution: {}'.format(save_path))
        fig.savefig(save_path)
        plt.close(fig)

    def plot_velocity_truth(self, path):
        ## Plot true distribution of velocity
        fig, ax = plt.subplots(nrows=1,ncols=1) # 100: 10000 pixel
        bin_width = float(self.velocity_params.xlim[1] - self.velocity_params.xlim[0])/self.velocity_params.num_bins
        ax.hist(self.true_velocity.values(),
            bins=np.arange(self.velocity_params.xlim[0], 
                self.velocity_params.xlim[1]+bin_width,bin_width), 
            normed = self.velocity_params.normed)
        ax.set_xlim(self.velocity_params.xlim)
        ax.set_ylim(self.velocity_params.ylim)
        ax.set_xlabel('velocity (meter/second)')
        ax.set_title('Velocity Distribution (n={})'.format(len(self.true_velocity)))
        save_path = os.path.join(path,'true_velocity_distribution_{}.png'.format(self.postfix))
        print('Saving figure of true velocity distribution: {}'.format(save_path))
        fig.savefig(save_path)
        plt.close(fig)

    def plot_tc_prediction(self, path):
        ## Plot predicted distribution of temporal clearance
        fig, ax = plt.subplots(nrows=1,ncols=1) # 100: 10000 pixel
        bin_width = float(self.tc_params.xlim[1] - self.tc_params.xlim[0])/self.tc_params.num_bins
        ax.hist(self.pred_temp_clear,
            bins=np.arange(self.tc_params.xlim[0], 
                self.tc_params.xlim[1]+bin_width, bin_width), 
            normed = self.tc_params.normed)
        ax.set_xlim(self.tc_params.xlim)
        ax.set_ylim(self.tc_params.ylim)
        ax.set_xlabel('temporal clearance (second)')
        ax.set_title('Predicted Temporal Clearance Distribution (n={})'.format(len(self.pred_temp_clear)))
        save_path = os.path.join(path,'pred_tc_distribution_{}.png'.format(self.postfix))
        print('Saving figure of predicted temporal clearance distribution: {}'.format(save_path))
        fig.savefig(save_path)
        plt.close(fig)

    def plot_distance_prediction(self, path):
        ## Plot predicted distribution of distance
        fig, ax = plt.subplots(nrows=1,ncols=1) # 100: 10000 pixel
        bin_width = float(self.distance_params.xlim[1] - self.distance_params.xlim[0])/self.distance_params.num_bins
        ax.hist(self.pred_distance.values(),
            bins=np.arange(self.distance_params.xlim[0], 
                self.distance_params.xlim[1]+bin_width, bin_width), 
            normed = self.distance_params.normed)
        ax.set_xlim(self.distance_params.xlim)
        ax.set_ylim(self.distance_params.ylim)
        ax.set_xlabel('distance (meter)')
        ax.set_title('Predicted Distance Distribution (n={})'.format(len(self.pred_distance)))
        save_path = os.path.join(path,'pred_distance_distribution_{}.png'.format(self.postfix))
        print('Saving figure of predicted distance distribution: {}'.format(save_path))
        fig.savefig(save_path)
        plt.close(fig)

    def plot_velocity_prediction(self, path):
        ## Plot predicted distribution of velocity
        fig, ax = plt.subplots(nrows=1,ncols=1) # 100: 10000 pixel
        bin_width = float(self.velocity_params.xlim[1] - self.velocity_params.xlim[0])/self.velocity_params.num_bins
        ax.hist(self.pred_velocity.values(),
            bins=np.arange(self.velocity_params.xlim[0], 
                self.velocity_params.xlim[1]+bin_width, bin_width), 
            normed = self.velocity_params.normed)
        ax.set_xlim(self.velocity_params.xlim)
        ax.set_ylim(self.velocity_params.ylim)
        ax.set_xlabel('velocity (meter/second)')
        ax.set_title('Predicted Velocity Distribution (n={})'.format(len(self.pred_velocity)))
        save_path = os.path.join(path,'pred_velocity_distribution_{}.png'.format(self.postfix))
        print('Saving figure of predicted velocity distribution: {}'.format(save_path))
        fig.savefig(save_path)
        plt.close(fig)

    def plot_tc_error(self, path):
        # # Plot error distribution of temporal clearance
        fig, ax = plt.subplots(nrows=1,ncols=1) # 100: 10000 pixel
        bin_width = float(self.tc_error_params.xlim[1] - self.tc_error_params.xlim[0])/self.tc_error_params.num_bins
        ax.hist(self.error_temp_clear,
            bins=np.arange(self.tc_error_params.xlim[0], 
                self.tc_error_params.xlim[1]+bin_width, bin_width), 
            normed = self.tc_error_params.normed)
        ax.set_xlim(self.tc_error_params.xlim)
        ax.set_ylim(self.tc_error_params.ylim)
        ax.set_xlabel('temporal clearance (second)')
        ax.set_title('Temporal Clearance Error (Pred-True) Distribution (n={})'.format(len(self.error_temp_clear)))
        save_path = os.path.join(path,'error_tc_distribution_{}.png'.format(self.postfix))
        print('Saving figure of temporal clearance error distribution: {}'.format(save_path))
        fig.savefig(save_path)
        plt.close(fig)

    def plot_distance_error(self, path):
        # # Plot error distribution of distance 
        fig, ax = plt.subplots(nrows=1,ncols=1) # 100: 10000 pixel
        bin_width = float(self.distance_error_params.xlim[1] - self.distance_error_params.xlim[0])/self.distance_error_params.num_bins
        ax.hist(self.error_distance,
            bins=np.arange(self.distance_error_params.xlim[0], 
                self.distance_error_params.xlim[1]+bin_width, bin_width), 
            normed = self.distance_error_params.normed)
        ax.set_xlim(self.distance_error_params.xlim)
        ax.set_ylim(self.distance_error_params.ylim)
        ax.set_xlabel('distance (meter)')
        ax.set_title('Distance Error (Pred-True) Distribution (n={})'.format(len(self.error_distance)))
        save_path = os.path.join(path,'error_distance_distribution_{}.png'.format(self.postfix))
        print('Saving figure of distance error distribution: {}'.format(save_path))
        fig.savefig(save_path)
        plt.close(fig)

    def plot_velocity_error(self, path):
        # # Plot error distribution of velocity
        fig, ax = plt.subplots(nrows=1,ncols=1) # 100: 10000 pixel
        bin_width = float(self.velocity_error_params.xlim[1] - self.velocity_error_params.xlim[0])/self.velocity_error_params.num_bins
        ax.hist(self.error_velocity,
            bins=np.arange(self.velocity_error_params.xlim[0], 
                self.velocity_error_params.xlim[1]+bin_width, bin_width), 
            normed = self.velocity_error_params.normed)
        ax.set_xlim(self.velocity_error_params.xlim)
        ax.set_ylim(self.velocity_error_params.ylim)
        ax.set_xlabel('velocity (meter/second)')
        ax.set_title('Velocity Error (Pred-True) Distribution (n={})'.format(len(self.error_velocity)))
        save_path = os.path.join(path,'error_velocity_distribution_{}.png'.format(self.postfix))
        print('Saving figure of velocity error distribution: {}'.format(save_path))
        fig.savefig(save_path)
        plt.close(fig)