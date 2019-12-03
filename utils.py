import numpy as np
import glob
import os
import kalman_filter as kf

def create_track_data(dir_path, transition_matrix, observation_matrix, res_path='track_data/CompromisedScenario/', XYZ='y'):
    files = glob.glob(dir_path + XYZ + '*.txt')
    for eachfile in files:
        print('Creating track for %s' %eachfile)
        data = np.loadtxt(eachfile)
        data = np.reshape(data, (data.shape[0], -1))
        track = kf.create_track_kf(data, transition_matrix, observation_matrix)
        np.save(res_path + 'track' + XYZ + eachfile[-5] + '.npy', track)
    print('Created data!')