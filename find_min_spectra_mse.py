import os
import numpy as np
from sklearn.metrics import mean_squared_error

path = '/Users/corymosiman/OneDrive - UCB-O365/Spring 2018/CSCI 5622 - Machine Learning/ML_Project/Github/plots/FF_1L/'

dirs = ['2dim_gaussian_9feat', '2dim_gaussian_9feat_tanh',
        '2dim_gaussian_27feat', '2dim_gaussian_27feat_tanh', 
        '2dim_median_9feat', '2dim_median_27feat',
        '2dim_noise_9feat', '2dim_noise_9feat_tanh',
        '2dim_noise_27feat', '2dim_noise_27feat_tanh',
        '2dim_physical_sharp_9feat_tanh', '2dim_physical_sharp_27feat_tanh', 
        '3dim_gaussian_27feat_tanh']

mini = []

full_dir = [os.path.join(path, dir) for dir in dirs]

counter = 0
for model_type in full_dir:
    dir_min = {}
    for root, dirs, files in os.walk(model_type):
        # print(files)
        count = 0
        for file in files:
            if file.endswith('true0.spectra'):
                neurons = os.path.split(root)[-1]
                f_path = os.path.join(root, file)
                true = np.loadtxt(f_path)
                count+=1
            elif file.endswith('predicted0.spectra'):
                neurons = os.path.split(root)[-1]
                f_path = os.path.join(root, file)
                pred = np.loadtxt(f_path)
                count+=1
            if count == 2:
                counter+=1
                count = 0
                mse = mean_squared_error(true, pred)
                dir_min[neurons] = mse
    mod = os.path.split(model_type)[-1]
    key_min = min(dir_min.keys(), key=(lambda k: dir_min[k]))
    to_append = 'Model: {} \tNeurons: {} \tMin Value: {}'.format(mod, key_min, dir_min[key_min])
    mini.append(to_append)

wr_file = path + 'min_spectra_mse_per_model.txt'
with open(wr_file, 'w+') as f:
    for line in mini:
        print(line)
        f.write(line + '\n')