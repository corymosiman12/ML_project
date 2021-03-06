import os
import numpy as np

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

for model_type in full_dir:
    dir_min = {}
    for root, dirs, files in os.walk(model_type):
        for file in files:
            if file.endswith('mse.txt'):
                neurons = os.path.split(root)[-1]
                f_path = os.path.join(root, file)
                vals = np.loadtxt(f_path)
                dir_min[neurons] = vals[0]
    mod = os.path.split(model_type)[-1]
    key_min = min(dir_min.keys(), key=(lambda k: dir_min[k]))
    to_append = 'Model: {} \tNeurons: {} \tMin Value: {}'.format(mod, key_min, dir_min[key_min])
    mini.append(to_append)

wr_file = path + 'min_mse_per_model.txt'
with open(wr_file, 'w+') as f:
    for line in mini:
        print(line)
        f.write(line + '\n')