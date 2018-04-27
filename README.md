# README

## Intro:
This project is based on Maulik and San's paper (2017): [A neural network approach for the blind deconvolution of turbulent flows](https://arxiv.org/abs/1706.00912), which applies a single-layer feed-forward ANN for the deconvolution of flow variables (i.e. a velocity field) from their coarse-grained computations.  The application of deconvolution to fluid flows is similar to image deblurring: given a coarse-grained velocity field (low resolution photo), increase the sharpness (remove blurring artifacts).  Deconvolution has been very successful in its application to images, however, the focus of the aforementioned work is applying blind deconvolution to turbulent flows.  The aim of this ANN is to recover subfilter-scale structure of the flow (deconvolution) without knowing the filter function applied (blind).  Maulik and San's work uses an Extreme Learning Machine (ELM) as proposed in [Huang, 2004](https://ieeexplore.ieee.org/document/1380068/) for the learning algorithm. Our project explores the ELM approach, but additionally considers single-layer and two-layer ANN's with the ADAM optimizer and a relu activation function.  

## Directory Structure:
local_project_directory  
|---Github/  
    |---README.md  
    |---data.py  
    |---extreme_learning_machine.py  
    |---main.py  
    |---nn_keras.py  
    |---plotting.py  
    |---README.md  
    |---tests.py
    |---utils.py  

|---data/  

    |---isotropic2048.npz  
    |---isotropic2048.csv  
    |---isotropic2048.vti  
    |---HIT_u.bin  
    |---HIT_v.bin  
    |---HIT_w.bin

|---plots/  

Plots and all directories and associated data files will be produced upon execution of the main function.

## To Run
The main script should be run from the `Github` directory in order to correctly call the data files for reading.  

`Github$ python main.py`

## main.py
Depending on the model, the program is set up to iterate through a different number of epochs and neurons, retraining a model for each and outputting separate results.  The following variables are configurable:

`model_type`:       type: `str`. Choose from: FF_1L, FF_2L, Olga_ELM, Rahul_ELM  
`num_features`:     type: `int`. Choose from: 9, 25, 27  
`num_epochs`:       type: `list`. Define the number of epochs to use for FF models
`num_neurons_L1`:   type: `list`. Define the number of neurons to use for FF models
`num_neurons_L2`:   type: `list`. Define the number of neurons to use for FF_2l models
`dimension`:        type: `int`. Choose from: 2 or 3. Defines the number of dimensions used for training the model.



