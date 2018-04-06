# README

## Directory Structure:

local_project_directory
|---Github
    |---README.md
    |---data.py
    |---main.py
    |---plotting.py
    |---utils.py
    |---createFeatures.py
|---data
    |---isotropic2048.npz
    |---isotropic2048.csv
    |---isotropic2048.vti

## To Run
The main script should be run from the `Github` directory in order to correctly call the data files for reading.

## Scripts
The repo consists of the following scripts:
1. data.py
2. main.py
3. plotting.py
4. utils.py

## main.py
This function basically serves to:
1. Run the `form_train_test_sets()` function
2. Given a 256x256 grid, call the `form_features` function to create an array of size 65,536x9, i.e. a feature vector for each point.
3. 

### data.py
#### Function: load_data()
This is simply used to load data from either csv or npz.
If you already have the `isotropic2048.npz` file in your directory, use this file to load the data (it is more efficient).   Do this by using the line `filename_cvs = None` and *COMMENTING OUT* `filename_cvs = '../data/isotropic2048.csv`.

Once the data is serialized as npz, it is stored in a dictionary, where the `key` is the direction of *VELOCITY*, and the values are 2048x2048 np arrays.

#### Function: example_of_data(velocity)
The purpose of this function is to create a sample image of the coarsened dataset.  This is done by:
1. Taking in a `velocity` dictionary
2. Reducing the size down to 256x256
3. Applying a gaussian filter
4. Plotting all three together

#### Function: form_train_test_sets(velocity)
The purpose of this function is to implement the shifting strategy and return 4 data structures: data_train, data_test, filtered_train, filtered_test.

1. data_train: dictionary
* Uses: `utils.sparse_dict(), utils.sparse_array()`

Takes the velocity dictionary (where each key `u, v, w` corresponds to a 2048x2048 array), and using the number of coarse points defined (256), randomly selects an initial starting row and column index, selects every nth point (8 in our case), then returns a smaller array (256,256) and stores it into a new dictionary with the same keys (`u, v, w`).  

2. data_test: list of dictionaries
* Uses: `utils.sparse_dict(), utils.sparse_array()`

This is similar to the creation of `data_train`, except that there are three testing sets created.  The method used to form the testing sets is the same as for the training sets, however, it is repeated for each of the three testing sets.

3. filtered_train: dictionary
* Uses: `ndimage.gaussian_filter()`
Using the `data_train`, applies a gaussian filter with a standard deviation of 1 to 'blur' the vector field.

4. filtered_test: list of dictionaries
* Uses: `ndimage.gaussian_filter()`
Using the `data_test`, applies 3 different gaussian filters to the three arrays in the data set. The gaussian filters applied have gaussian filters of `[1, 0.9, 1.1]`.  This is to test how the model performs on data filtered with different filters.

### createFeatures.py
#### Function: form_features()
This functions basically just checks if what is passed is a `list` or `dict`.  It iterates through all dictionaries if it is a list (i.e. for x_test) and passes to:

#### Function: forming_features()
Given a dictionary of a grid of 256x256 points, create dictionary of an array of 65,536x9 feature points, i.e. a feature vector for each of the 65,536 points.