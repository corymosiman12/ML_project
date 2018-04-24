import data
import logging
import numpy as np
import extreme_learning_machine as elm
import sys
import plotting

plot_folder = './plots/'
Npoints_coarse2D = 256
Npoints_coarse3D = 64



def main():
    """
    X_train: dictionary
        dict.keys(): 'u', 'v', 'w'
        dict.values():
            - 256x256 randomly created array from 2048x2048 array using shifting strategy
            - Random filter applied with sigma = 1
    y_train: dictionary
        dict.keys(): 'u', 'v', 'w'
        dict.values():
            - 256x256 randomly create array from 2048x2048
            - y_train = X_train before filter applied (i.e. NN tries to recover unfiltered)
    X_test: list of dictionaries where dict.keys() and dict.values() are same as X_train, except:
        X_test[0]: Filter applied with sigma = 1
        X_test[1]: Filter applied with sigma = 1.1
        X_test[2]: Filter applied with sigma = 0.9
    y_test: list of dictionaries
        - y_test = X_test before filters applied
    """




    plot_folder = './plots/'
    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.INFO)
    logging.info('platform {}'.format(sys.platform))
    logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    logging.info('numpy {}'.format(np.__version__))
    logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))

    ########################## UNCOMMENT FOR 2D ANALYSIS ##########################
    # velocity = data.load_data()
    # X_train, y_train, X_test, y_test = data.form_train_test_sets(velocity)

    ########################## UNCOMMENT FOR 3D ANALYSIS ##########################
    velocity = data.load_data(dimension=3)
    X_train, y_train, X_test, y_test = data.form_train_test_sets(velocity, Npoints_coarse3D)

    ########################## OLGA START ##########################


    logging.info('\nNN is Extreme learning machine (algorithm from the paper)\n')
    y_predict = elm.extreme_learning_machine(X_train, y_train, X_test, y_test)
    plotting.plot_velocities_and_spectra(X_test, y_test, y_predict, plot_folder)
    plotting.plot_vorticity_pdf(X_test, y_test, y_predict, plot_folder)


    # # ########################## CORY START ##########################

    # # Create single layer model
    # # Define the number of inputs to be used for creating the feature vectors
    # n_features = [9, 27]
    # num_epochs = [5, 10, 15]
    # num_neurons = [50, 100, 150]
    #
    # for features in n_features:
    #     # Define on what your model will be trained: u, v, or w
    #     key = 'u'
    #
    #     # X_train_final shape = (256*256, n_features)
    #     # y_train_final shape = (256*256, 1)
    #     X_train_final, y_train_final = utils.final_transform(X_train, y_train, n_features=features, train=True, index=key)
    #
    #     """
    #     X_test_final and y_test_final are both lists of dictionaries:
    #         X_test_final[0]: Filter applied with sigma = 1
    #         X_test_final[1]: Filter applied with sigma = 1.1
    #         X_test_final[2]: Filter applied with sigma = 0.9
    #         And similar for y_test_final
    #     Each list element has a dictionary with 3 keys: 'u', 'v', 'w'. The shapes are equivalent to the training set:
    #         X_test_final[0]['u'].shape = (256*256, n_features)
    #         y_test_final[0]['u'].shape = (256*256, 1)
    #     """
    #     X_test_final, y_test_final = utils.final_transform(X_test, y_test, features)
    #     for epochs in num_epochs:
    #         for neurons in num_neurons:
    #             logging.info('Evaluating model for {} features, {} epochs, and {} neurons'.format(str(features), str(epochs), str(neurons)))
    #
    #             # Create folder for plots
    #             plot_folder = './plots/'
    #             plot_folder = os.path.join(plot_folder, '{}_features'.format(str(features)),
    #                                                     '{}_neurons'.format(str(neurons)),
    #                                                     '{}_epochs'.format(str(epochs)))
    #             if not os.path.isdir(plot_folder):
    #                 os.makedirs(plot_folder)
    #
    #             model = nnk.my_keras(epochs, neurons, features)
    #
    #             # Evaluate model, validating on same test set key as trained on
    #             model.evaluate_model(X_train_final, y_train_final, X_test_final[0][key], y_test_final[0][key], plot_folder)
    #
    #             # Predict on each of the test sets and plot MSE:
    #             # MSE plotting currently not working
    #             model.evaluate_test_sets(X_test_final, y_test_final)
    #
    #             plotting.plot_velocities_and_spectra(X_test, y_test, model.predictions, plot_folder)
    # ########################## CORY END ##########################

if __name__ == '__main__':
    main()
