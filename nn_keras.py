import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pylab as plt

class my_keras():
    def __init__(self, num_epochs, num_neurons):
        self.epochs = num_epochs
        self.num_neurons = num_neurons
        self.predictions = []
        self.mse = []

    # def baseline_model(num_neurons, input_shape):
    def baseline_model(self):
        # create model
        model = Sequential()
        model.add(Dense(self.num_neurons, input_shape=(9,), kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        # compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        logging.info(model.summary())
        self.model = model
        return self.model

    def evaluate_model(self, X_train, y_train, X_validation, y_validation):
    # def evaluate_model(self, X_train, y_train):
        seed = 12
        seed = np.random.seed(seed)
        shp = X_train.shape

        logging.info('X_train shape: {}'.format(X_train.shape))

        # Keras requirese to pass blank function to `build_fn`
        self.estimator = KerasRegressor(build_fn=self.baseline_model, epochs=self.epochs, batch_size=64, verbose=2)
        self.estimator_trained = self.estimator.fit(X_train, y_train, validation_data = (X_validation, y_validation))
        logging.info("Train loss: {} Val loss: {}".format(self.estimator_trained.history['loss'], self.estimator_trained.history['val_loss']))
        self.plot_loss_per_epoch()

    def plot_loss_per_epoch(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        ax.plot(range(1,self.epochs+1), self.estimator_trained.history['loss'], color="steelblue", marker="o", label="training")
        ax.plot(range(1,self.epochs+1), self.estimator_trained.history['val_loss'], color="green", marker="o", label="validation")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=10)
        plt.xticks(range(1,self.epochs+2,2))
        ax.set_xlabel("epoch", fontsize=16)
        ax.set_ylabel("loss", fontsize=16)
        plt.savefig('plots/' + datetime.now().strftime('%Y-%m-%d %H_%M') + '_loss_per_epoch.png')

    def evaluate_test_sets(self, X_test_list, y_test_list):
        for test_set in range(len(X_test_list)):
            prediction = self.estimator.predict(X_test_list[test_set]['u'])
            error = mean_squared_error(y_test_list[test_set]['u'], prediction)
            self.predictions.append(prediction)
            self.mse.append(error)
        self.plot_mse()

    def plot_mse(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        sigma = [1, 0.9, 1.1]
        labels = [r'$\sigma = 1$', r'$\sigma = 0.9$', r'$\sigma = 1.1$']
        colors = ['steelblue','green','black']
        for mse in range(len(self.mse)):
            ax.scatter(sigma[mse], self.mse[mse], color=colors[mse], marker="o", label=labels[mse])
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=10)
        bottom = min(self.mse) - 0.1*min(self.mse)
        top = max(self.mse) + 0.1*min(self.mse)
        ax.set_ylim(bottom, top)
        # plt.xticks(range(1,self.epochs+2,2))
        ax.set_xlabel("sigma", fontsize=16)
        ax.set_ylabel("MSE", fontsize=16)
        plt.savefig('plots/' + datetime.now().strftime('%Y-%m-%d %H_%M') + '_MSE.png')

# LEFTOVERS
    # results = cross_val_score(estimator, X_train, y_train, cv=kfold)    