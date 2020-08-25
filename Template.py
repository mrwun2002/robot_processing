# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 09:08:11 2020

@author: Nathan Wu
"""
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D #Necessary for 3D graphing

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 


def process_string_genomes(genomes_as_string, dimensions):
    """Takes in genomes in a string format as well as the dimensions of the genome and outputs a 4-D np array.

    Args:
        genomes_as_string (str): Genomes in a string format
        dimensions (tuple): (x, y, z)

    Returns:
        np array: An np array with dimensions corresponding to the number of robots, the x dim, ydim, and z dim.
    """
    genomes = genomes_as_string.split(' ')

    for i in range(len(genomes)):
        genomes[i] = list(genomes[i])
    
    genomes = np.array(genomes).reshape((len(genomes), dimensions[2], dimensions[1], dimensions[0]))
    genomes = genomes.transpose(0, 3, 2, 1)

    return genomes

    
def process_string_labels(labels_as_string):
    labels = np.array(labels_as_string.split(' ')).astype(float)
    return labels


def train_linear_nn (features, labels):
    print("Note: The SKLearn model uses an error function of MSE/2 rather than just MSE.")
    model = MLPRegressor(hidden_layer_sizes = (120, 84), activation = 'relu',
                         solver = 'adam', batch_size = 200, max_iter = 99,
                         verbose = True, alpha = 0, early_stopping = True, validation_fraction=0.1)
    #alpha = 0 turns of Ridge Regression
    
    
    train_features, val_features, train_labels, val_labels = train_test_split(features, labels,
                                                                 test_size=0.2)
    
    
    model.fit(train_features, train_labels)

          
    print("\n\nDONE TRAINING\n\n")
    evaluate_linear_model(train_features, train_labels, model)
    evaluate_linear_model(val_features, val_labels, model) 
    
    return model


  
  
def evaluate_linear_model(features, labels, model):  
    
    
    predictions = model.predict(features)
    
    actual = labels
    
    mean_loss2 = mean_squared_error(actual, predictions)
    
    
    print("\nMean loss on Val Dataset: {}".format(mean_loss2))
    
    plt.plot(actual, predictions, 'o')
    plt.xlabel('Actual Distance Traveled')
    plt.ylabel('Predicted Distance Traveled')
    
    plt.show()

def encode_matrix(matrix):
    new_matrix = []
    for genome in matrix:
        new_matrix.append(encode_genome(genome))
    
    return np.array(new_matrix)

def encode_genome(genome):
    channel_0 = (genome == '0').astype("float32")
    channel_1 = (genome == '1').astype("float32")
    channel_2 = (genome == '2').astype("float32")
    channel_3 = (genome == '3').astype("float32")
    channel_4 = (genome == '4').astype("float32")
    channels = [channel_0, channel_1, channel_2, channel_3, channel_4]
    
    return np.array(channels)


def run():
    dimensions = (6, 6, 6)

    genomes_as_string = "<genomes_as_string>"
    labels_as_string = "<labels_as_string>"

    genomes = process_string_genomes(genomes_as_string, dimensions)
    labels = process_string_labels(labels_as_string)


    encoded_genomes = encode_matrix(genomes)


    flattened_genomes = np.reshape(encoded_genomes, (genomes.shape[0], -1))
    
    
    lin_model = train_linear_nn(flattened_genomes, labels)
    print("End program.")
    print(lin_model)
    return lin_model


run()


    




