# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:43:00 2020

@author: Nathan Wu
"""

import matplotlib.pyplot as plt
#IMPORT PACKAGES
from lxml import etree


import numpy as np

#from mpl_toolkits.mplot3d import Axes3D #Necessary for 3D graphing

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 

import os #Used for file parsing
import sys #Throws errors and exits the program - not completely necessary



class Report():
    '''
    Takes a file for a report and turns it into a dictionary indexed by robot
    and with values of "total_distance_of_all_voxels" - if more are needed, 
    might need to make a "report_data" class for organizational purposes.
    
    Each report is associated with one folder.
    '''
    def __init__(self, report_file):
        self.distances = {}
        
        report_code = etree.parse(report_file)
        robots = report_code.xpath("//detail/*")
        
        for robot_report in robots:
            self.distances[robot_report.tag] = float(
                robot_report.xpath("total_distance_of_all_voxels")[-1].text)
        
        



class Robot():
    '''
    Pass in the name for the robot, a source file.
    Has properties name, short_name, dimensions, and genome.
    When printed, prints the genome along with the name.        
    '''
    def __init__(self, name: str, robot_file: str):
        self.name = name
        
        #Short_name removes the folder and the file type - IS NOT UNIQUE
        self.short_name = (self.name.split('/')[-1]).split('.')[0]
    
        
        robot_code = etree.parse(robot_file)
        

        self.dimensions = {
            "x": int(robot_code.xpath("//Structure/X_Voxels")[-1].text),
            "y": int(robot_code.xpath("//Structure/Y_Voxels")[-1].text),
            "z": int(robot_code.xpath("//Structure/Z_Voxels")[-1].text)
            }
        
        #Create an (x*y) by z NumPy array, filled with the value None.
        #If the value None is retained, an error has been made.
        self.genome = np.full(
            (self.dimensions["z"], self.dimensions["x"]*self.dimensions["y"]),
            None)
        self.genome_for_nn = np.full((self.dimensions["x"], 
                                     self.dimensions["y"], 
                                     self.dimensions["z"]),
                                     None)
        
        
        #Takes the genome from the XML file and puts it into self.genome
        
        layers = robot_code.xpath("//Structure/Data/Layer")
        layer_num = 0
        for layer in layers:
            char_num = 0
            for char in layer.text:
                try:
                    self.genome[layer_num, char_num] = char
                    self.genome_for_nn[char_num%self.dimensions['x'],
                                       char_num//self.dimensions['y'],
                                       layer_num] = char
                except IndexError:
                        sys.exit("Genome is larger than dimensions given!")
                
                char_num += 1
            layer_num += 1
        
        self.num_genome = self.genome.astype("float32")
        
        self.num_genome_for_nn = self.genome_for_nn.astype("float32")
        
        

    
    
    def __repr__(self): #When a robot is printed, print "name: genome". 
        
        print("\n{}:\n{}".format(self.name, self.num_genome))
        
        return ""
        



def get_files(path, all_files = False):
    '''
    
    Parameters
    ----------
    path : str
        The path to where all of the files are.
    all_files : bool, optional
        Whether or not to get all files available. The default is False.

    Returns
    -------
    A tuple of (robot_files: dict, report_files: dict)
    Each are indexed by what makes them unique: in the case of robot_files, 
    this is the folder/file name. In the case of the other 2, it is just 
    the folder.
    THIS FUNCTION ASSUMES THAT ROBOTS ARE .vxd, REPORTS ARE .xml, AND 
    BASE FILES ARE .vxa

    '''
    robot_files = {}

    report_files = {}
    
    folders = []
    
    for folder in os.scandir(path):
        folders.append(folder.name)
        
    
    if all_files == False:
        folders = folders[0:3]
        
    
    for folder in folders:
        path_branch = path + "/" + folder
        
                
        for file in os.scandir(path_branch):
            
            if file.name.endswith(".xml"):
                report_files[folder] = path + "/" + folder + "/" + file.name
            elif file.name.endswith(".vxd"):
                robot_files[folder + "/" + file.name] = path + "/" + folder + "/" + file.name

            
    return (robot_files, report_files)



def create_reports(report_files: dict):
        
    reports = {}
    
    for key in report_files:
        reports[key] = Report(report_files[key])
        
    return reports
        


def build_robots(robot_files: dict):
    '''
    
    Parameters
    ----------
    robot_files : dict
        Dictionary containing all robot file paths, indexed by the final folder and 
        the file name.

    Returns
    -------
    robots : list
        List containing robot objects.

    '''    
    robots = []
    
    for key in robot_files:
        new_robot = Robot(key, robot_files[key])
        robots.append(new_robot)
        
    return robots


def get_channels(array):
    channel_0 = (array == '0').astype("float32")
    channel_1 = (array == '1').astype("float32")
    channel_2 = (array == '2').astype("float32")
    channel_3 = (array == '3').astype("float32")
    channel_4 = (array == '4').astype("float32")
    channels = [channel_0, channel_1, channel_2, channel_3, channel_4]
    
    return np.array(channels)


def get_train_features(robots: list):
    '''
    
    Parameters
    ----------
    robots : list
        List of robots.

    Returns
    -------
    np array
        Dimensions len(train_features), channels, x dim, y dim, z dim.

    '''
    train_features = []
    
    for robot in robots:
        next_robot_channels = get_channels(robot.genome_for_nn)
        train_features.append(next_robot_channels)
    
    return np.array(train_features)


def get_train_labels(robots: list, reports: dict):
    '''
    
    Parameters
    ----------
    robots : list
        A list of all robots.
    reports : dict
        A dictionary, indexed by folder, cotaining respective Report objects.

    Returns
    -------
    train_labels : np array
        An np array of all of the labels, in the same order as the robots.

    '''
    
    train_labels = []
    
    for robot in robots:
        new_label = reports[robot.name.split('/')[0]].distances[robot.short_name]
        train_labels.append(new_label)
    
    return np.array(train_labels, dtype = "float32")



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





#MAIN FUNCTION
if __name__ == "__main__":
    
    #path = os.path.dirname(__file__) + "/farm"
    path = "farm"
    
    #Get the files
    robot_files, report_files = get_files(path)
    print("All files retrieved.")
    
    
    #Create the reports - this is done before breaking out into robots 
    #to avoid creating many identical reports
    reports  = create_reports(report_files)
    print("All reports created.")
    
    robots = build_robots(robot_files)
    print("All robots built.")
    
    train_features = get_train_features(robots)
    train_labels = get_train_labels(robots, reports)
    print("Features and labels extracted.")
    
    print("Shape of features array: {}".format(np.shape(train_features)))
    print("Shape of labels array: {}".format(np.shape(train_labels)))
    print(train_labels)
    
    flattened_train_features = np.reshape(train_features, (train_features.shape[0], -1))
    
    
    lin_model = train_linear_nn(flattened_train_features, train_labels)
    print("End program.")
    




