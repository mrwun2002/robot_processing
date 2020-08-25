# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:43:00 2020

@author: Nathan Wu
"""


#IMPORT PACKAGES
from lxml import etree


import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D #Necessary for 3D graphing


import torch
from torch.utils.data import DataLoader, TensorDataset
#neural net imports
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.autograd import Variable 

from sklearn.model_selection import train_test_split

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
        folders = [folders[0], folders[1]]
        
    
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



def train_nn (features, labels, num_epochs = 20):
    train_features, val_features, train_labels, val_labels = train_test_split(features, labels,
                                                                 test_size=0.2)
    
    train_features_tensor = torch.tensor(train_features)
    train_labels_tensor = torch.tensor(train_labels)
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    
    
    val_features_tensor = torch.tensor(val_features)
    val_labels_tensor = torch.tensor(val_labels)
    val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
    


    for epoch in range(num_epochs):
        print("\nEpoch {}/{}:".format(epoch + 1, num_epochs))
        train_epoch(train_dataset, batch_size = 200)
        give_feedback(val_dataset)
          
    print("\n\nDONE TRAINING\n\n")
    evaluate_model(val_dataset)    


def train_epoch(dataset, batch_size = 32): #Train the model and report on loss from it
    model.train()
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    for features, labels in data_loader:
        
        
        output = model(features)
        
        loss = criterion(output.view(-1), labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print ('Training Loss: {:.4f}' .format(loss.item()))


def give_feedback(dataset): #Run the val dataset on the model
    #Not passing in a dataloader to avoid batches
    model.eval()
    loss = 0
    
    data_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
    
    for features, labels in data_loader:
        output = model(features)
        loss += F.mse_loss(output.view(-1), labels).item() #Ignore batches
        
    
    loss /= len(dataset)
    
    print('Average Val Loss: {:.4f}'.format(loss))
    
    
def evaluate_model(dataset):  
    predictions = []
    actual = []
    
    model.eval()
    
    data_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
    
    for features, labels in data_loader:
        output = model(features)
        predictions.append(output.item())
        actual.append(labels.item())
    
    plt.plot(actual, predictions, 'o')
    plt.xlabel('Actual Distance Traveled')
    plt.ylabel('Predicted Distance Traveled')
    
    plt.show()



class LinNet(nn.Module):
    def __init__(self):
        super(LinNet, self).__init__()
        #First unit of convolution
        #self.conv_unit_1 = nn.Sequential(
       #     nn.Conv3d(5, 5, kernel_size=3, padding=1),
        #    nn.BatchNorm3d(5),
         #   nn.ReLU(),
          #  #nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
           # )

        #Second unit of convolution        
        #self.conv_unit_2 = nn.Sequential(
         #   nn.Conv3d(30, 5, kernel_size=3, padding=1),
          #  nn.BatchNorm3d(30),
           # nn.ReLU(),
            #nn.MaxPool3d(kernel_size=2, stride=2)
            #)

        #Fully connected layers
        self.fc1 = nn.Linear(1080, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)     
        
    def forward(self, out):       
        #out = self.conv_unit_1(out)
        #out = self.conv_unit_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)  
        out = F.relu(out)
        out = self.fc3(out)      
        #out = F.relu(out)                        
        return out
  
    
    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #First unit of convolution
        self.conv_unit_1 = nn.Sequential(
            nn.Conv3d(5, 30, kernel_size=3, padding=1),
            #nn.BatchNorm3d(5),
            nn.ReLU(),
            #nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
            )

        #Second unit of convolution        
        self.conv_unit_2 = nn.Sequential(
            nn.Conv3d(30, 5, kernel_size=3, padding=1),
            #nn.BatchNorm3d(30),
            nn.ReLU(),
            #nn.MaxPool3d(kernel_size=2, stride=2)
            )

        #Fully connected layers
        self.fc1 = nn.Linear(1080, 1) 
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 1)     
        
    def forward(self, out):       
        out = self.conv_unit_1(out)
        out = self.conv_unit_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        #out = F.relu(out)
        #out = self.fc2(out)  
        #out = F.relu(out)
        #out = self.fc3(out)      
        #out = F.relu(out)                        
        return out




#MAIN FUNCTION
if __name__ == "__main__":
    
    path = os.path.dirname(__file__) + "/farm"
    
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
    
    
    #model = ConvNet() #See above

    for model in [LinNet()]:#, ConvNet()]:
    
        #Loss function
        criterion = nn.MSELoss() #CHANGE THIS
        optimizer = torch.optim.Adam(model.parameters())    
        print(model)
        
        
        train_nn(train_features, train_labels, num_epochs = 10)
    
    



