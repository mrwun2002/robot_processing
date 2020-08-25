# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:43:00 2020

@author: 18028
"""


#IMPORT PACKAGES
from lxml import etree


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #Necessary for 3D graphing


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
        
        


class Material():#This will be updated.
    '''
    Materials are stored/created in Palettes and assigned to Voxels after creating Robots.
    Reads in an etree element.
    '''
    def __init__(self, material_code = None):#Takes in an Element object
        
        self.display = {}
        self.mechanical = {}
        
        if material_code == None:
            identifier = '0'
            self.display["red"] = 0
            self.display["green"] = 0
            self.display["blue"] = 0
            self.display["alpha"] = 0
            
            self.mechanical["placeholder"] = "meh"
        
        else:
            identifier = material_code.get("ID")
            
            self.display["red"] = float(material_code.xpath("Display/Red")[-1].text)
            self.display["green"] = float(material_code.xpath("Display/Green")[-1].text)
            self.display["blue"] = float(material_code.xpath("Display/Blue")[-1].text)
            self.display["alpha"] = float(material_code.xpath("Display/Alpha")[-1].text)
            
            
            
        self.symbol = identifier
        self.rgba = (self.display["red"], self.display["green"], self.display["blue"], self.display["alpha"])
        
        

class Palette():
    '''
    Really just a dictionary of Materials, indexed by their symbols.
    Separate from the environment as a whole so that multiple palettes
    can be used within the same environment.
    '''
    def __init__(self, palette_file):
        self.materials = {
                '0': Material()    
            }
        
        palette_code = etree.parse(palette_file)
        elements = palette_code.xpath("//Palette/Material")
        for element in elements:
            self.materials[element.get("ID")] = Material(element)


class Voxel():
    '''
    Voxels are components of each Robot, stored in the set "voxels"
    '''
    def __init__(self, material: str, palette: Palette, x: int, y: int, z: int):
        try:
            self.material = palette.materials[material]
        except:
            sys.exit("A character in the genome is not defined in the given palette!")
        self.x = x
        self.y = y
        self.z = z
    def __repr__(self):
        return "{} at ({}, {}, {})".format(self.material, self.x, self.y, self.z)


class Robot():
    '''
    Pass in the name for the robot, a source file, and a Palette to apply to the robot.
    Has properties name, short_name, palette, dimensions, voxels, and genome.
    When printed, prints the genome along with the name.        
    '''
    def __init__(self, name: str, robot_file: str, palette: Palette):
        self.name = name
        
        #Short_name removes the folder and the file type - IS NOT UNIQUE
        self.short_name = (self.name.split('/')[-1]).split('.')[0]
        
        self.palette = palette
        
        robot_code = etree.parse(robot_file)
        

        self.dimensions = {
            "x": int(robot_code.xpath("//Structure/X_Voxels")[-1].text),
            "y": int(robot_code.xpath("//Structure/Y_Voxels")[-1].text),
            "z": int(robot_code.xpath("//Structure/Z_Voxels")[-1].text)
            }
        

       #Start with an empty set of voxels. As the genome is parsed through, add to this set.
        self.voxels = []
        
        #Create an (x*y) by z NumPy array, filled with the value None.
        #If the value None is retained, an error has been made.
        self.genome = np.full(
            (self.dimensions["z"], self.dimensions["x"]*self.dimensions["y"]),
            None)
        
        #Takes the genome from the XML file and puts it into self.genome
        
        layers = robot_code.xpath("//Structure/Data/Layer")
        layer_num = 0
        for layer in layers:
            char_num = 0
            for char in layer.text:
                try:
                    self.genome[layer_num, char_num] = char
                    #Create a voxel object and add it to the set "voxels"
                    new_voxel = Voxel(char, self.palette, 
                                      char_num%self.dimensions['x'],
                                      char_num//self.dimensions['y'],
                                      layer_num)
                    self.voxels.append(new_voxel)
                except IndexError:
                        sys.exit("Genome is larger than dimensions given!")
                
                char_num += 1
            layer_num += 1
            
    
    
    
    def visualize(self):
        '''
        Uses matplotlib to create a 3D visualization of the robot.
        '''
        viz_matrix = np.zeros((self.dimensions['x'], self.dimensions['y'], self.dimensions['z']))
        
        color_matrix = np.zeros((self.dimensions['x'], self.dimensions['y'], self.dimensions['z'], 4))
        
        for voxel in self.voxels:
            viz_matrix[voxel.x, voxel.y, voxel.z] = voxel.material.symbol != '0'
            color_matrix[voxel.x, voxel.y, voxel.z] = voxel.material.rgba

        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(viz_matrix, facecolors = color_matrix, edgecolor='k')
        plt.title(self.name)
        
        plt.show()

    
    
    def __repr__(self): #When a robot is printed, print "name: genome" and visualize. 
        
        print("\n{}:\n{}".format(self.name, self.genome))
        
        self.visualize()
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
    A tuple of (robot_files: dict, palette_files: dict, report_files: dict)
    Each are indexed by what makes them unique: in the case of robot_files, 
    this is the folder/file name. In the case of the other 2, it is just 
    the folder.
    THIS FUNCTION ASSUMES THAT ROBOTS ARE .vxd, REPORTS ARE .xml, AND 
    BASE FILES ARE .vxa

    '''
    robot_files = {}
    base_files = {}
    report_files = {}
    
    folders = []
    
    for folder in os.scandir(path):
        folders.append(folder.name)
        
    
    if all_files == False:
        folders = [folders[0]]
        
    
    for folder in folders:
        path_branch = path + "/" + folder
        
                
        for file in os.scandir(path_branch):
            
            if file.name.endswith(".vxa"):
                base_files[folder] = path + "/" + folder + "/" + file.name
            elif file.name.endswith(".xml"):
                report_files[folder] = path + "/" + folder + "/" + file.name
            elif file.name.endswith(".vxd"):
                robot_files[folder + "/" + file.name] = path + "/" + folder + "/" + file.name
            else: 
                sys.exit("Unknown file type in folder {}".format(folder.name))
            
    return (robot_files, base_files, report_files)


def create_palettes(palette_files: dict):
    '''
    Parameters
    ----------
    palette_files : dict
        A dictionary containing the paths to the files.

    Returns
    -------
    Duplicated dictionary of palette_files with the same keys, but 
    the palettes objects instead of the strings containing the file path as 
    the values.
    '''

    palettes = {}
    
    for key in palette_files:
        palettes[key] = Palette(palette_files[key])
    
    return palettes


def create_reports(report_files: dict):
        
    reports = {}
    
    for key in report_files:
        reports[key] = Report(report_files[key])
        
    return reports
        


def build_robots(robot_files: dict, palettes: dict):
    '''
    
    Parameters
    ----------
    robot_files : dict
        Dictionary containing all robot file paths, indexed by the final folder and 
        the file name.
    palettes : dict
        Dictionary containing all palettes, indexed by the folder.

    Returns
    -------
    robots : list
        List containing robot objects.

    '''    
    robots = []
    
    for key in robot_files:
        robot_file_folder = key.split('/')[0]
        new_robot = Robot(key, robot_files[key], palettes[robot_file_folder])
        robots.append(new_robot)
        
    return robots


def get_train_features(robots: list):
    
    #5D array - 1st dimension is each robot, 2nd is each material channel,
    #3-5 correspond to X, Y, and Z
    train_features = np.zeros((len(robots), 
                              len(robots[0].voxels[0].material.mechanical),
                              robots[0].dimensions["x"],
                              robots[0].dimensions["y"],
                              robots[0].dimensions["z"]))
    
    return train_features


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
    train_labels : list
        A list of all of the labels, in the same order as the robots.

    '''
    
    train_labels = []
    
    for robot in robots:
        new_label = reports[robot.name.split('/')[0]].distances[robot.short_name]
        train_labels.append(new_label)
    
    return train_labels



#MAIN FUNCTION
if __name__ == "__main__":
    
    path = os.path.dirname(__file__) + "/farm"
    
    #Get the files
    robot_files, base_files, report_files = get_files(path)
    print("All files retrieved.")
    
    
    #Create the palettes and reports - this is done before breaking out into robots 
    #to avoid creating many identical palettes and reports
    palettes = create_palettes(base_files)
    print("All palettes created.")
    reports  = create_reports(report_files)
    print("All reports created.")
    
    robots = build_robots(robot_files, palettes)
    print("All robots built.")
    
    train_features = get_train_features(robots)
    train_labels = get_train_labels(robots, reports)
    print("Features and labels extracted.")
    
    
    print(np.argmax(train_labels))
    print(np.max(train_labels))
    print(robots[np.argmax(train_labels)])
    
    for i in range(len(robots)):
        print(robots[i])
        print("Distance traveled by all voxels: {}".format(train_labels[i]))
        print("\n")

    




