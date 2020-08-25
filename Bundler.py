# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:43:00 2020

@author: Nathan Wu

Tries to minimally process the stuff it's reading in
"""


#IMPORT PACKAGES
from lxml import etree

import os #Used for file parsing

#THIS CODE USED PYMINIFIER
#https://liftoff.github.io/pyminifier/index.html#


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
            self.distances[robot_report.tag] = round(float(
                robot_report.xpath("total_distance_of_all_voxels")[-1].text), ndigits=6)
        
        



class Robot():
    '''
    Pass in the name for the robot, a source file. Has the name, a "short name", and the genome as a string.       
    '''
    def __init__(self, name: str, robot_file: str):
        self.name = name
        
        #Short_name removes the folder and the file type - IS NOT UNIQUE
        self.short_name = (self.name.split('/')[-1]).split('.')[0]
    
        
        robot_code = etree.parse(robot_file)
        
        '''
        self.dimensions = {
            "x": int(robot_code.xpath("//Structure/X_Voxels")[-1].text),
            "y": int(robot_code.xpath("//Structure/Y_Voxels")[-1].text),
            "z": int(robot_code.xpath("//Structure/Z_Voxels")[-1].text)
            }
        '''
      
        self.genome_as_string = ""
        #Takes the genome from the XML file and puts it into self.genome
        
        layers = robot_code.xpath("//Structure/Data/Layer")
        
        for layer in layers:
            self.genome_as_string += str(layer.text)


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

def get_labels_as_string(robots: list, reports: dict):
    '''
    
    Parameters
    ----------
    robots : list
        A list of all robots.
    reports : dict
        A dictionary, indexed by folder, cotaining respective Report objects.

    Returns
    -------
    labels_as_string : str
        An string of all of the labels, each separated by a space.

    '''
    
    labels_as_string = ""
    
    for robot in robots:
        new_label = reports[robot.name.split('/')[0]].distances[robot.short_name]
        labels_as_string += str(new_label)
        labels_as_string += " "
    
    return labels_as_string[:-1]


def get_genomes_as_string(robots: list):
    """Retrieve the genomes and put them in a np array.

    Args:
        robots (list): List of the robots.

    Returns:
        genomes_as_string : An string containing the genomes of the robots, in order, separated by a space.
    """    
    genomes_as_string = ""

    for robot in robots:
        genomes_as_string += robot.genome_as_string
        genomes_as_string += " "


    return genomes_as_string[:-1]



if __name__ == "__main__":
    
    

    path = os.path.dirname(__file__) + "/"
    robot_path = path + "farm"
    robot_files, report_files = get_files(robot_path, all_files = False)
    #Robot files - need to extract the genome and put it into a list (in order)


    #Create the reports - this is done before breaking out into robots 
    #to avoid creating many identical reports
    reports  = create_reports(report_files)
    print("All reports created.")
    
    robots = build_robots(robot_files)
    print("All robots built.")


    labels_as_string = get_labels_as_string(robots, reports)
    genomes_as_string = get_genomes_as_string(robots)
    

    with open(path + 'Template.py', 'r') as template, open(path + '/file_to_run.py', 'w') as new: 
        code = template.read()

        code = code.replace("<genomes_as_string>", genomes_as_string)

        code = code.replace("<labels_as_string>", labels_as_string)

        new.write(code)

    #If pyminifier is not present, no worries - just run the file_to_run
    os.system('pyminifier --obfuscate --outfile={}file_to_run_small.py {}file_to_run.py'.format(path, path))
    