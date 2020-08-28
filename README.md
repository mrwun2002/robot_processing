# robot_processing

### About:
This project is a continuation of https://github.com/liusida/DeepFinal (made by Star Liu).
Star's project explored the use of neural networks to predict the distance a voxelbot (a robot made from combinations of cubes of different materials) will travel given the structure/genome of the voxelbot. The purpose of this project is to make such a neural network runnable on UVMComputes, an in-browser supercomputer currently in development. UVMComputes in unable to take input from multiple files, so a file will be generated from a template and "bundler", which will take in file input and generate a string version of the data. That string will be substituted in for tags in the template, and new file will be created. 

![Diagram](https://i.ibb.co/LrzyHdy/image.jpg)

### Files:
#### Robot Builder.py
A general framework to read in robot, base, and report files and process them. Each robot is its own object, with its own properties. Each voxel in each robot is also an object, given a material type with properties determined by an associated palette/base file. Currently, the only properties extracted for each material are rgba values, but the code can easily be adapted to pull out other values as well. This palette/material structure can be implemented in future neural networks to allow for the modeling/prediction of robots with custom materials. This file also comes with a "visualize" function that plots the robots in 3d.

![Robot Builder Structure](https://i.ibb.co/xXt8ytc/image.png)


#### PyTorch Robot Prediction.py
A neural network to predict the total distance traveled by all voxels of a given robot. Runs two different models - the first one is a convolutional 3d neural network, and the second one is a standard, fully-connected deep neural network. The feature vectors are broken up into 5 channels of 0's and 1's, four of them corresponding to a material and one indicating no material at all. A 1 indicates that that material is present at that location. This is a version of one-hot encoding. 


#### NumPy Robot Prediction.py
A modification of the PyTorch Robot Prediction model to run only on NumPy/SKlearn. Pyodide can handle both of these packages, but not PyTorch. Only the linear neural network was replicated - Star's prior work showed that there was negligible difference between the accuracy of convolutional and linear models (https://github.com/liusida/DeepFinal), and implementing a convolutional neural network without pytorch would have been an extremely time-consuming endeavor. However, it can be done and remains a possible direction for future work. Also, the linear neural network has lots of different options/possible modifications because it is using SKLearn - see documentation here. 
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html


#### Bundler.py and Template.py
These two files are related. Because files cannot be read in through UVMcomputes, these two files are used to output a file that can be run on UVMcomputes. The code used is that in NumPy Robot Prediction.py. Bundler.py processes all of the files, reads in code from Template.py, replaces tags in Template.py with strings representing robots and their distances traveled, and writes that in a new file called file_to_run.py. The package pyminifier is also used to reduce the size of file_to_run.py through minification and obfuscation, although it is not very useful - the smaller file ends up being over 99.9% the size of the original. This smaller file is called file_to_run_small.py.
Currently, when file_to_run.py is run, it does not give any output - it merely gives confirmation that the model has been trained. Also, there are a number of outputs that aren't shown when run on UVMComputes but appear when run on your own computer. In the future, the bundler file should be made to read in test robots to make predictions on and a csv should be outputted. 

