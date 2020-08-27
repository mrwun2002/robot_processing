# robot_processing


Robot Builder.py:
A general framework to read in robot, base, and report files and process them. Each robot is its own object, with its own properties. Each voxel in each robot is also an object, given a material type with properties determined by an associated palette/base file. Currently, the only properties extracted for each material are rgba values, but the code can easily be adapted to pull out other values as well. This palette/material structure can be implemented in future neural networks to allow for the modeling/prediction of robots with custom materials.


PyTorch Robot Prediction.py
A neural network to predict the total distance traveled by all voxels of a given robot. Runs two different models - the first one is a convolutional 3d neural network, and the second one is a standard, fully-connected deep neural network. The feature vectors are broken up into 5 channels of 0's and 1's, each corresponding to one of five materials. A 1 indicates that that material is present at that location. This is a version of one-hot encoding. 


NumPy Robot Prediction.py
A modification of the PyTorch Robot Prediction model to run only on NumPy/SKlearn. Pyodide can handle both of these packages, but not PyTorch. Only the linear neural network was replicated - Star's prior work showed that there was negligible difference between the accuracy of convolutional and linear models, and implementing a convolutional neural network without pytorch would have been an extremely time-consuming endeavor. However, it can be done and remains a possible direction for future work. 


Bundler.py and Template.py
These two files are related. Because files cannot be read in through UVMcomputes, these two files are used to output a file that can be run on UVMcomputes. The code used is that in NumPy Robot Prediction.py. Bundler.py processes all of the files, reads in code from Template.py, replaces tags in Template.py with strings representing robots and their distances traveled, and writes that in a new file called file_to_run.py. The package pyminifier is also used to reduce the size of file_to_run.py through minification and obfuscation, although it is not very useful - the smaller file ends up being over 99.9% the size of the original. This smaller file is called file_to_run_small.py.

