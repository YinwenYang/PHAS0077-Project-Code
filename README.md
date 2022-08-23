# PHAS0077-Project-Code
All the code for my PHAS0077 Project is placed in this repository.  
This repository includes three .py files and a folder. These code are using for generating the dataset, plotting how does the data look like, building random forest regression model and testing the accuracy of line selection implemented by the regression model.  

Environment: Python 3.8.8 or Jupyter-notebook 6.4.8  

## Note
1. The programs import two Python modules: UCLCHEM and SpectralRadex, that need to be installed before running, otherwise, it is likely to get some errors.  

2. In 'And example plot'folder, there are three files:  
(1) csv file is output data of one UCLCHEM model;  
(2) 'Example_plot.ipynb' is a jupyter notebook file that will plot how does the data look like;  
(3) And the plot is saved into a png file.  
Just an example to give a rough feeling about the UCLCHEM, so we use a very simple grid to do it, with a few parameters and species.  
Therefore, running this part is not necessarily required for this project.  

## Workflow
The running order of the code files are as follows:  
1. Generate_grids.py
2. RFR_Modelling.py
3. 2tests_for_RFR.py  

## How to run
First, starting with 'Generate_grids.py' file to generate the dataset. The code in this file will first use the UCLCHEM model to run over a large grid of parameter combinations, the outputs are chemical abundances of various species. Then with some calculations, the processed output data are taken as the inputs of the RADEX model to run RADEX. The ultimate outputs will be produced and saved into a csv file called 'Results.csv', and it is the dataset for this project, the other two .py files are both based on this dataset. The 'Results.csv' file is not contained in the repository, but after running Generate_grids.py, it will be automatically generated.  

Secondly, run 'RFR_Modelling.py' to do some random forest regression modelling. It will print the score and mean squared error of the model, produce the plot of predicted values, and provide the importance of each line and rank these lines.  

Finally, run '2tests_for_RFR.py' to perform two tests which test the accuracy of line selection implemented by the random regression models. It will calculate the rate of change of each line, and rank these lines based on their rate of change (absolute values) in descending order.  
