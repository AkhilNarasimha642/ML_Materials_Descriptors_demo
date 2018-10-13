## Materials Descriptors Machine Learning Demo

This a demo project to show various ways of encoding structural and chemical information of a material into a "materials descriptor" which is used to train a machine learning model that associates structo-chemical information with a materials' property. Given a new materials descriptor one can use the model to predict the corresponding property.

## Getting Started

In this project tehre are 3 ways of encoding structo-chemical information are presented: XRD300, OFM, CM_ES. You can learn more about the methods themselves at the end of this document. 
The input files are structural files in cif or POSCAR format. The output properties are in pandas format. Random Forest is used as a ml algorithm. 

The complete step by step instructions are below. To give a taste of the code several cif files from Materials Project with corresponding output properties taken from Mathematica are placed in the data directory.
           
Note. This is a demo/bare bones version of the project. Other descriptors as well as batching, post processing functions, plotting etc. are not included. This branch is only intended for demonstrations with up to 50,000 files. For the complete version of the project please contact Pavel Dolin at dolin@ucsb.edu. 

## Setting up

### Preparing directory input cif / POSCAR files (Skip if using provided demo files)
1. Go to /data and create a new directory with the name of the database. Remember the name as you will use it for next steps.
2. Place your output_props.csv (materials properties in pandas dataframe format) in database directory
3. In the newly created database directory create another directory called structure_files
4. Place your cif or POSCAR files in structure_files directrory

### Convert your cif/POSCAR files to descriptors of your choice
1. From your terminal go to Materials_Descriptors_Machine_Learning_demo/code/descriptors/construction/
2. Execute: python3 create_descriptors.py name_database descriptor_name
3. Your descriptor df will be storred in /data/name_of_database/descriptor_dfs/descriptor_name

### Machine learning
1. Go to /code/ml/ 
2. Run: python3 machine_learning.py name_database descriptor_name output_prop
3. Your results will be storred in 
/data/name_database/machine_learning/results/300trees/descriptor_name/output_prop/set_numer-set-split_fraction
4. By default every time you run machine_learning.py you create a new shuffled training/testing set
   if you want to chose a specific training/testing set specify it by -sn set_number
5. Similarly the train/test split is by default 0.6. To change that add -sf trainig_set_fraction.
6. So a complete the complete picture would look like this:
python3 machine_learning.py name_database descriptor_name output_prop -sn set_number -sf trainig_set_fraction

## Example                                             
For demo purposes cif files from Materials Project were added to mp database directory.
The associated properties in output_props.csv were also added and were are taken from Mathematica.
So for example. Go to the root of Materials_Descriptors_Machine_Learning_demo directory and run:

    python3 ./code/descriptors/construction/create_descriptors.py mp ofm
    python3 ./code/mlmachine_learning.py mp ofm Band_Gap
    
Your can find your results in this directory:

    ./data/mp/machine_learning/results/300trees/ofm/Band_Gap/0-set-0.6


## Prerequisites

### Python 3

- **SciPy** ([https://www.scipy.org](https://www.scipy.org)). The particular SciPy packages needed are:
    - **numpy**  ([http://www.numpy.org](http://www.numpy.org))
    - **pandas** ([http://pandas.pydata.org](http://pandas.pydata.org))

- **scikit-learn** ([http://scikit-learn.org](http://scikit-learn.org)) 

- **pymatgen** ([http://pymatgen.org](https://pandas.pydata.org/))

## Additional Info

### Descriptors papers

Descriptors coded by Pavel Dolin are based on:

OFM: https://aip.scitation.org/doi/abs/10.1063/1.5021089?journalCode=jcp

CM_ES: https://arxiv.org/pdf/1503.07406.pdf

XRD300: loosely based on https://journals.iucr.org/m/issues/2017/04/00/fc5018/index.html

### Authors

* **Pavel Dolin** - [dnieper](https://github.com/dnieper)

The reserch project was done by Pavel Dolin while working at Van der Ven group, Materials Department, University of California Santa Barbara.

### Demo files

Demo cif files are downloaded from Materials Project 
https://materialsproject.org/

Materials properties are taken from Wolfram
https://reference.wolfram.com/language/ref/ElementData.html