## RUL Estimation Code Base

## Instructions: 

Once you clone the repo, make sure 2nd_Test directory is unzipped. This file includes the raw data in csv form from NASA IMS Bearing Dataset (Dataset #2)
The entire dataset and its description can be also found and downloaded from here https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/ 
It is the fourth dataset on the web site.

## Python Files
1. utils.py: includes the helper function that is used in run_data.py.
2. run_data.py: reads the 2nd_Test directory raw data and performs the following operations.

It follows the algorithm from Ahmad et al. 2017 paper which is also included in this repo. Instead of Linear Rectification, I chose to utilize Gaussian smoothing. 
The rest of the algo code follows the procedures described in the paper. 
Also as a supporting evidence to compare the results, the authors second paper (also included in this repo) is chosen. In this paper (file name: Ahmad_Second_Paper.pdf)in Fig 2,
the authors shos the RUL estimation and the values related to the Dataset 2 and bearing1. The estimated numbers match the results generated with this code. The Results folder include 
Figure files generated via this code. 