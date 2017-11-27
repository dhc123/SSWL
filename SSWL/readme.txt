=============================SSWL Matlab Code============================

This package contains the source code for the following paper:
    Hao-Chen Dong and Yu-Feng Li and Zhi-Hua Zhou. Learning from Semi-SupervisedWeak-Label Data. AAAI 2018.
______________________________________________________________________________________
---- * Manual * ----------------------------------------------------------------------
The main algorithm is given in SSWL.m, whose manual is as following: 

   input varables:
       Train_Matrix: input feature vectors of training data and unlabeled data
       Train_Label: label matrix of training data and unlabeled data
       Test_Matrix: input feature vectors of test data
       alpha: the parameter controls smoothness of prediction
       beta: the parameter controls the consistence of two models
       zeta: the parameter controls the second model's prediction on uncertain elements
   output varables:
       pre_Test_labels: prediction on the test data
       W: coefficient matrix of first model
       W_star: coefficient matrix of second model
       L: label similarity matrix

_________________________________________________________________________________________
----- * DEMO * --------------------------------------------------------------------------
A demo script named 'demo_SSWL.m’ is provided. It runs SSWL on the yeast dataset. 

_________________________________________________________________________________________
---- * Attention* -----------------------------------------------------------------------
This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn).

This package was developed by Hao-Chen Dong. For any problem concerning the codes, please feel free to contact Mr. Dong (donghc@lamda.nju.edu.cn).
