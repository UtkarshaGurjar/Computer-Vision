## ---------------------------- ##
## 
## sample_student.py
##
## Example student submission for programming challenge. A few things: 
## 1. Before submitting, change the name of this file to your firstname_lastname.py.
## 2. Be sure not to change the name of the method below, count_fingers.py
## 3. In this challenge, you are only permitted to import numpy, and methods from 
##    the util module in this repository. Note that if you make any changes to your local 
##    util module, these won't be reflected in the util module that is imported by the 
##    auto grading algorithm. 
## 4. Anti-plagarism checks will be run on your submission
##
##
## ---------------------------- ##


import numpy as np
import sys
def breakIntoGrids(im, s = 9):
    '''
    Break overall image into overlapping grids of size s x s, s must be odd.
    '''
    grids = []

    h = s//2 #half grid size minus one.
    for i in range(h, im.shape[0]-h):
        for j in range(h, im.shape[1]-h):
            grids.append(im[i-h:i+h+1,j-h:j+h+1].ravel())

    return np.vstack(grids)

def reshapeIntoImage(vector, im_shape, s = 9):
    '''
    Reshape vector back into image. 
    '''
    h = s//2 #half grid size minus one. 
    image = np.zeros(im_shape)
    image[h:-h, h:-h] = vector.reshape(im_shape[0]-2*h, im_shape[1]-2*h)

    return image

def count_fingers(im):
    '''
    Example submission for coding challenge. 
    
    Args: im (nxm) unsigned 8-bit grayscale image 
    Returns: One of three integers: 1, 2, 3
    
    '''

    ## ------ Input Pipeline Develped in this Module ----- ##
    #You may use the finger pixel detection pipeline we developed in this module:
    #You may also replace this code with your own pipeline if you prefer
    im = im > 92 #Threshold image
    im = im[5:30, 5:33]
    X = breakIntoGrids(im, s = 9) #Break into 9x9 grids

    #Use rule we learned with decision tree
    treeRule1 = lambda X: np.logical_and(np.logical_and(X[:, 40] == 1, X[:,0] == 0), X[:, 53] == 0)
    yhat = treeRule1(X)

    #Reshape prediction ino image:
    yhat_reshaped = reshapeIntoImage(yhat, im.shape)

    ## ----- Your Code Here ---- ##
    count=0
    i=0
    #print(np.where(yhat_reshaped>0))
    while(i<len(yhat_reshaped[0])-1):
        if(yhat_reshaped[12][i]>0  or yhat_reshaped[12][i+1]>0 or yhat_reshaped[18][i]>0 or yhat_reshaped[18][i+1]>0):
            if(yhat_reshaped[12][i+2]>0 or yhat_reshaped[18][i+2]>0):
                count=count+1
                i=i+6
        i=i+1
    return count