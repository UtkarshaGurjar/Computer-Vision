import numpy as np
def convert_to_grayscale(im): 
    return np.mean(im, axis = 2)
def filter_2d(im, kernel):
    M = kernel.shape[0] 
    N = kernel.shape[1]
    H = im.shape[0]
    W = im.shape[1]
    
    filtered_image = np.zeros((H-M+1, W-N+1), dtype = 'float64')

    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            image_patch = im[i:i+M, j:j+N]
            filtered_image[i, j] = np.sum(np.multiply(image_patch, kernel))
            
    return filtered_image

Kx = np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])

Ky = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]])
def classify(im):
    im = im[10:235, 10:235]
    gray = convert_to_grayscale(im/255.)
    Gx = filter_2d(gray, Kx) 
    Gy = filter_2d(gray, Ky) 
    G = np.sqrt(Gx**2+Gy**2)
    Gmean = np.mean(G)
    #print(np.mean(G))
    if(Gmean>0.11 and Gmean<0.21):
        return "brick"
    elif(Gmean>0.084 and Gmean<0.11):
        return "cylinder"
    return 'ball'