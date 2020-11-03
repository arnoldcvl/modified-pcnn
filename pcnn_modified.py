import sys
import numpy as np
import cv2 as cv
from scipy import signal
import math

Alpha_L = 0.1
Alpha_T = 0.5

V_T = 1.0

Num = 10
Beta = 0.1

T_extra = 63

link_arrange=9
center_x=round(link_arrange/2)
center_y=round(link_arrange/2)
W = np.zeros((link_arrange, link_arrange), np.float)
for i in range(link_arrange):
    for j in range(link_arrange):
        if (i==center_x) and (j==center_y):
            W[i,j]=1
        else:
            W[i,j]=1/math.sqrt(((i)-center_x)**2+((j)-center_y)**2)

def main(argv):
    default_file = 'img/3.jpeg'
    filename = argv[0] if len(argv) > 0 else default_file
    
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    src = cv.normalize(src.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    cv.imshow("Original", src)

    R = src[:,:,0]
    G = src[:,:,1]
    B = src[:,:,2]
    Intensity = np.divide(R + G + B, 3)

    dim = Intensity.shape

    F = np.zeros( dim, np.float)
    L = np.zeros( dim, np.float)
    Y = np.zeros( dim, np.float)
    T = np.ones( dim, np.float) + T_extra
    Y_AC = np.zeros( dim, np.float)
    
    #normalize image
    S = cv.normalize(Intensity.astype('float'), None, 0.0, 64.0, cv.NORM_MINMAX)  
    
    for cont in range(Num):
        #numpy.convolve(W, Y, mode='same')
        L = Alpha_L * signal.convolve2d(Y, W, mode='same')
        U = S * (1 + Beta * L)

        YC = 1 - Y      
        T = T - Alpha_T
        T = ((Y*T)*V_T) + (YC*T)

        Y = (U>T).astype(np.float)
        Y_AC = Y_AC + Y
    
    cv.imshow("Result", Y)
    Y_AC = cv.normalize(Y_AC.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    cv.imshow("Result ac", Y_AC)
        
    cv.waitKey()
    return 0
        
if __name__ == "__main__":
    main(sys.argv[1:])
