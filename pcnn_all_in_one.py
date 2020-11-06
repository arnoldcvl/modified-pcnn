import sys
import numpy as np
import cv2 as cv
from scipy import signal
from math import cos, sin, radians, sqrt

from matplotlib import pyplot as plt
from kht import kht

'''
Alpha_L = 0.1
Alpha_T = 0.5

V_T = 1.0

Num = 10
Beta = 0.1

T_ini = 63
'''

link_arrange=9
center_x=round(link_arrange/2)
center_y=round(link_arrange/2)
W = np.zeros((link_arrange, link_arrange), np.float)
for i in range(link_arrange):
    for j in range(link_arrange):
        if (i==center_x) and (j==center_y):
            W[i,j]=1
        else:
            W[i,j]=1/sqrt(((i)-center_x)**2+((j)-center_y)**2)

alpha_slider_max = 255

title_window = 'Parameters'
trackbar_name_Beta = 'Beta/100 D:10'
trackbar_name_Num = 'Num D:10'

trackbar_name_Tini = 'Initial T D:63'
trackbar_name_VT = 'VT/100 D:100'

trackbar_name_AL = 'AL/100 D:10'
trackbar_name_AT = 'AT/100 D:50'

trackbar_name_Lines = 'Lines D:10'
trackbar_name_KHT_one = 'cluster minSize D:10'
trackbar_name_KHT_two = 'cluster_min_deviation/10 D:20'

trackbar_name_save = 'Save image on 1'

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

    #Generate I component from HSI
    R = src[:,:,0]
    G = src[:,:,1]
    B = src[:,:,2]
    Intensity = np.divide(R + G + B, 3)

    #normalize image
    global S
    S = cv.normalize(Intensity.astype('float'), None, 0.0, 64.0, cv.NORM_MINMAX) 

    cv.namedWindow(title_window)
    cv.createTrackbar(trackbar_name_Beta, title_window , 10, alpha_slider_max, on_trackbar)
    cv.createTrackbar(trackbar_name_Num, title_window , 10, alpha_slider_max, on_trackbar)

    cv.createTrackbar(trackbar_name_Tini, title_window , 63, alpha_slider_max, on_trackbar)
    cv.createTrackbar(trackbar_name_VT, title_window , 100, alpha_slider_max, on_trackbar)

    cv.createTrackbar(trackbar_name_AL, title_window , 10, alpha_slider_max, on_trackbar)
    cv.createTrackbar(trackbar_name_AT, title_window , 50, alpha_slider_max, on_trackbar)

    cv.createTrackbar(trackbar_name_Lines, title_window , 10, alpha_slider_max, on_trackbar)
    cv.createTrackbar(trackbar_name_KHT_one, title_window , 10, alpha_slider_max, on_trackbar)
    cv.createTrackbar(trackbar_name_KHT_two, title_window , 20, alpha_slider_max, on_trackbar)

    cv.createTrackbar(trackbar_name_save, title_window , 0, 1, on_trackbar)


    on_trackbar(0)
        
    cv.waitKey()

def on_trackbar(val):
    #Get trackbar values
    Beta = (cv.getTrackbarPos(trackbar_name_Beta, title_window))/100
    Num = cv.getTrackbarPos(trackbar_name_Num, title_window)

    T_ini = (cv.getTrackbarPos(trackbar_name_Tini, title_window))
    V_T = (cv.getTrackbarPos(trackbar_name_VT, title_window))/100

    Alpha_L = (cv.getTrackbarPos(trackbar_name_AL, title_window))/100
    Alpha_T = (cv.getTrackbarPos(trackbar_name_AT, title_window))/100

    lines_count = cv.getTrackbarPos(trackbar_name_Lines, title_window)
    KHT_cluster_min_size = cv.getTrackbarPos(trackbar_name_KHT_one, title_window)
    KHT_cluster_deviation = (cv.getTrackbarPos(trackbar_name_KHT_two, title_window))/10.0

    if KHT_cluster_min_size < 2.0: KHT_cluster_min_size=2.0

    #initialization
    dim = S.shape

    F = np.zeros( dim, np.float)
    L = np.zeros( dim, np.float)
    Y = np.zeros( dim, np.float)
    T = np.ones( dim, np.float) + T_ini
    Y_AC = np.zeros( dim, np.float)  
     
    #PCNN Modified
    for cont in range(Num):
        #numpy.convolve(W, Y, mode='same')
        L = Alpha_L * signal.convolve2d(Y, W, mode='same')
        U = S * (1 + Beta * L)

        YC = 1 - Y      
        T = T - Alpha_T
        T = ((Y*T)*V_T) + (YC*T)

        Y = (U>T).astype(np.float)
        Y_AC = Y_AC + Y
    
    Y = (Y*255).astype(np.uint8)
     
    bw_Y = cv.Canny(Y, 80, 200)
    canny_copy = bw_Y.copy()
    
    height, width = Y.shape
    lines_Y = kht(bw_Y, cluster_min_size=KHT_cluster_min_size, cluster_min_deviation=KHT_cluster_deviation)

    for (rho, theta) in lines_Y[:lines_count]:
        
        theta = radians(theta)
        cos_theta, sin_theta = cos(theta), sin(theta)
        '''
        if sin_theta != 0:
            x = (-width / 2, width / 2 - 1)
            y = ((rho - x[0] * cos_theta) / sin_theta, (rho - x[1] * cos_theta) / sin_theta)
        else:
            x = (rho, rho)
            y = (-height / 2, height / 2 - 1)
        x = (int(x[0] + width / 2), int(x[1] + width / 2))
        y = (int(y[0] + height / 2), int(y[1] + height / 2))
        '''
        h2 = height/2
        w2 = width/2
        if sin_theta != 0:
            one_div_sin_theta = 1 / sin_theta
            x = (round(0) , round(h2 + (rho + w2 * cos_theta) * one_div_sin_theta))
            y = (round(w2+w2) , round(h2 + (rho - w2 * cos_theta) * one_div_sin_theta))
        else:
            x = (round(w2 + rho), round(0))
            y = (round(w2 + rho), round(h2+h2))

        cv.line(Y, x, y, (120,0,255), 1, cv.LINE_AA)

    on_save(Y, canny_copy)
    cv.imshow("Result", Y)

def on_save(image_one, image_two):
    if cv.getTrackbarPos(trackbar_name_save, title_window) == 1:
        cv.imwrite('saving kht.jpg', image_one)
        cv.imwrite('saving canny.jpg', image_two)
        
if __name__ == "__main__":
    main(sys.argv[1:])
