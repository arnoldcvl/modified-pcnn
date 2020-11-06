# Copyright (C) Leandro A. F. Fernandes and Manuel M. Oliveira
#
# author     : Fernandes, Leandro A. F.
# e-mail     : laffernandes@ic.uff.br
# home page  : http://www.ic.uff.br/~laffernandes
# 
# This file is part of the reference implementation of the Kernel-Based
# Hough Transform (KHT). The complete description of the implemented
# techinique can be found at:
# 
#     Leandro A. F. Fernandes, Manuel M. Oliveira
#     Real-time line detection through an improved Hough transform
#     voting scheme, Pattern Recognition (PR), Elsevier, 41:1, 2008,
#     pp. 299-314.
# 
#     DOI.........: https://doi.org/10.1016/j.patcog.2007.04.003
#     Project Page: http://www.ic.uff.br/~laffernandes/projects/kht
#     Repository..: https://github.com/laffernandes/kht
# 
# KHT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# KHT is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with KHT. If not, see <https://www.gnu.org/licenses/>.

import cv2, os
from kht import kht
from math import cos, sin, radians
from matplotlib import pyplot as plt
from os import path


trackbar_name_KHT_one = 'cluster minSize D:10'
trackbar_name_KHT_two = 'cluster_min_deviation/10 D:20'
trackbar_name_KHT_three = 'Amount of revelant lines + 10 D:0'
title_window = 'Parameters'
alpha_slider_max = 255

# The main function.
def main():
    global im, bw, filenames, relevant_lines, base_folder
    # Set sample image files and number of most relevant lines.
    base_folder = path.dirname(os.path.abspath(__file__))
    filenames = ["result.jpg", "result Acumulated.jpg"]
    relevant_lines = [10,10]

    cv2.namedWindow(title_window)
    cv2.createTrackbar(trackbar_name_KHT_one, title_window , 10, alpha_slider_max, on_trackbar)
    cv2.createTrackbar(trackbar_name_KHT_two, title_window , 20, alpha_slider_max, on_trackbar)
    cv2.createTrackbar(trackbar_name_KHT_three, title_window , 0, alpha_slider_max, on_trackbar)

    on_trackbar(0)

    cv2.waitKey()

def on_trackbar(val):
    print('xd')
        # Process each one of the images.
    for (filename, lines_count) in zip(filenames, relevant_lines):
        # Load input image.
        im = cv2.cvtColor(cv2.imread(path.join(base_folder, filename)), cv2.COLOR_BGR2RGB)

        # Convert the input image to a binary edge image.
        bw = cv2.Canny(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), 80, 200)
        #cv2.imwrite('canny.jpg', bw)
        height, width, _ = im.shape

        KHT_cluster_min_size = max(cv2.getTrackbarPos(trackbar_name_KHT_one, title_window),2)
        KHT_cluster_deviation = (cv2.getTrackbarPos(trackbar_name_KHT_two, title_window))/10.0
        extra_lines = cv2.getTrackbarPos(trackbar_name_KHT_three, title_window)
        # Call the kernel-base Hough transform function.
        lines = kht(bw, cluster_min_size=KHT_cluster_min_size, cluster_min_deviation=KHT_cluster_deviation)

        # Show current image and its most relevant detected lines.

        for (rho, theta) in lines[:lines_count + extra_lines]:
            theta = radians(theta)
            cos_theta, sin_theta = cos(theta), sin(theta)

            h2 = height/2
            w2 = width/2
            if sin_theta != 0:
                one_div_sin_theta = 1 / sin_theta
                x = (round(0) , round(h2 + (rho + w2 * cos_theta) * one_div_sin_theta))
                y = (round(w2+w2) , round(h2 + (rho - w2 * cos_theta) * one_div_sin_theta))
            else:
                x = (round(w2 + rho), round(0))
                y = (round(w2 + rho), round(h2+h2))

            cv2.line(im, x, y, (120,0,255), 1, cv2.LINE_AA)

        cv2.imshow("Result of %s" %filename, im)
        #cv2.imwrite('arnold kht %s.jpg' %filename, im)

if __name__ == "__main__":
    main()