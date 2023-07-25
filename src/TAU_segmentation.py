import numpy as np
import cv2 as cv
import copy
import time
from typing import Tuple

from interfaces import TAUSegmentationInterface


class TAUSegmentation(TAUSegmentationInterface):
    '''
    Implements the segmentation via Dilate(edges) AND color_filter 
    '''

    def __init__(self, input_img: np.ndarray, all_colors: list, init_col: int = 0):

        self.img = input_img
        self.all_colors = all_colors
        self.init_col = init_col
        canny = cv.Canny(copy.deepcopy(self.img), 125, 175)
        self.canny_dilated = cv.dilate(copy.deepcopy(canny), (9,9), iterations=3)


    def exec(self, color_cable: list, thr1: float = 60, thr2: float = 1.5,erosion: int = 3) -> Tuple[np.ndarray, list]:

        b,g,r = cv.split(self.img)
        blank_cables = np.zeros((self.img.shape[0], self.img.shape[1]), dtype='uint8')
        compare_thr = ((thr1*0.1)**2)*3 #Equivalent to a 10% diff in each of the 3 color channels
        for row in range(self.img.shape[0]):
            for col in range(self.img.shape[1]):
                if self.canny_dilated[row, col]==255: #To not check all the pixels and safe time, as we will do a intersection of color filter and dilated edge detection
                    if abs(b[row][col] - color_cable[0])<thr1 and abs(g[row][col] - color_cable[1])<thr1 and abs(r[row][col] - color_cable[2])<thr1:
                        color_dif = (b[row][col] - color_cable[0])**2 + (g[row][col] - color_cable[1])**2 + (r[row][col] - color_cable[2])**2
                        #color_dif = abs(b[i][j] - color_cable[0]) + abs(g[i][j] - color_cable[1])+ abs(r[i][j] - color_cable[2])
                        other_color = False
                        if color_dif > compare_thr: #If the color is not that similar, we compare if it was confused with other cable
                            for color_ch in self.all_colors:
                                if color_ch != color_cable:
                                    color_dif_ch = ((b[row][col] - color_ch[0])**2 + (g[row][col] - color_ch[1])**2 + (r[row][col] - color_ch[2])**2)*thr2 #The higher admits more colors
                                    #color_dif_ch = (abs(b[i][j] - color_ch[0]) + abs(g[i][j] - color_ch[1]) + abs(r[i][j] - color_ch[2]))*more_colors_thr #The higher admits more colors
                                    if color_dif_ch < color_dif:
                                        other_color = True
                                        break
                        if not other_color:
                            blank_cables[row][col] = 255
                        else:
                            blank_cables[row][col] = 0
                    else:
                        blank_cables[row][col] = 0
        
        if erosion > 8:
            iterations = 2
        else:
            iterations = 1
        if erosion <= 0:
            color_filter_eroded = copy.deepcopy(blank_cables)
        else:
            color_filter_eroded = cv.erode(copy.deepcopy(blank_cables), (erosion,erosion), iterations=iterations)

        segm_img = copy.deepcopy(color_filter_eroded)

        segm_pixels = []
        for row in range(segm_img.shape[0]):
            for col in range(segm_img.shape[1]):
                if segm_img[row][col] == 255:
                    segm_pixels.append([row,col])

        return segm_img, segm_pixels
