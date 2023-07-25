import numpy as np
import cv2 as cv
import copy
import time
from typing import Tuple
from TAU_segmentation import TAUSegmentation
from TAU_img_functions import *
from TAU_cable_points_estimation import TAUCablePointsEstimation

from interfaces import TAUForwardPropagationInterface


class TAUForwardPropagation(TAUForwardPropagationInterface):
    '''
    Implements the forward propagation. A cable is calculated propagating its pixels forward
    '''

    def __init__(self, mm_per_pixel):
        self.points_estimation = TAUCablePointsEstimation()
        self.mm_per_pixel = mm_per_pixel

    def exec(self, segm_img: np.ndarray, segm_pixels: list, initial_point: list, window_size: list, cable_length: float, n_miss_max: int = 5, n_cables: int = 1) -> Tuple[list, int, bool, int, int, bool]:
        
        #Determine the real initial point, the segmented pixel that is closer to the theoretical pixel
        min_dist_init = 999
        init = [0,0]
        window_init = 20
        init_success = False
        for col in range(-window_init,window_init,1):
            for row in range(-window_init,window_init,1):
                if segm_img[max(initial_point[0]+row, 0)][max(initial_point[1]+col, 0)] == 255:
                    dist_init = row**2 + col**2
                    if dist_init < min_dist_init:
                        min_dist_init = dist_init
                        init = [max(initial_point[0]+row, 0), max(initial_point[1]+col, 0)]
                        init_success = True

        n_segm_pixels = len(segm_pixels)
        if not init_success:
            return [[initial_point[0], initial_point[1]-1],initial_point], 0, False, 0, 0, init_success
        
        last_x = int(min(segm_img.shape[1], initial_point[1] + (cable_length/self.mm_per_pixel)))
        points_cable, captured_points, captured_points_yx, success_points, count_no_borders, count_free_steps = self.points_estimation.exec(segm_img, init=init, last_x = last_x, window_size=window_size, n_miss_max = n_miss_max)
        #print("Percentage of captured points: " + str((captured_points/n_segm_pixels)*100) + "%, " + str(n_cables) + " cables")
        segm_points_img = copy.deepcopy(segm_img)

        if ((captured_points/n_segm_pixels)*n_cables*100) < 2 or not success_points: #Lower than 2%
            return points_cable, captured_points, False, count_no_borders, count_free_steps, init_success
        
        #cv.imshow("Points mask", segm_points_img)
        return points_cable, captured_points, success_points, count_no_borders, count_free_steps, init_success