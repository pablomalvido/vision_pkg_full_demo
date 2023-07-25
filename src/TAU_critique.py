import numpy as np
import cv2 as cv
import copy
import time
from typing import Tuple
from TAU_segmentation import TAUSegmentation
from TAU_img_functions import *
from TAU_cable_points_estimation import TAUCablePointsEstimation
from TAU_cable_line_estimation2 import TAUCableLineEstimation

from interfaces import TAUCritiqueInterface


class TAUCritique(TAUCritiqueInterface):
    '''
    Implements the result evaluation. In case the result is not successful it tunes the system parameters
    '''

    def __init__(self, pixel_D, thr1_init, thr2_init, erosion_init, window_size_init, forward = True, print=True, max_tries=6):
        self.print_info_iteration = print
        self.max_tries = max_tries
        self.pixel_D = pixel_D
        self.success = False
        self.result_error = 0
        self.prev_corr = 0
        self.try_n = 0
        self.n_miss_max = 10
        self.n_captured_points_last = 0
        self.n_segm_pixels_last = 1
        self.success_points_last = False
        self.thr1_last = thr1_init
        self.thr2_last = thr2_init
        self.erosion_last = erosion_init
        self.evaluation_window_last = window_size_init
        self.forward = forward
        self.cable_lkh_last = 0


    def cable_check(self, img, points) -> Tuple[bool, float]:    
        '''
        Checks if the detected cable actually looks like a cable. Basically it checks evaluation widnows in the segmented image in all the calculated points. 
        It checks first with winsows of the cable diameter size and then with double. If more than a 71.5% of the big window pixels are in the small window too, we consider it a cable
        '''
        img_points = copy.deepcopy(img)
        count_line_points = 0
        count_rectangles_points = 0
        sizeD = self.pixel_D + 1

        for i in range(len(points)-1):
            for row in range(int((points[i][0]+points[i+1][0])/2)-(sizeD*2), min(int((points[i][0]+points[i+1][0])/2)+(sizeD*2),img_points.shape[0])):
                for col in range(points[i][1], min(points[i+1][1], img_points.shape[1])):
                    if img_points[row][col]==255:
                        count_rectangles_points+=1

            ev_line_n = max(abs(points[i][0]-points[i+1][0]), abs(points[i][1]-points[i+1][1]))
            if ev_line_n != 0:
                for w in range(-int((sizeD+1)/2),int((sizeD+1)/2)):
                    step_y = (points[i+1][0]-points[i][0])/ev_line_n #3, 3/8
                    step_x = (points[i+1][1]-points[i][1])/ev_line_n #8, 1
                    disp_w_y = int(w*step_x)
                    disp_w_x = -int(w*step_y)
                    for n in range(ev_line_n):
                        new_point = [min(points[i][0]+int(step_y*n)+disp_w_y, img_points.shape[0]-1), min(points[i][1]+int(step_x*n)+disp_w_x, img_points.shape[1]-1)]
                        if img_points[new_point[0]][new_point[1]] == 255:
                            count_line_points += 1

        if count_rectangles_points > 0:
            line_likelihood = 100*(count_line_points/count_rectangles_points)
        else:
            line_likelihood = 0
        if self.print_info_iteration:
            print("Line likelihood: " + str(line_likelihood))
        if line_likelihood >= 71.5: #73
            isline = True
        else:
            isline = False
        #cv.imshow("Check", img_points)
        return isline, line_likelihood


    def exec(self, points_cable: list, segm_img: np.ndarray, n_segm_pixels: int, n_captured_points: int, success_points: bool, n_cables: int, thr1: float, thr2: float, evaluation_window: list, erosion: int, init_success: bool = True, count_no_borders: int = 0, count_free_steps: int = 0) -> Tuple[bool, int, float, float, int, list]:       
        
        break_signal = False
        self.try_n += 1
        if self.try_n >= self.max_tries:
            break_signal = True

        if success_points:
            is_cable, cable_lkh = self.cable_check(copy.deepcopy(segm_img), points_cable)
        else:
            cable_lkh = 0

        if (((n_captured_points/n_segm_pixels) < (self.n_captured_points_last/self.n_segm_pixels_last)) and (cable_lkh < self.cable_lkh_last)) or (self.success_points_last and not success_points): #Comes back to the old values
            thr1 = self.thr1_last
            thr2 = self.thr2_last
            erosion = self.erosion_last
            evaluation_window = self.evaluation_window_last
            n_captured_points = self.n_captured_points_last
            n_segm_pixels = self.n_segm_pixels_last
            success_points = self.success_points_last
        else:
            self.thr1_last = thr1
            self.thr2_last = thr2
            self.erosion_last = erosion
            self.evaluation_window_last = evaluation_window
            self.n_captured_points_last = n_captured_points
            self.n_segm_pixels_last = n_segm_pixels
            self.success_points_last = success_points
            self.cable_lkh_last = cable_lkh

        if (n_captured_points/n_segm_pixels)*n_cables < 0.5 and success_points:
            is_cable, cable_lkh = self.cable_check(copy.deepcopy(segm_img), points_cable)
            if is_cable:
                result_error = 0
                self.success = True #It is correct
            else:
                result_error = 1
            if self.print_info_iteration and not self.success:
                print("Too many points: " + str((n_captured_points/n_segm_pixels)*n_cables))
            #Too many points: detect less other colors, if fails again for this try decreasing the threshold and see if there is an improvement
            if self.prev_corr != 1 and self.prev_corr != 2 and self.prev_corr != 7 and thr1 >= 40:
                thr1 *= 0.85
                self.prev_corr = 1
            elif thr2 >= 1 and self.prev_corr == 1:
                thr2 *= 0.75
                self.prev_corr = 2
            elif self.prev_corr == 2 and self.n_miss_max<=6:
                self.n_miss_max += 1
                self.prev_corr = 7
            elif self.prev_corr == 7:
                erosion += 3
                self.prev_corr = 9
            elif self.prev_corr == 9 and evaluation_window[1]<=6*self.pixel_D:
                evaluation_window[0] += int(self.pixel_D*0.34) 
                evaluation_window[1] += int(self.pixel_D*0.5)
                self.prev_corr = 5

        elif not success_points or not init_success: #In BWP we consider init_success as True
            result_error = 2
            if (not (count_no_borders < 4 and count_free_steps < 4) or not init_success) or not self.forward: #If it is BWP we will do this option
                if self.print_info_iteration:
                    print("Fail due to too little points")
                if self.prev_corr != 1 and self.prev_corr != 3 and self.prev_corr != 4:
                    #thr1 *= 1.15
                    thr1 += 20
                    self.prev_corr = 3
                elif self.prev_corr != 4:
                    #thr2 *= 1.25
                    thr2 += 0.5
                    self.prev_corr = 4
                else:
                    if erosion > 0:
                        erosion -= 3
                        self.prev_corr = 8
                    else:
                        thr2 += 0.5
                        self.prev_corr = 3

            else: #probably fails due to an occlusion (entanglement):
                result_error = 3
                if self.print_info_iteration:
                    print("Fail due to occlusions")
                #increase kernel size and if the same problem happens detect more other colors
                if self.prev_corr != 5 and self.prev_corr != 6 and self.prev_corr != 7 and evaluation_window[1]<=4*self.pixel_D:
                    evaluation_window[1] += self.pixel_D
                    self.prev_corr = 5
                elif self.prev_corr == 5 and self.n_miss_max<=6:
                    self.n_miss_max += 1
                    self.prev_corr = 7
                elif self.prev_corr == 7:
                    thr2 *= 1.25
                    self.prev_corr = 6 #Not 3 so if can enter in 5 if it comes from 3
                else:
                    thr1 *= 1.15
                    self.prev_corr = 4

        else:
            result_error = 0
            self.success = True #It is correct

        if break_signal and not self.success: #Not correct but the max was reached
            self.success = True #But result_error != 0

        if not self.success:
            if self.print_info_iteration:
                print("Bad estimation")
        else:
            self.success = True
            if self.print_info_iteration:
                if result_error == 0:
                    print("Good estimation after: " + str(self.try_n) + " iterations")
                else:
                    print("Regular estimation")
        print("Threshold cable: " + str(thr1) + ", threshold color: " + str(thr2) + ", kernel size: " + str(evaluation_window))
        print("--------------------------------------------------")

        return self.success, result_error, thr1, thr2, erosion, evaluation_window #Just if success = True but result_error != 0, it means that there is still an error but it stopped due to iterations limit
