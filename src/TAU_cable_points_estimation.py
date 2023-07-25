import numpy as np
import cv2 as cv
import copy
import time
from typing import Tuple
from TAU_img_functions import *

from interfaces import TAUCablePointsEstimationInterface


class TAUCablePointsEstimation(TAUCablePointsEstimationInterface):
    '''
    Implements the estimation of the cable points. It propagates its points starting from a point
    '''

    def __init__(self):

        pass


    def evaluate_first_point(self, img: np.ndarray, init: list, window_size: list) -> list:
        '''
        Evaluates where is the second point, this point is calculated different as we don't know the cable direction yet. It selects the pixel of the border of a rectangle (window_size) with more pixels in a straight line connecting it with the initial point
        '''
        window_size = [window_size[0]*3, window_size[1]*2]
        evaluated_points = []
        for i in range(-window_size[0], window_size[0]+1, 1): #i == down-up rows
            for j in range(int(window_size[1]*0.7), int(window_size[1]*1.3)+1, 1):
                new_point = [min(init[0] + i, img.shape[0]-1), min(init[1] + j, img.shape[1]-1)]
                if img[new_point[0]][new_point[1]] == 255:
                    evaluated_points.append(new_point)
        for i in range(int(window_size[1]/2), window_size[1]+1, 1):
            new_point = [min(init[0] + window_size[0], img.shape[0]-1), min(init[1] + i, img.shape[1]-1)]
            if img[new_point[0]][new_point[1]] == 255:
                evaluated_points.append(new_point)
            new_point = [min(init[0] - window_size[0], img.shape[0]-1), min(init[1] + i, img.shape[1]-1)]
            if img[new_point[0]][new_point[1]] == 255:
                evaluated_points.append(new_point)

        max_ev_line_points = 0
        if len(evaluated_points)>0:
            best_second = evaluated_points[0]
            for ev_point in evaluated_points:
                count_ev_line_points = 0
                ev_line_n = max(abs(init[0]-ev_point[0]), abs(init[1]-ev_point[1]))
                if ev_line_n != 0:
                        step_y = (ev_point[0]-init[0])/ev_line_n #3, 3/8
                        step_x = (ev_point[1]-init[1])/ev_line_n #8, 1
                        for n in range(ev_line_n):
                            new_point = [min(init[0]+int(step_y*n), img.shape[0]-1), min(init[1]+int(step_x*n), img.shape[1]-1)]
                            if img[new_point[0]][new_point[1]] == 255:
                                count_ev_line_points += 1
                if count_ev_line_points > max_ev_line_points:
                    max_ev_line_points = count_ev_line_points
                    best_second = ev_point
        else:
            best_second = [init[0]+window_size[0], init[1]]

        return best_second


    def exec(self, img: np.ndarray, init: list, window_size: list, last_x: int = 0, n_miss_max: int = 3, evaluate_init: bool = True, max_deg: int = 40) -> Tuple[list, int, list, bool, int, int]:

        if last_x == 0:
            last_x = img.shape[1]

        points_spline = []
        original_window_size = copy.deepcopy(window_size)
        count_captured = 0
        captured_points_yx = []
        count_free_steps = 0
        count_no_borders = 0
        success = False    

        points_i = []
        points_i.append([init[0], init[1]-1])
        points_i.append(init)
        #first point
        if evaluate_init:
            sec_point = self.evaluate_first_point(img, init, window_size)
            points_i.append(sec_point)

        #first_miss = True
        n_miss = 0
        window_size = copy.deepcopy(original_window_size)
        point_direction = False
        check_content = False

        prev_m = 0
        last_cycle_m = True
        use_prev_direction_counter = n_miss_max + 1 #More than 3

        while True:
            #Calculate the detected points intersecting with the window border
            evaluated_points = []
            for i in range(-window_size[0], window_size[0]+1, 1): #i == down-up rows
                new_point = [min(points_i[-1][0] + i, img.shape[0]-1), min(points_i[-1][1] + window_size[1], img.shape[1]-1)]
                if img[new_point[0]][new_point[1]] == 255:
                    evaluated_points.append(new_point)
            for i in range(int(window_size[1]/2), window_size[1]+1, 1):
                new_point = [min(points_i[-1][0] + window_size[0], img.shape[0]-1), min(points_i[-1][1] + i, img.shape[1]-1)]
                if img[new_point[0]][new_point[1]] == 255:
                    evaluated_points.append(new_point)
                new_point = [min(points_i[-1][0] - window_size[0], img.shape[0]-1), min(points_i[-1][1] + i, img.shape[1]-1)]
                if img[new_point[0]][new_point[1]] == 255:
                    evaluated_points.append(new_point)

            if len(evaluated_points) == 0 or check_content:
                #Check the area, not just the borders
                check_content = False
                save_copy_ev_points = copy.deepcopy(evaluated_points)
                evaluated_points = []
                for i in range(window_size[1]-1, 0, -1):
                    for j in range(window_size[0]-1, 0, -1):
                        new_point = [min(points_i[-1][0] + j, img.shape[0]-1), min(points_i[-1][1] + i, img.shape[1]-1)]
                        if img[new_point[0]][new_point[1]] == 255:
                            evaluated_points.append(new_point)
                        if j!=0: #Otherwise + is the same as -
                            new_point = [min(points_i[-1][0] - j, img.shape[0]-1), min(points_i[-1][1] + i, img.shape[1]-1)]
                            if img[new_point[0]][new_point[1]] == 255:
                                evaluated_points.append(new_point)
                if len(evaluated_points) == 0:
                    evaluated_points = save_copy_ev_points
                else:
                    use_prev_direction_counter = 0
                    count_no_borders += 1

            if len(evaluated_points) == 0:
                #Try with more size
                if n_miss<=0:
                    n_miss += 1
                    #window_size[0] = int(original_window_size[0] * 1.5)
                    #window_size[1] = int(original_window_size[1] * 1.5)
                    continue #This is not applied now
                elif n_miss<=n_miss_max: #Invent new point in the cable direction
                    n_miss += 1
                    point_direction = True
                    count_free_steps += 1
                else:
                    success = False
                    break
            else:
                n_miss = 0
            
            #Calculate the intersection between the cable direction and the window border
            if use_prev_direction_counter < n_miss_max:
                use_prev_direction_counter += 1
                try:
                    if points_i[-1][1] - points_i[-4][1] != 0:
                        m = (points_i[-1][0] - points_i[-4][0])/(points_i[-1][1] - points_i[-4][1])
                    elif points_i[-1][0] - points_i[-4][0] > 0: #if the points have the same x pixel, the slope should be infinite
                        m = 10
                    else:
                        m = -10
                except:
                    m=0
                    #n = points_i[-1][0] - (points_i[-1][1] * m)
            else:
                last_cycle_m = True

            if last_cycle_m: #If we can still calculate m with the last cycle info
                if points_i[-1][1] - points_i[-2][1] != 0:
                    m = (points_i[-1][0] - points_i[-2][0])/(points_i[-1][1] - points_i[-2][1])
                elif points_i[-1][0] - points_i[-2][0] > 0: #if the points have the same x pixel, the slope should be infinite
                    m = 10
                else:
                    m = -10
                #n = points_i[-1][0] - (points_i[-1][1] * m)
                if use_prev_direction_counter == 1:
                    last_cycle_m = False #We cannot calculate m in the next cycle, then we save it
                    prev_m = m
                else:
                    last_cycle_m = True

            dir_point = [0,0]
            slope_limit = window_size[0]/window_size[1] #To know if we should consider the intersection with the vertical or horizontal border of the window
            if m<=slope_limit and m>=-slope_limit:
                dir_point[0] = points_i[-1][0] + window_size[1]*m
                dir_point[1] = points_i[-1][1] + window_size[1]
            elif m>slope_limit:
                dir_point[1] = points_i[-1][1] + (window_size[0])/m
                dir_point[0] = points_i[-1][0] + window_size[0]
            else:
                dir_point[1] = points_i[-1][1] + (window_size[0])/m
                dir_point[0] = points_i[-1][0] - window_size[0]

            #Calculate the closest intersecting point to the cable direction
            if point_direction: #Pick an invented point following the trajectory
                min_dist_point = [0,0]
                min_dist_point[0] = int(dir_point[0])
                min_dist_point[1] = int(dir_point[1])
                point_direction = False
            else:
                min_dist = 999
                min_dist_point = [0,0]
                for ev_point in evaluated_points:
                    dist = points_dist2D(ev_point, dir_point)
                    if dist <= min_dist:
                        min_dist = dist
                        min_dist_point = ev_point

                if min_dist_point in points_i:
                    success = False
                    break

            #Check if there is a too big change of slope, more than a certain angle, this point is considered incorrect
            prev_m = m
            if min_dist_point[1] - points_i[-1][1] != 0:
                m = (min_dist_point[0] - points_i[-1][0])/(min_dist_point[1] - points_i[-1][1])
            elif min_dist_point[0] - points_i[-1][0] > 0: #if the points have the same x pixel, the slope should be infinite
                m = 10
            else:
                m = -10
            prev_deg = math.atan(prev_m)*180/math.pi
            new_deg = math.atan(m)*180/math.pi
            if abs(prev_deg-new_deg)> max_deg: #Angles difference is too big
                min_dist_point = [0,0]
                min_dist_point[0] = int(dir_point[0])
                min_dist_point[1] = int(dir_point[1])
                point_direction = False

            points_i.append(min_dist_point)

            #Count the number of captured points in the window, this is used to later evaluate the results
            for rel_row in range(-original_window_size[0], original_window_size[0]+1, 1):
                for col in range(points_i[-2][1]+1, min(points_i[-1][1]+1, img.shape[1]-1), 1):
                    if img[min(points_i[-2][0]+rel_row, img.shape[0]-1)][col] == 255:
                        count_captured += 1
                        captured_points_yx.append([points_i[-2][0]+rel_row, col])

            #Restart for next loop
            window_size = copy.deepcopy(original_window_size)
            if min_dist_point[0] >= (img.shape[0]-1) or min_dist_point[1] >= (img.shape[1]-1):
                success = True
                break

        points_spline = copy.deepcopy(points_i)

        if points_spline[-1][1] > 0.7*last_x: #Otherwise if the cable is not visible at the end due to a big occlusion it will fail. Something like 70% of width should be enough
            success = True

        return points_spline, count_captured, captured_points_yx, success, count_no_borders, count_free_steps