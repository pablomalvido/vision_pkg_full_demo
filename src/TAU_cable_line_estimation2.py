import numpy as np
import cv2 as cv
import copy
import time
from typing import Tuple
from TAU_img_functions import *
import random


from interfaces import TAUCableLineEstimationInterface


class TAUCableLineEstimation(TAUCableLineEstimationInterface):
    '''
    Implements the estimation of the cable line with a polynomial regression
    '''

    def __init__(self, max_order, min_order):

        self.max_order = max_order
        self.min_order = min_order


    def exec(self, points: list, step_line: int = 1, limits: list = [0,0]) -> Tuple[list, list]:

        #Training(70) and testing(30) dataset (Cross validation)
        indexes = list(range(len(points)))
        random.shuffle(indexes)
        train_len = int(len(points) * 0.5)
        test_len = len(points) - train_len
        error_order = [0] * (self.max_order+1)
        points_x = []
        points_y = []

        for i in range(len(points)):
            train_set_x = []
            train_set_y = []
            test_set_x = []
            test_set_y = []
            points_x.append(points[i][1]) #columns
            points_y.append(points[i][0]) #rows

            if i<test_len:
                train_set_indexes = indexes[i:i+train_len]
                test_set_indexes = indexes[0:i] + indexes[i+train_len:len(points)]
            else:
                test_set_indexes = indexes[i-test_len:i]
                train_set_indexes = indexes[0:i-test_len] + indexes[i:len(points)]

            for i in train_set_indexes:
                train_set_x.append(points[i][1])
                train_set_y.append(points[i][0])
            for i in test_set_indexes:
                test_set_x.append(points[i][1])
                test_set_y.append(points[i][0])

            #Estimate splines of different orders using polyfit
            for order in range(self.min_order,self.max_order+1,1): #Till order max_order
                try:
                    p_fit = np.polyfit(train_set_x, train_set_y, order)
                except:
                    continue
            
                #Check how well each order fits the testing dataset
                for test_i in range(len(test_set_x)):
                    estimated = 0
                    for n in range(order+1):
                        estimated += p_fit[n]*(test_set_x[test_i]**(order-n)) #Estimates the value of the testing point
                    error_order[order] += (estimated-test_set_y[test_i])**2 #Add the error to the error of the order

        #Estimate the spline with all the values
        best_order = error_order.index(min(error_order[self.min_order:]))
        #print("Best order spline: " + str(best_order) + " with an error: " + str(error_order[best_order]))
        p_fit_best = np.polyfit(points_x, points_y, best_order)
        if limits==[0,0]:
            xmin_line = min(points_x)
            xmax_line = max(points_x)
        else:
            xmin_line = limits[0]
            xmax_line = limits[1]
        points_line = []
        points_line_yx = []
        points_line_xy = []
        for x_line in range(xmin_line, xmax_line+1, step_line):
            y_line = 0
            for n in range(best_order+1):
                y_line += p_fit_best[n]*(x_line**(best_order-n))
            points_line_yx.append([int(y_line), x_line])
            points_line_xy.append([x_line, int(y_line)])

        return points_line_yx, points_line_xy