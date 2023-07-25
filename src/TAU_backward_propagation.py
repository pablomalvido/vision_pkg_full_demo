import numpy as np
import cv2 as cv
import copy
import time
from typing import Tuple
from TAU_segmentation import TAUSegmentation
from TAU_img_functions import *
from TAU_cable_points_estimation import TAUCablePointsEstimation
from TAU_cable_line_estimation2 import TAUCableLineEstimation

from interfaces import TAUBackwardPropagationInterface


class TAUBackwardPropagation(TAUBackwardPropagationInterface):
    '''
    Implements the backward propagation. A cable is calculated propagating all the segmented pixels backwards, then we discard small and repeated segments, 
    we join the segments that are part of the same line and we select the final line that is closer to the theoretical initial point
    '''

    def __init__(self, mm_per_pixel):
        self.points_estimation = TAUCablePointsEstimation()
        self.line_estimation = TAUCableLineEstimation(max_order=8, min_order=0)
        self.mm_per_pixel = mm_per_pixel

    def exec(self, segm_img: np.ndarray, segm_pixels: list, initial_point: list, n_cables: int, window_size: list, cable_length: float, init_index: int = 0) -> Tuple[list, int, bool]:
        #Check all the white points from left to right and for each of them check with kernels the possible points
        #Go backwards (we can just mirror and use known functions)
        img_evaluate = copy.deepcopy(segm_img)
        img_evaluate = cv.flip(img_evaluate, 1) #Mirror image so we can apply forward propagation
        tested_lines = []
        captured_points_list = []
        starting_col = int(min(segm_img.shape[1], initial_point[init_index][1] + (cable_length/self.mm_per_pixel)))
        starting_col_mirror = segm_img.shape[1] - starting_col
        for row in range(0,img_evaluate.shape[0],2):
            for col in range(starting_col_mirror, img_evaluate.shape[1],2): #From 0?
                if img_evaluate[row][col] == 255 and not([row,col] in captured_points_list):
                    points_cable, captured_points, captured_points_yx, success_points, count_no_borders, count_free_steps = self.points_estimation.exec(img_evaluate, init=[row,col], last_x = img_evaluate.shape[1] - initial_point[init_index][1] - (10/self.mm_per_pixel), window_size=window_size, n_miss_max = 2, evaluate_init = False)
                    captured_points_list += captured_points_yx
                    tested_lines.append({"captured_perc": float(len(captured_points_yx))/float(len(segm_pixels)), "captured_yx": captured_points_yx, "success": success_points, "points": points_cable})

        #If a point dies with less than a certain % don't check it or the points inside the checked kernels again
        tested_lines = list(filter(lambda x: x['captured_perc'] >= (0.025/n_cables), tested_lines)) #Discard segments with very little points
        sorted_lines = sorted(tested_lines, key=lambda x:x["captured_perc"], reverse=True)

        #Discard repeated lines (More than 60% (change to 80%??) of repeated points)
        detected_lines = []
        detected_lines.append(sorted_lines[0])
        for line in sorted_lines[1:]:
            for line_det in detected_lines:
                count_common_points = 0
                common_line = False
                for point_line in line["captured_yx"]:
                    if point_line in line_det["captured_yx"]:
                        count_common_points+=1
                if float(count_common_points)/float(len(line["captured_yx"])) > 0.6: #0.8?
                    common_line = True
                    break
            if not common_line:
                detected_lines.append(line)

        #Join cable segments
        join_dict = {}
        join_dict_final = {}
        i=0
        for line in detected_lines:
            j=0
            for line2 in detected_lines:
                if line["points"][-2][1] < line2["points"][0][1] and (points_dist2D(line["points"][-1], line2["points"][0]) <= (50/self.mm_per_pixel)):
                    if line["points"][-3][1] != line["points"][-1][1] and line2["points"][2][1] != line2["points"][0][1]:
                        if line["points"][-3][0] >= line["points"][-1][0]:
                            angle_line = math.atan((line["points"][-3][0] - line["points"][-1][0])/(line["points"][-1][1] - line["points"][-3][1]))*180/math.pi
                        else:
                            angle_line = 180 - math.atan((line["points"][-1][0] - line["points"][-3][0])/(line["points"][-1][1] - line["points"][-3][1]))*180/math.pi
                        
                        if line["points"][-2][0] >= line2["points"][0][0]:
                            angle_join_line = math.atan((line["points"][-2][0] - line2["points"][0][0])/(line2["points"][0][1] - line["points"][-2][1]))*180/math.pi
                        else:
                            angle_join_line = 180 - math.atan((line2["points"][0][0] - line["points"][-2][0])/(line2["points"][0][1] - line["points"][-2][1]))*180/math.pi
                        
                        if line2["points"][0][0] >= line2["points"][2][0]:
                            angle_line2 = math.atan((line2["points"][0][0] - line2["points"][2][0])/(line2["points"][2][1] - line2["points"][0][1]))*180/math.pi
                        else:
                            angle_line2 = 180 - math.atan((line2["points"][2][0] - line2["points"][0][0])/(line2["points"][2][1] - line2["points"][0][1]))*180/math.pi
                        
                        if abs(angle_line - angle_join_line) <= 35 and abs(angle_line2 - angle_join_line) <= 35 and abs(angle_line - angle_line2) <= 35: #Check if the two segments belong to the same line
                            if not i in join_dict:
                                join_dict[i] = []
                            join_dict[i].append({"line2": j, "dist": points_dist2D(line["points"][-1], line2["points"][0]), "angle_dif": abs(angle_line - angle_join_line)}) 
                j+=1 #Every time a line can be joined to the current line
            if i in join_dict:
                if len(join_dict[i])>1:
                    join_dict_final[i] = min(join_dict[i], key = lambda x: x['dist'])["line2"] #The joined line is the closer one from all the possible (all the js)
                else:
                    join_dict_final[i] = join_dict[i][0]["line2"]
            i+=1

        detected_join_lines_dict = []
        end_indexes= []
        start_indexes = []
        join_lines_indexes_seq = []
            
        #We merge the lines we detected that had to be joined
        for line_i in range(len(detected_lines)):
            if line_i in join_dict_final:
                start_index = line_i
                end_index = join_dict_final[line_i]
                    
                if start_index in end_indexes:
                    i = 0
                    for item in join_lines_indexes_seq:
                        if item[-1] == start_index:
                            join_lines_indexes_seq[i].append(end_index) 
                        i+=1
                                
                elif end_index in start_indexes:
                    i = 0
                    for item in join_lines_indexes_seq:
                        if item[0] == end_index:
                            join_lines_indexes_seq[i].insert(0,start_index) 
                        i+=1
                        
                else:
                    join_lines_indexes_seq.append([start_index, end_index])
                        
                start_indexes.append(start_index)
                end_indexes.append(end_index)

        detected_join_lines_dict = []
        for line in join_lines_indexes_seq:
            new_line = []
            new_line_captured = 0
            for segment in line:
                new_line += detected_lines[segment]['points']
                new_line_captured += len(detected_lines[segment]['captured_yx'])
            n_join_lines = len(line)
            detected_join_lines_dict.append({"line": new_line, "captured_n": new_line_captured, "n_join_lines": n_join_lines})

        for segment in range(len(detected_lines)): #Adds the lines that were not joined to other lines
            independent_line = True
            for segments_joined in join_lines_indexes_seq:
                if segment in segments_joined:
                    independent_line = False
            if independent_line:
                detected_join_lines_dict.append({"line": detected_lines[segment]['points'], "captured_n": len(detected_lines[segment]['captured_yx']),  "n_join_lines": 0})
        
        detected_join_lines_dict = sorted(detected_join_lines_dict, key=lambda x:x['captured_n'], reverse=True)

        #We get just the n lines with more points (n is the number of cables of that color)
        detected_join_lines_trim = copy.deepcopy(detected_join_lines_dict)
        detected_join_lines_trim = detected_join_lines_trim[:min(n_cables,len(detected_join_lines_dict))]

        #When all the points have been covered by any kernel, we should have all the lines of that color, compare then with the starting points to know which is each one
        corrected_lines = []
        img_evaluate2 = copy.deepcopy(img_evaluate)
        for line in detected_join_lines_trim:
            corrected_line = []
            #for point in line["points"]:
            for point in line['line']:
                corrected_line_point = [point[0], segm_img.shape[1]-point[1]] #Unmirror point
                #points_cable_img = cv.rectangle(points_cable_img, (corrected_line_point[0]-5,corrected_line_point[1]-5), (corrected_line_point[0]+5,corrected_line_point[1]+5), [255,0,0], 2)
                img_evaluate2 = cv.rectangle(img_evaluate2, (point[1]-window_size[1],point[0]-window_size[0]), (point[1]+window_size[1],point[0]+window_size[0]), 150, 2)#Points are x-y, not row-col
                corrected_line.append(corrected_line_point)
            corrected_lines.append(corrected_line)
        for i in range(len(corrected_lines)):
            corrected_lines[i].reverse()
            corrected_lines[i] = list(filter(lambda x: x[1] >= initial_point[0][1], corrected_lines[i]))
        #print(corrected_lines)

        all_lines = []
        for line_points in corrected_lines:
            all_points_yx, all_points_xy = self.line_estimation.exec(line_points, step_line=1, limits = [initial_point[init_index][1], max(line_points,key=lambda x:x[1])[1]]) #initial_point is yx
            all_lines.append(all_points_yx)

        #Identify each cable if there are more than 1
        min_dist_init = 1000
        index_selected = 0
        index = 0
        selected_line = all_lines[0]
        n_captured_points = detected_join_lines_trim[0]['captured_n']
        selected_n_join_lines = detected_join_lines_trim[0]['n_join_lines']
        selected_points = corrected_lines[0]
        init_point_selected = initial_point[init_index]
        init_index_sorted = 0

        #Get order of indexes
        li=[] 
        for i in range(len(all_lines)):
            li.append([all_lines[i][0][0],i])
        li.sort()
        sort_index = []
        
        for x in li:
            sort_index.append(x[1])

        all_lines_sorted = []
        for i in range(len(all_lines)):
            all_lines_sorted.append(all_lines[sort_index[i]]) #Sort the lines by its column in the first cable point, to match it later with the theoretical initial point

        if n_cables > 1: #Test with > 0
            init_index = 0 #Comment this line
            init_ref = initial_point[init_index]
            initial_point_sorted = sorted(initial_point, key=lambda x:x[0]) #Sort by y (rows)
            init_index_sorted = initial_point_sorted.index(init_ref)
            selected_line = all_lines_sorted[init_index_sorted]
            n_captured_points = detected_join_lines_trim[sort_index[init_index_sorted]]['captured_n']
            selected_n_join_lines = detected_join_lines_trim[sort_index[init_index_sorted]]['n_join_lines']
            selected_points = corrected_lines[sort_index[init_index_sorted]]
        
        success_points = False
        if points_dist2D(selected_line[0], initial_point[0]) < 15/self.mm_per_pixel: #Real detected initial point, not an estimated regression line
            success_points = True
        else:
            line_index = 0
            min_dist_init = 10000
            for line_points in all_lines:
                dist_init = points_dist2D(line_points[0], initial_point[0])
                if dist_init < min_dist_init:
                    min_dist_init = dist_init
                    selected_line = copy.deepcopy(line_points)
                    selected_points = copy.deepcopy(corrected_lines[line_index])
                    index_selected = line_index #To know which points to draw with rectangles
                line_index += 1
            if points_dist2D(selected_line[index_selected], initial_point[0]) < 15/self.mm_per_pixel: #Real detected initial point, not an estimated regression line
                success_points = True

        """
        ##TEST
        img_lines_test2 = cv.flip(copy.deepcopy(img_evaluate), 1)
        for point in selected_points:
                img_lines_test2 = cv.rectangle(img_lines_test2, (point[1]-window_size[1],point[0]-window_size[0]), (point[1]+window_size[1],point[0]+window_size[0]), 150, 2)#Points are x-y, not row-col
        cv.imshow("BB", img_lines_test2)
        cv.imshow("CC", img_evaluate)

        img_lines_test = copy.deepcopy(img_evaluate)
        for line in detected_join_lines_dict:
            #for point in line["points"]:
            for point in line['line']:
                #points_cable_img = cv.rectangle(points_cable_img, (corrected_line_point[0]-5,corrected_line_point[1]-5), (corrected_line_point[0]+5,corrected_line_point[1]+5), [255,0,0], 2)
                img_lines_test = cv.rectangle(img_lines_test, (point[1]-window_size[1],point[0]-window_size[0]), (point[1]+window_size[1],point[0]+window_size[0]), 150, 2)#Points are x-y, not row-col
        cv.imshow("AA", img_lines_test)
        """

        new_selected_points = []
        new_selected_points.append(init_point_selected)
        for point in selected_points:
            new_selected_points.append([point[0], point[1]])

        return new_selected_points, n_captured_points, success_points        