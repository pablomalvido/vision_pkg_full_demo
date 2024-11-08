#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import cv2 as cv
from TAU_segmentation import TAUSegmentation
from TAU_segmentation_no_borders import TAUSegmentationNoBorders
from TAU_segmentation_UNet import TAUSegmentationUNet
from TAU_preprocessing import TAUPreprocessing
from TAU_forward_propagation import TAUForwardPropagation
from TAU_backward_propagation import TAUBackwardPropagation
from TAU_cable_line_estimation2 import TAUCableLineEstimation
from TAU_critique import TAUCritique
from TAU_grasp_evaluation import TAUGraspEvaluation
from TAU_grasp_point_calculation import get_best_grasping_point
from plot_cables import plot_cables
import copy
import math
import numpy as np
import time
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import *
from vision_pkg_full_demo.srv import *

rospy.init_node('DLO_detection_node')

class DLO_estimator():

    def __init__(self, img_path, all_colors, color_order, con_points, cable_D, con_dim, cable_lengths, pixel_D, model, grasping_point_eval_mm=[0,0], grasp_area_mm=[0,0], corner_mold=[[0,0],[0,0]], mold_size = 0, segm_opt = 0, index_upper=0, analyzed_length=500, simplified=False):

        global show_imgs

        self.red_color_paint = [0,0,255]
        self.background_color = [221,160,221]
        self.color_order = color_order
        self.pixel_D = pixel_D
        self.cable_lengths = cable_lengths
        self.img_name = img_path.split('/')[-1]
        self.index_upper = index_upper
        self.simplified = simplified
        print("A")
        img = cv.imread(img_path)
        print("B")

        if mold_size>0:
            self.theta = math.atan2(corner_mold[0][1]-corner_mold[1][1], corner_mold[1][0]-corner_mold[0][0])
        else:
            self.theta = 0

        self.preprocessing = TAUPreprocessing(input_img = img, cable_D = cable_D, n_cables = len(color_order), con_points = con_points, con_dim=con_dim, cable_length=max(cable_lengths), pixel_D=pixel_D, grasping_point_eval_mm=grasping_point_eval_mm, grasp_area_mm=grasp_area_mm, mold_corner=corner_mold, mold_size=mold_size, analyzed_length=analyzed_length)
        resized_img, img_init_points, con_points_resized, self.init_points, self.mm_per_pixel, self.window_size, grasping_point_eval_resized, grasp_area_resized, self.corner_mold_top_resized = self.preprocessing.exec()
        if show_imgs:
            cv.imshow("Initial points", img_init_points)
        init_col = min(con_points_resized[0][1], con_points_resized[1][1])
        
        #Segmentation options
        if segm_opt == 0:
            self.segmentation = TAUSegmentation(input_img=resized_img, all_colors=all_colors, init_col = init_col)
        elif segm_opt == 1:
            self.segmentation = TAUSegmentationNoBorders(input_img=resized_img, all_colors=all_colors, init_col = init_col, pixel_D = self.pixel_D)
        elif segm_opt == 2:
            self.segmentation = TAUSegmentationUNet(input_img=resized_img, all_colors=all_colors, model=model, IMAGE_SIZE=512, init_col = init_col)

        self.FWP = TAUForwardPropagation(self.mm_per_pixel)
        self.BWP = TAUBackwardPropagation(self.mm_per_pixel)
        self.line_estimation = TAUCableLineEstimation(max_order=8, min_order=0)
        self.grasp_evaluation = TAUGraspEvaluation(grasping_point_eval_resized, grasp_area_resized)

        #Creates a background image to draw the estimated cables
        bkg = np.zeros((resized_img.shape[0], resized_img.shape[1], 3), dtype='uint8')
        for row in range(bkg.shape[0]):
            for col in range(bkg.shape[1]):
                bkg[row][col] = self.background_color
        self.points_cables_img = copy.deepcopy(resized_img)
        self.lines_cables_img = copy.deepcopy(resized_img)
        self.lines_cables_bkg = copy.deepcopy(bkg)
        self.grasp_img = copy.deepcopy(resized_img)


    def exec(self, forward, iteration, determine_GP=False, evaluate_grasp=False, thr1_init=60, thr2_init=1.5, erosion_init=3):

        global show_imgs

        index = 0
        all_points_cables = []
        all_points_cables_dict = {}
        grasped_cables = []
        wrong_estimated_cables = []
        plot_cables_data = []
        cable_lines_coeffs = []
        for color in self.color_order:
            if self.simplified and not (index >= self.index_upper-2):
                index+=1
                continue

            thr1 = thr1_init
            thr2 = thr2_init
            erosion = erosion_init
            WinSize = self.window_size
            success_cable = False

            i=0
            init_points_BW = [self.init_points[index]]
            for c in self.color_order:
                if i != index and c == color:
                    init_points_BW.append(self.init_points[i])
                i+=1
            n_cables = len(init_points_BW)
            cable_length = self.cable_lengths[index]

            if iteration:
                self.critique = TAUCritique(self.pixel_D, thr1, thr2, erosion, WinSize, forward)                

            ite=1
            while not success_cable:
                print("Cable " + str(index) + ": calculating (iteration " + str(ite) + ")...")
                ite+=1

                #Segmentation
                segm_img, segm_pixels  = self.segmentation.exec(color_cable = color, thr1=thr1, thr2=thr2, erosion=3)
                if show_imgs:
                    cv.imshow("Segmentation", segm_img)

                #Cable points calculation
                #FWP
                if forward:
                    points_cable, n_captured_points, success_points, count_no_borders, count_free_steps, init_success = self.FWP.exec(segm_img, segm_pixels, self.init_points[index], WinSize, cable_length)
                
                #BWP
                else:
                    points_cable, n_captured_points, success_points = self.BWP.exec(segm_img, segm_pixels, init_points_BW, n_cables, WinSize, cable_length)
                    init_success = True; count_no_borders = 0; count_free_steps = 0

                #Critique
                if iteration:
                    success_cable, result_error, thr1, thr2, erosion, WinSize = self.critique.exec(points_cable, segm_img, len(segm_pixels), n_captured_points, success_points, n_cables, thr1, thr2, WinSize, erosion, init_success, count_no_borders, count_free_steps)
                else:
                    success_cable = True

            #Cable polynomial function estimation
            cable_line_yx, cable_line_xy, poly_coeffs = self.line_estimation.exec(points_cable)
            all_points_cables.append(cable_line_yx)
            all_points_cables_dict[index] = cable_line_yx
            cable_lines_coeffs.append(poly_coeffs)

            #Non successful estimation
            if not success_cable or (not iteration and (not success_points or not init_success)):
                wrong_estimated_cables.append(index)

            #Paint results
            for point in points_cable:
                self.points_cables_img = cv.rectangle(self.points_cables_img, (point[1]-self.window_size[1],point[0]-self.window_size[0]), (point[1]+self.window_size[1],point[0]+self.window_size[0]), [0,255,0], 2)
            plotted_line = {'x':[], 'y': [], 'color': color}
            for point1, point2 in zip(cable_line_xy, cable_line_xy[1:]): 
                self.lines_cables_img = cv.line(self.lines_cables_img, point1, point2, self.red_color_paint, 2)
                self.lines_cables_bkg = cv.line(self.lines_cables_bkg, point1, point2, color, 2)
            for point in cable_line_xy:
                plotted_line['x'].append(point[0])
                plotted_line['y'].append(point[1])
            plot_cables_data.append(plotted_line)

            print("Cable " + str(index) + " CALCULATED")
            print("--------------------------------------------------")
            index += 1

        plot_cables(plot_cables_data, self.img_name, self.lines_cables_img.shape)
        if show_imgs:
            cv.imshow("Points cables", self.points_cables_img)     
            cv.imshow("Lines cables", self.lines_cables_img)
            cv.imshow("Lines cables bkg", self.lines_cables_bkg)

        save_img_name = "Estimation1_" + self.img_name
        save_img_path = os.path.join(os.path.dirname(__file__), '../imgs/'+save_img_name)
        cv.imwrite(save_img_path, self.lines_cables_img)
        save_img_name = "Estimation2_" + self.img_name
        save_img_path = os.path.join(os.path.dirname(__file__), '../imgs/'+save_img_name)
        cv.imwrite(save_img_path, self.lines_cables_bkg)

        #Find optimal grasp points to separate cable groups
        if determine_GP:
            grasp_point, success_GP = get_best_grasping_point(all_points_cables_dict, self.index_upper, self.mm_per_pixel, self.lines_cables_img)
            if success_GP:
                self.grasp_img = cv.rectangle(self.grasp_img, (grasp_point[0] - 3, grasp_point[1] - 3), (grasp_point[0] + 3, grasp_point[1] + 3), [0,255,0], 2)
                save_img_name = "Grasp_point_" + self.img_name
                save_img_path = os.path.join(os.path.dirname(__file__), '../imgs/'+save_img_name)
                cv.imwrite(save_img_path, self.grasp_img)
                cv.imwrite('/home/remodel/UI-REMODEL/src/assets/img/grasp_image.jpg', self.grasp_img)
                grasp_point_from_corner = []
                grasp_x = (grasp_point[0] - self.corner_mold_top_resized[1])*self.mm_per_pixel
                grasp_z = (self.corner_mold_top_resized[0] - grasp_point[1])*self.mm_per_pixel
                grasp_point_from_corner_x_aligned = math.cos(self.theta)*grasp_x - math.sin(self.theta)*grasp_z
                grasp_point_from_corner_z_aligned = math.sin(self.theta)*grasp_x + math.cos(self.theta)*grasp_z
                print(grasp_point_from_corner)
                return all_points_cables_dict, grasp_point_from_corner_x_aligned, grasp_point_from_corner_z_aligned, True
            else:
                print("Grasp point determination error")
                return all_points_cables_dict, 0,0, False

        #Evaluate the result of the robotic separation of specific wire harness cable groups
        elif evaluate_grasp:
            for index in all_points_cables_dict:
                if self.grasp_evaluation.exec(all_points_cables_dict[index]):
                    grasped_cables.append(index)
            return all_points_cables_dict, grasped_cables, wrong_estimated_cables

        else:
            return all_points_cables_dict, wrong_estimated_cables, cable_lines_coeffs


def estimate_cables_shape_srv_callback(req):
    global WH_info
    global DLO_model
    global show_imgs

    print(req.img_path)
    show_imgs = req.visualize
    resp = cablesShapeResponse()

    total_time_init = time.time()
    p = DLO_estimator(img_path=req.img_path, all_colors=WH_info[req.wh_id]['cable_colors'], color_order = WH_info[req.wh_id]['cables_color_order'], con_points=[req.connector_lower_corner, req.connector_upper_corner], cable_D=WH_info[req.wh_id]['cable_D'], con_dim=WH_info[req.wh_id]['con_dim'], cable_lengths=WH_info[req.wh_id]['cable_lengths'], model=DLO_model, pixel_D=req.pixel_D, analyzed_length=req.analyzed_length, segm_opt=req.segm_opt)
    if not show_imgs:
        all_points_cables, wrong_estimated_cables, cable_lines_coeffs = p.exec(req.forward, req.iteration, determine_GP = False)
        total_time = time.time() - total_time_init
        print("Computation time: " + str(total_time) + "s")
        print("--------------------------------------------------")
        if len(all_points_cables) > 0:
            resp.success = True
        else:
            resp.success = False
        i=0
        result_msg = ""
        for c in cable_lines_coeffs:
            result_msg += "Cable " + str(i) + " polynomial coefficients: " + str(c) + "\n"
            i+=1
        if len(wrong_estimated_cables)>0:
            result_msg += "################\n"
            result_msg += "Wrong estimation for cables: " + str(wrong_estimated_cables)
        resp.result = result_msg
    else:
        cv.waitKey(0)
    return resp

rospy.Service('/vision/estimate_cables_shape_srv', cablesShape, estimate_cables_shape_srv_callback)


def grasp_point_determination_srv_callback(req):
    global WH_info
    global DLO_model
    global show_imgs

    show_imgs = req.visualize
    resp = cablesSeparationResponse()
    resp.success = False
    
    desired_grasp_cables_tuple = req.separated_cable_index
    desired_grasp_cables = list(desired_grasp_cables_tuple)

    total_time_init = time.time()
    p = DLO_estimator(img_path=req.img_path, all_colors=WH_info[req.wh_id]['cable_colors'], color_order = WH_info[req.wh_id]['cables_color_order'], con_points=[req.connector_lower_corner, req.connector_upper_corner], cable_D=WH_info[req.wh_id]['cable_D'], con_dim=WH_info[req.wh_id]['con_dim'], cable_lengths=WH_info[req.wh_id]['cable_lengths'], model=DLO_model, corner_mold=[req.mold_upper_corner, req.mold_lower_corner], mold_size=WH_info[req.wh_id]['mold_dim'], index_upper=desired_grasp_cables[0], pixel_D=req.pixel_D, analyzed_length=req.analyzed_length, segm_opt=2, simplified=req.simplified)
    if not show_imgs:
        all_points_cables, grasp_point_from_corner_x, grasp_point_from_corner_z, success = p.exec(req.forward, req.iteration, determine_GP = True)
        total_time = time.time() - total_time_init
        print("Computation time: " + str(total_time) + "s")
        print("--------------------------------------------------")

        resp.grasp_point = [grasp_point_from_corner_x, grasp_point_from_corner_z]
    else:
        cv.waitKey(0)
    resp.success = success
    return resp

rospy.Service('/vision/grasp_point_determination_srv', cablesSeparation, grasp_point_determination_srv_callback)


def check_cable_separation_srv_callback(req):
    global WH_info
    global DLO_model
    global show_imgs

    show_imgs = req.visualize
    resp = cablesSeparationResponse()
    resp.success = False
    
    desired_grasp_cables_tuple = req.separated_cable_index
    desired_grasp_cables = list(desired_grasp_cables_tuple)
    required_estimations = copy.deepcopy(desired_grasp_cables)
    lower_cable = min(desired_grasp_cables) - 1
    upper_cable = max(desired_grasp_cables) + 1
    if lower_cable >= 0:
        required_estimations.append(lower_cable)
    if upper_cable < len(WH_info[req.wh_id]['cables_color_order']):
        required_estimations.append(upper_cable)

    total_time_init = time.time()
    p = DLO_estimator(img_path=req.img_path, all_colors=WH_info[req.wh_id]['cable_colors'], color_order = WH_info[req.wh_id]['cables_color_order'], con_points=[req.connector_lower_corner, req.connector_upper_corner], cable_D=WH_info[req.wh_id]['cable_D'], con_dim=WH_info[req.wh_id]['con_dim'], cable_lengths=WH_info[req.wh_id]['cable_lengths'], model=DLO_model, grasping_point_eval_mm=req.grasp_point_eval_mm, grasp_area_mm=[50,100], corner_mold=[req.mold_upper_corner, req.mold_lower_corner], mold_size=WH_info[req.wh_id]['mold_dim'], index_upper=desired_grasp_cables[0], pixel_D=req.pixel_D, analyzed_length=req.analyzed_length, segm_opt=2, simplified=req.simplified)
    if not show_imgs:
        all_points_cables, grasped_cables, wrong_estimated_cables = p.exec(req.forward, req.iteration, evaluate_grasp = True)
        total_time = time.time() - total_time_init
        print("Computation time: " + str(total_time) + "s")
        print("--------------------------------------------------")

        if grasped_cables == desired_grasp_cables and not (wrong_estimated_cables in required_estimations):
            resp.result = "SUCCESSFUL CABLE SEPARATION - Grasped cables: " + str(grasped_cables)
            resp.separation_success = True
        else:
            resp.result = "REVISE CABLE SEPARATION, IT MIGHT BE WRONG - Grasped cables: " + str(grasped_cables)
            resp.separation_success = False
    else:
        cv.waitKey(0)
    resp.success = True
    return resp

rospy.Service('/vision/check_cable_separation_srv', cablesSeparation, check_cable_separation_srv_callback)


if __name__ == "__main__": #Load the model and define some info for each WH in a dictionary

    #Global variables
    show_imgs = False
    WH_info = {}

    #All cable colors in BGR scale
    yellow_cable = [40,141,171]
    blue_cable = [157,99,48]
    green_cable = [43,108,50]
    red_cable = [53,49,181]
    black_cable = [57,54,71]
    white_cable = [181,184,191]
    """ 
    #Adust the colors to your image
    yellow_cable = [40,141,171]
    blue_cable = [157,99,48]
    green_cable = [43,108,50]
    green_cable2 = [36,110,38]
    red_cable = [53,49,181]
    black_cable = [10,10,10]#[57,54,71]
    white_cable = [181,184,191]
    brown_cable = [27,43,80]
    pink_cable = [134,133,179]
    """

    #Define WHs info
    WH_info['1'] = {}
    WH_info['1']['cable_colors'] = [yellow_cable, blue_cable, green_cable, red_cable, black_cable, white_cable]
    #order: from bottom to top
    WH_info['1']['cables_color_order'] = [yellow_cable, green_cable, blue_cable, white_cable, yellow_cable, green_cable, blue_cable, white_cable, red_cable, black_cable]
    #All measures in mm
    WH_info['1']['cable_lengths'] = [550, 550, 550, 550, 550, 550, 550, 550, 550, 550]
    WH_info['1']['con_dim'] = 27
    WH_info['1']['cable_D'] = 1.32
    WH_info['1']['mold_dim'] = 40
    
    """
    WH_info['2'] = {}
    WH_info['2']['cable_colors'] = [yellow_cable, blue_cable, green_cable2, red_cable, black_cable, pink_cable, brown_cable, white_cable]
    #order: from bottom to top
    WH_info['2']['cables_color_order'] = [red_cable, black_cable, white_cable, green_cable2, blue_cable, yellow_cable, pink_cable, brown_cable, yellow_cable, black_cable, red_cable]
    #All measures in mm
    WH_info['2']['cable_lengths'] = [80, 80, 80, 95, 95, 95, 95, 430, 430, 430, 430]
    WH_info['2']['con_dim'] = 27
    WH_info['2']['cable_D'] = 1.32
    WH_info['2']['mold_dim'] = 40
    """

    #Load model
    path_model = os.path.dirname(os.path.realpath(__file__)) + "/../models/my_model_v1"
    DLO_model = tf.keras.models.load_model(path_model, compile=False)
    DLO_model.compile(optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=[tf.keras.metrics.FalseNegatives()])

print ("Vision node ready...")
rospy.spin()
#cv.waitKey(0)