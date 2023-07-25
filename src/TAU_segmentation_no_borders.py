import numpy as np
import cv2 as cv
import copy
import time
from typing import Tuple

#from interfaces import TAUSegmentationInterfaceNoBorders


#class TAUSegmentationNoBorders(TAUSegmentationInterfaceNoBorders):
class TAUSegmentationNoBorders():
    '''
    Implements the segmentation via Dilate(edges) AND color_filter 
    '''

    def __init__(self, input_img: np.ndarray, all_colors: list, init_col: int = 0, pixel_D = 5):

        self.img = input_img
        self.all_colors = all_colors
        self.init_col = init_col
        self.pixel_D = pixel_D
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

        #Addition from here (no borders)
        segm_pixels2 = []
        segm_img2 = np.zeros((self.img.shape[0], self.img.shape[1]), dtype='uint8')
        segm_img_wrong = np.zeros((self.img.shape[0], self.img.shape[1]), dtype='uint8')
        count_0 = 0
        count_1 = 0
        count_2 = 0
        compare_thr_same = ((36)**2)*3 #36
        for pixel in segm_pixels:
            sides_different = 0

            #color_dif_upper1 = (b[max(pixel[0]-int(self.pixel_D*1.5),0)][pixel[1]] - b[pixel[0]][pixel[1]])**2 + (g[max(pixel[0]-int(self.pixel_D*1.5),0)] - g[pixel[0]][pixel[1]])**2 + (r[max(pixel[0]-int(self.pixel_D*1.5),0)][pixel[1]] - r[pixel[0]][pixel[1]])**2
            #color_dif_lower1 = (b[min(pixel[0]+int(self.pixel_D*1.5),self.img.shape[0]-1)][pixel[1]] - b[pixel[0]][pixel[1]])**2 + (g[min(pixel[0]+int(self.pixel_D*1.5),self.img.shape[0]-1)][pixel[1]] - g[pixel[0]][pixel[1]])**2 + (r[min(pixel[0]+int(self.pixel_D*1.5),self.img.shape[0]-1)][pixel[1]] - r[pixel[0]][pixel[1]])**2
            color_dif_upper2 = (b[max(pixel[0]-int(self.pixel_D*1.5),0)][pixel[1]] - color_cable[0])**2 + (g[max(pixel[0]-int(self.pixel_D*1.5),0)][pixel[1]] - color_cable[1])**2 + (r[max(pixel[0]-int(self.pixel_D*1.5),0)][pixel[1]] - color_cable[2])**2
            color_dif_lower2 = (b[min(pixel[0]+int(self.pixel_D*1.5),self.img.shape[0]-1)][pixel[1]] - color_cable[0])**2 + (g[min(pixel[0]+int(self.pixel_D*1.5),self.img.shape[0]-1)][pixel[1]] - color_cable[1])**2 + (r[min(pixel[0]+int(self.pixel_D*1.5),self.img.shape[0]-1)][pixel[1]] - color_cable[2])**2

            if color_dif_upper2 > compare_thr_same:
                sides_different += 1
            if color_dif_lower2 > compare_thr_same:
                sides_different += 1

            if sides_different == 0:
                segm_pixels2.append([pixel[0],pixel[1]])
                segm_img2[pixel[0]][pixel[1]] = 255
                count_0 +=1

            elif sides_different == 2:
                segm_pixels2.append([pixel[0],pixel[1]])
                segm_img2[pixel[0]][pixel[1]] = 255
                count_2 +=1
            
            else:
                segm_img_wrong[pixel[0]][pixel[1]] = 255
                count_1+=1

        #cv.imshow("Segm_test", segm_img)
        #cv.waitKey(0)
        #print("0: " + str((count_0/(count_0+count_1+count_2))*100))
        #print("1: " + str((count_1/(count_0+count_1+count_2))*100))
        #print("2: " + str((count_2/(count_0+count_1+count_2))*100))  

        segm_pixels3 = []
        kernel = np.ones((3,7), np.uint8) #5,11
        wrong_dilated = cv.dilate(copy.deepcopy(segm_img_wrong), kernel, iterations=1) 
        segm_img3 = np.zeros((self.img.shape[0], self.img.shape[1]), dtype='uint8')
        for pixel in segm_pixels2:
            if wrong_dilated[pixel[0]][pixel[1]]==0:
                segm_img3[pixel[0]][pixel[1]] = 255
                segm_pixels3.append([pixel[0],pixel[1]])

        #kernel = np.ones((1,3), np.uint8)
        #segm_img3 = cv.dilate(segm_img3, kernel, iterations=1)   #21,65

        return segm_img3, segm_pixels3
