import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import numpy as np
import cv2 as cv
import copy
import time
from typing import Tuple
from PIL import Image
import math

#from interfaces import TAUSegmentationInterfaceUNet


#class TAUSegmentationUNet(TAUSegmentationInterfaceUNet):
class TAUSegmentationUNet():
    '''
    Implements the segmentation via Dilate(edges) AND color_filter 
    '''

    def __init__(self, input_img: np.ndarray, all_colors: list, model, IMAGE_SIZE: int = 512, init_col: int = 0):

        self.img = input_img
        self.all_colors = all_colors
        self.init_col = init_col
        self.model = model
        self.mask_model_final = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=np.uint8)
        if self.img.shape[0] < self.img.shape[1]: #If wide image
            for i in range(math.floor(self.img.shape[1]/self.img.shape[0]),-1,-1):
                image_model = (copy.deepcopy(self.img))[:, min(self.img.shape[0]*(i), self.img.shape[1]-self.img.shape[0]):min(self.img.shape[0]*(i+1), self.img.shape[1])]
                mask_model_resized = self.get_mask_model(image_model, IMAGE_SIZE)
                self.mask_model_final[:, min(mask_model_resized.shape[0]*(i), self.img.shape[1]-mask_model_resized.shape[0]):min(mask_model_resized.shape[0]*(i+1), self.img.shape[1])] = mask_model_resized

        elif self.img.shape[0] > self.img.shape[1]: #If tall image
            for i in range(math.floor(self.img.shape[0]/self.img.shape[1]),-1,-1):
                image_model = (copy.deepcopy(self.img))[min(self.img.shape[1]*(i), self.img.shape[0]-self.img.shape[1]):min(self.img.shape[1]*(i+1), self.img.shape[0]), :]
                mask_model_resized = self.get_mask_model(image_model, IMAGE_SIZE)
                self.mask_model_final[min(mask_model_resized.shape[1]*(i), self.img.shape[0]-mask_model_resized.shape[1]):min(mask_model_resized.shape[1]*(i+1), self.img.shape[0]), :] = mask_model_resized

        else:
            self.mask_model_final = self.get_mask_model(copy.deepcopy(self.img), IMAGE_SIZE)
        
        #cv.imshow("unet filter3", self.mask_model_final)
        cv.imwrite(os.path.join(os.path.dirname(__file__), '../imgs/filter_image_0.jpg'), self.mask_model_final)


    def get_mask_model(self, image_model, IMAGE_SIZE):

        original_model_size = (image_model.shape[0], image_model.shape[1])
        image_model = cv.resize(image_model, (IMAGE_SIZE, IMAGE_SIZE)) #resize
        x = image_model/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        #start_time = time.time()
        pred_mask = self.model.predict(x)[0] > 0.5
        #print("Total time: " + str(time.time() - start_time) + "s")
        prediction_image = tf.keras.preprocessing.image.array_to_img(pred_mask)
        prediction_image2 = tf.keras.preprocessing.image.img_to_array(prediction_image)
        mask_model_col = cv.cvtColor(prediction_image2.astype('uint8'), cv.COLOR_RGB2BGR)
        mask_model_gray = cv.cvtColor(mask_model_col, cv.COLOR_BGR2GRAY)
        th, mask_model = cv.threshold(mask_model_gray, 128, 255, 0)
        mask_model_resized = cv.resize(mask_model, (original_model_size[1], original_model_size[0])) #resize back
        return mask_model_resized


    def exec(self, color_cable: list, thr1: float = 60, thr2: float = 1.5,erosion: int = 3) -> Tuple[np.ndarray, list]:

        b,g,r = cv.split(self.img)
        blank_cables = np.zeros((self.img.shape[0], self.img.shape[1]), dtype='uint8')
        compare_thr = ((thr1*0.1)**2)*3 #Equivalent to a 10% diff in each of the 3 color channels
        for row in range(self.img.shape[0]):
            for col in range(self.img.shape[1]):
                if self.mask_model_final[row, col]==255: #To not check all the pixels and safe time, as we will do a intersection of color filter and dilated edge detection
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
