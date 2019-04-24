#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 02:12:37 2019

@author: hugomeyer
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt



class Image(object):
    def __init__(self, img=None, path=None, name=None):
        if img is not None:
            self.pix_vals = img
        elif path is not None:
            self.pix_vals = cv2.imread(path)
        else:
            raise ValueError("Provide either an image array (img) or a file (path)")
        if name is not None:
            self.name = name
        else:
            self.name = "IMG"

    '''
    def face_detect(self, mode, model):
        img, self.face_detected = face_detection(self.pix_vals.copy(), mode, model)
        
        if self.face_detected == False and model=='both':
            img, self.face_detected = face_detection(self.pix_vals.copy(), mode, 'dlib')
            
        self.pix_vals = img
            
            
        if Export.no_face_detected_export == True and self.face_detected == False:
            self.export(Export.no_face_detected_path)
            
        
            
        return img.shape[1], img.shape[0]
    '''


    
    def convert_to_gray(self): 
        self.pix_vals = cv2.cvtColor(self.pix_vals, cv2.COLOR_BGR2GRAY)

    def export(self, path, exp_format="png"):
        export_path=os.path.join(path, self.name) + '.' + exp_format
        
        #if self.pix_vals.shape[0] < self.pix_vals.shape[1]:
         #   export_img = self.pix_vals
          #  cv2.imwrite(export_path, export_img)
        #else:
         #   print('coucou')
        cv2.imwrite(export_path, self.pix_vals)


   

    def resize(self, w, h):
        self.pix_vals = cv2.resize(self.pix_vals,(w,h))

    def rotate(self, angle, keep_same_dim=False):
        if keep_same_dim == False:
            k = angle/90
            rotated = np.rot90(self.pix_vals, k)
            self.pix_vals = rotated

        else:
            (h, w) = self.pix_vals.shape[:2]
            scale=h/w
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            self.pix_vals = cv2.warpAffine(self.pix_vals, M, (w, h))
        
        
        
    def crop(self, yi, yf, xi, xf, inplace=True):
        img = self.pix_vals[yi:yf, xi:xf]
        
        if inplace == True:
            self.pix_vals = img
        else:
            return img
        return 1

    def thresholding(self):
        #_,self.pix_vals = cv2.threshold(self.pix_vals,127,255,cv2.THRESH_BINARY)
        self.pix_vals = cv2.adaptiveThreshold(self.pix_vals,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    def display(self):
        plt.imshow(self.pix_vals,'gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
        
    def resize_no_distortion(self, size=300):
        
        im = self.pix_vals.copy()
        old_size = im.shape[:2] # old_size is in (height, width) format
        
        ratio = float(size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        
        # new_size should be in (width, height) format
        
        im = cv2.resize(im, (new_size[1], new_size[0]))
        
        delta_w = size - new_size[1]
        delta_h = size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        
        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        
        self.pix_vals = new_im