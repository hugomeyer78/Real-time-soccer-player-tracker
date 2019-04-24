#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 02:06:45 2019

@author: hugomeyer
"""

from image import Image
from text_recognition import text_reco
import matplotlib.pyplot as plt

path = '../Images/psg_info.png'


def main():
    img = Image(path=path, name='psg_info_out')
    #img.resize_no_distortion()
    #img.convert_to_gray()
    #img.thresholding()
    #img.export('../Images')
    textX, textY = text_reco(img.pix_vals)
    plt.figure(figsize=(10, 5))
    plt.scatter(textX, textY)
    plt.show
    #img.display()





if __name__ == '__main__':    
    main()