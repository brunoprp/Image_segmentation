#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:42:49 2021

@author: bruno
"""

import numpy as np
import cv2


# definir os limites superior e inferior do pixel HSV
# intensidades a serem consideradas 'pele'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

video = cv2.VideoCapture('http://192.168.10.254:4747/video')


while True:
    #frame = cv2.imread('imagens/cor_car_1.jpeg')
    conectado, frame = video.read()
    
    # redimensione o frame, convertendo para o espaço de cores HSV
    # e determinar as intensidades de pixel HSV que se enquadram nos
    #  limites superiores e inferiores especificados
    frame = cv2.resize(frame, (700, 700))
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    
    # aplicando uma série de erosões e dilatações na máscara usando elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    # Desfocando a máscara para ajudar a remover o ruído 
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    
    cv2.imshow("frame", frame)
    cv2.imshow("mask Skin", skin)
    cv2.imshow("mask", skinMask)
    

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()