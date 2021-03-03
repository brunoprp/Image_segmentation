#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:30:30 2021

@author: bruno
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

class SegImg():
    def __init__(self, img_path):
        self.img_path = img_path
        
        
    def plotImg(self, list_imgs, name_seve):
        fig, ax1 = plt.subplots(2, 2, figsize=(11, 8))
        
        fig.suptitle("Resultado da segmentação")
        
        ax1[0][0].imshow(list_imgs[0])
        ax1[0][0].set_title("Imagem original")
        
        ax1[0][1].imshow(list_imgs[1])
        ax1[0][1].set_title("Imagem segmentada")
        
        ax1[1][0].imshow(list_imgs[2])
        ax1[1][0].set_title("Imagem segmentada com cor")
        
        ax1[1][1].imshow(list_imgs[3])
        ax1[1][1].set_title("Imagem final com contornos")
        
        plt.tight_layout()
        plt.savefig(name_seve+'.jpg', format='jpg')
        plt.show()
        
    
    
    def segImagem(self):
        #Lendo a imagem e melhorando o contraste da mesma
        img = cv2.imread(self.img_path) # Lendo a imagem    
        img_normal = img.copy()
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Tranformando em escala de cinsar 1 canal
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)) # Criando o objeto de equalização
        cl1 = clahe.apply(gray_image) # Aplicando o equalizador na imagem 
        
        
        # Segmentado a imagem
        output_adapthresh = cv2.adaptiveThreshold(cl1, 255.0,
		cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 3)
        
        # Invertendo as cores da segmentação
        for i in range(0, output_adapthresh.shape[0]):
            for j in range(0, output_adapthresh.shape[1]):
                if output_adapthresh[i][j] == 255:
                    output_adapthresh[i][j] = 0
                else:
                    output_adapthresh[i][j] = 255
                    
        # Removendo ruido com morfologia matematica
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(output_adapthresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=2)            
        
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        
        
        cl1 = cv2.cvtColor(cl1, cv2.COLOR_BGR2RGB)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        
        markers = cv2.watershed(img,markers)
        cl1[markers == -1] = [255,0,0]
        
        output_adapthresh_1 = cv2.cvtColor(output_adapthresh, cv2.COLOR_BGR2RGB)
        
        
        return [img_normal, output_adapthresh_1, output_adapthresh, cl1]
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        