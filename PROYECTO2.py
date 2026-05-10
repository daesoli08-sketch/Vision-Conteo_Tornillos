# -*- coding: utf-8 -*-
"""
Created on Sun May  3 06:57:39 2026

@author: Daniela
"""

from skimage import data, color
import numpy as np
import matplotlib.pyplot as plt
import cv2

def filtro_espacial (img, tam = 5, sigma = 0.5):
    # 1. Crear el kernel Gaussiano manualmente
    # Esto demuestra que conoces la teoría matemática
    ax = np.linspace(-(tam - 1) / 2., (tam - 1) / 2., tam)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    
    # 2. Normalizar el kernel (la suma debe ser 1)
    kernel = kernel / np.sum(kernel)
    
    # 3. Aplicar la convolución con tu kernel propio
    return cv2.filter2D(img, -1, kernel)

def transformacion_log(img):
#Normalizacion
    img_norm = (img - img.min())/(img.max()- img.min())

#Transformacion
    c = 1
    log_transform = c*np.log(1 + img_norm)  #Modifica la intensidad de pixeles

    log_transform = np.round(log_transform * 255).astype(np.uint8)
    return log_transform

def obtener_bordes(img_g):
#CV_64F = es usado para no perder valores negativos en la resta
    borde_x = cv2.Sobel(img_g, cv2.CV_64F, 1, 0, ksize =3)
    borde_y = cv2.Sobel(img_g, cv2.CV_64F, 0, 1, ksize =3)
#Se usa absoluto para pasar los valores a 8 bits (0/255)
    abs_borde_x = cv2.convertScaleAbs (borde_x)
    abs_borde_y = cv2.convertScaleAbs (borde_y)
#Calculando borde total
    return  cv2.addWeighted(abs_borde_x, 0.5, abs_borde_y, 0.5, 0)
    
def umbralizacion (img_filtrada):
#
    val_otsu, img_seg = cv2.threshold(img_filtrada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#MERFOLOGIA
    kernel = np.ones((3,3), np.uint8)
    img_limpia = cv2.morphologyEx(img_seg,cv2.MORPH_OPEN, kernel)
#CONTANDO OBJETOS
    obj_con, obj = cv2.connectedComponents(img_limpia)
    return img_limpia, obj_con - 1

#%%CODIGO FUENTE  
ruta = ("C:/Users/Daniela/OneDrive/Documentos/8 SEMESTRE/VISION ROBOTICA/PROYECTO 2/Tornillos1.jpg")
img = cv2.imread(ruta)
    #TRANSFORMCION A ESC GRISES
img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #TRANSFORMACION DE INTENSIDAD
img_log = transformacion_log(img_g)
    #FILTRADO ESPACIAL
img_filtrada = filtro_espacial(img_log, tam = 5, sigma = 0.4)
    #DETECTANDO BORDES
bordes = obtener_bordes (img_filtrada)
 
#%%Ploteado  
plt.figure(1)
plt.subplot(2, 3, 1), plt.imshow(img_g, cmap='gray'),plt.title("1. Original (Gris)"),plt.axis('off')
plt.subplot(2, 3, 2), plt.imshow(img_log, cmap='gray'), plt.title("2. Transf. Logarítmica"), plt.axis('off')
plt.subplot(2, 3, 3), plt.imshow(img_filtrada, cmap='gray'), plt.title("3. Filtro Espacial"), plt.axis('off')


plt.tight_layout()
plt.show()