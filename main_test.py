# -*- coding: utf-8 -*-
"""
Created on Mon May 11 08:54:59 2026

@author: Daniela

"""

import cv2
import matplotlib.pyplot as plt
# IMPORTANTE: Aquí importas tus propias funciones de la subcarpeta
from vision_conteo.conteoPiezas import (clahe, filtro_espacial, transformacion_log, umbralizacion, DetBord, Cont)
ruta = "IMAGENES/semillas3.jpg" 
img = cv2.imread(ruta)

if img is None:
    print("Revisa la ruta. No se encontro la imagen.")
else:
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_cont = clahe(img_g)
    img_log = transformacion_log(img_g)
    img_filtrada = filtro_espacial(img_cont, tam=7, sigma=1.2)
    img_bin = umbralizacion(img_filtrada)
    img_bordes = DetBord(img_log)
    
    # Mostrando las áreas y centroides automáticamente
    img_resultado = Cont(img_bin)

    plt.figure(1)
    
    plt.subplot(2, 2, 1), plt.imshow(img_g, cmap='gray'), plt.title("1. Original (Gris)"), plt.axis('off')
    plt.subplot(2, 2, 2), plt.imshow(img_bin, cmap='gray'), plt.title("2. Segmentación (Binaria)"), plt.axis('off')
    plt.subplot(2, 2, 3), plt.imshow(img_bordes, cmap='gray'), plt.title("3. Detección de Bordes"), plt.axis('off')
    plt.subplot(2, 2, 4), plt.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB)), plt.title("4. Resultado Final (Conteo)"), plt.axis('off')

    plt.tight_layout()
    plt.show()