# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:37:57 2026

@author: wicho
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import data
import os

def filtro_espacial (img, tam = 5, sigma = 0.5):
    #Suaviza la imagen, mejorando los bordes el pixel se parece mas a su vecino cercano que al lejano
    ax = np.linspace(-(tam - 1) / 2., (tam - 1) / 2., tam)      #Eje de coordenadas (x,y /Con respecto al kernel) (Creando un vector centrado en 0)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))     #Se realiza la parte exponencial del Gauss
        #Donde sigma es la desviacion estandar (-Suaviza poco/ + suaviza mucho)
    kernel = np.outer(gauss, gauss)                             #Produccto exterior del Gaussiano
    kernel = kernel / np.sum(kernel)                            #Normalizar el kernel (la suma debe ser 1, para no cambiar iluminancion)
    return cv2.filter2D(img, -1, kernel)                        #Convoluciona la matrix a la imagen

def transformacion_log(img):
    #Realza las zonas oscuras y comprime las zonas claras
#Normalizacion
    img_norm = (img - img.min())/(img.max()- img.min())
#Transformacion
    c = 1
    log_transform = c*np.log(1 + img_norm)  #Modifica la intensidad de pixeles
    log_transform = np.round(log_transform * 255).astype(np.uint8)
    return log_transform

def clahe (img):        #Contrast Limited Adaptive Histogram Equalization
    #Ayuda en las imagenes de tornillos para resaltar el metal de la madera
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Cra el objeto con limitacion de 2.0, para que no amplifique demasiado las zonas uniformes
    #Dividiendo la imagen en una matriz de 8 * 8, analizando de matriz por matriz
    return clahe.apply(img_g)

def umbralizacion (img_filtrada):
    u, th2 = cv2.threshold(img_filtrada, 125, 255, cv2.THRESH_BINARY_INV)
    return th2

#+cv2.THRESH_OTSU

def DetBord(gray):
    # Convertimos la imagen a tipo float32 para evitar pérdidas en cálculos de gradiente
    gray = np.float32(gray)

    # Suavizado con filtro Gaussiano (reduce ruido)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

    #Detector de Bordes Canny

    # Suavizado nuevamente (este sí se usa para Canny)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)

    # Normalizamos a rango [0,255] y convertimos a uint8
    # (Canny requiere imagen de 8 bits)
    blur_uint8 = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Detector de bordes Canny
    # 50 y 150 = umbrales para detector
    edges_canny = cv2.Canny(blur_uint8, 50, 150)

#40 350
    # Retornas SOLO Canny (los otros cálculos no se usan)
    return edges_canny


def Cont(img_bin):
    # Convertimos la imagen binaria (1 canal) a BGR (3 canales)
    # Esto permite dibujar en color (rectángulos, textos, etc.)
    resultado_img = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

    # MORFOLOGÍA (limpieza de ruido)

    # Creamos un kernel (matriz 3x3 de unos)
    # Se usa para operaciones morfológicas
    kernel = np.ones((3,3), np.uint8)

    # OPENING = erosión seguida de dilatación
    # Sirve para eliminar ruido pequeño (pixeles blancos aislados)
    opening = cv2.morphologyEx(img_bin,cv2.MORPH_OPEN,kernel,iterations=1)

    # CLOSING = dilatación seguida de erosión
    # Sirve para cerrar huecos dentro de los objetos
    # (ej. imperfecciones en tornillos o tuercas)
    closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel,iterations=2    )

    #Detección de contornos

    # Busca los contornos externos de los objetos
    # RETR_EXTERNAL: solo contornos exteriores
    # CHAIN_APPROX_SIMPLE: simplifica puntos del contorno
    contornos, _ = cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # Lista donde se guardarán las propiedades de cada objeto
    resultados = []

    #Medición de Propiedades
    for cnt in contornos:

        # Área del contorno (número de pixeles)
        area = cv2.contourArea(cnt)

        # Filtrar ruido:
        # - muy pequeños → ruido
        # - muy grandes → posibles errores
        if area < 250 or area > 30000:
            continue

        # Rectángulo envolvente (bounding box)
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtrar objetos demasiado pequeños en ancho o alto
        if w < 10 or h < 10:
            continue

        # Perímetro del contorno
        perimetro = cv2.arcLength(cnt, True)

        # CENTROIDE (centro de masa del objeto)
        M = cv2.moments(cnt)

        # Evitar división entre cero
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        #Circularidad
        # Fórmula: 4π * Área / Perímetro²
        # - ≈1 → forma circular
        # - menor → forma alargada o irregular
        if perimetro != 0:
            circularidad = 4 * np.pi * area / (perimetro**2)
        else:
            circularidad = 0

        # Guardar resultados en diccionario
        resultados.append({
            "objeto": len(resultados)+1,
            "area": round(area,2),
            "perimetro": round(perimetro,2),
            "centroide": (cx, cy),
            "bounding_box": (x,y,w,h),
            "circularidad": round(circularidad,3)
        })

        #Dibuja los resultados sobre la imagen

        # Dibujar rectángulo alrededor del objeto
        cv2.rectangle(resultado_img,(x,y),
            (x+w, y+h),
            (0,255,0),  # verde
            2
        )

        # Dibujar centroide
        cv2.circle(
            resultado_img,
            (cx,cy),
            4,
            (0,0,255),  # rojo
            -1
        )

        # Escribir número del objeto
        cv2.putText(resultado_img,str(len(resultados)),(x, y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0), 2) # azul

#Dando propiedades.
    print(f"Objetos detectados: {len(resultados)}")
    
    # Imprimir propiedades de cada objeto
    for r in resultados:
        print(f"\nObjeto {r['objeto']}")
        print(f"Área: {r['area']}")
        print(f"Perímetro: {r['perimetro']}")
        print(f"Centroide: {r['centroide']}")
        print(f"Bounding Box: {r['bounding_box']}")
        print(f"Circularidad: {r['circularidad']}")

    return resultado_img



#%%CODIGO FUENTE  
ruta = ("C:/Users/Daniela/OneDrive/Documentos/8 SEMESTRE/VISION ROBOTICA/IMAGENES/semillas3.jpg")
img = cv2.imread(ruta)
if img is not None:
    #TRANSFORMCION A ESC GRISES
    img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #TRANSFORMACION DE CONTRASTE
    img_cont = clahe(img_g)
    #TRANSFORMACION DE INTENSIDAD
    img_log = transformacion_log(img_g)
    #FILTRADO ESPACIAL
    img_filtrada = filtro_espacial(img_cont, tam = 7, sigma = 1.2)
    #UMBRALIZACION
    img_bin= umbralizacion(img_filtrada)
    #Detector de bordes
    gray = DetBord(img_log)
    #Contador
    contador = Cont(img_bin)

 
#ploteado 
    plt.figure(1)
    plt.subplot(2, 4, 1), plt.imshow(img_g, cmap='gray'),plt.title("1. Escala de Gris"),plt.axis('off')
    plt.subplot(2, 4, 2), plt.imshow(img_cont, cmap='gray'), plt.title("2. Transf. Clare"), plt.axis('off')
    plt.subplot(2, 4, 3), plt.imshow(img_log, cmap='gray'), plt.title("3. Transf. Logarítmica"), plt.axis('off')
    plt.subplot(2, 4, 4), plt.imshow(img_filtrada, cmap='gray'), plt.title("4. Filtro Espacial"), plt.axis('off')
    plt.subplot(2, 4, 5), plt.imshow(img_bin, cmap='gray'), plt.title("5. Umbralización"), plt.axis('off')
    plt.subplot(2, 4, 6), plt.imshow(gray, cmap='gray'), plt.title("6. Detector de bordes"), plt.axis('off')
    plt.subplot(2, 4, 7), plt.imshow(contador, cmap='gray'), plt.title("7. Conteo de piezas"),plt.axis("off")
    
    plt.tight_layout()
    plt.show()

else:
    print("Revisa la ruta. No se encontro la imagen.")