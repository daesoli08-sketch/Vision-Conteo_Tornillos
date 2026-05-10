import numpy as np
import matplotlib.pyplot as plt
import cv2

def filtro_espacial(img, tam = 5, sigma = 1.0):
    ax = np.linspace (-(tam-1)/2.,(tam-1)/2., tam)
    gauss = np.exp(-0.5 * np.square(ax)/ np.square (sigma))
    
    kernel = np.outer(gauss, gauss)
    
    kernel = kernel/np.sum(kernel)
    return cv2.filter2D(img, -1, kernel)
    

def transformacion_log(img):
    # Normalización estable
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
    # s = c * log(1 + r)
    log_transform = np.log1p(img_norm) 
    return np.round(log_transform * 255).astype(np.uint8)

def obtener_bordes(img_g):
    # Uso de Sobel con CV_64F para no perder transiciones negativas
    borde_x = cv2.Sobel(img_g, cv2.CV_64F, 1, 0, ksize=3)
    borde_y = cv2.Sobel(img_g, cv2.CV_64F, 0, 1, ksize=3)
    abs_borde_x = cv2.convertScaleAbs(borde_x)
    abs_borde_y = cv2.convertScaleAbs(borde_y)
    return cv2.addWeighted(abs_borde_x, 0.5, abs_borde_y, 0.5, 0)

def umbralizacion(img_filtrada):
   valor_umbral, img_binaria = cv2.threshold(img_filtrada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
   kernel_um = np.ones((5,5), np.uint8)
   limpia = cv2. morphologyEx (img_binaria, cv2.MORPH_OPEN, kernel_um)
   relleno = cv2.morphologyEx(limpia, cv2.MORPH_CLOSE, kernel_um)
   return relleno, valor_umbral

def conteo(img_binaria, img_original):
    estadisticas = cv2.connectedComponentsWithStats(img_binaria, 8, cv2.CV_32S)
    num_lab, lab, stats, centros = estadisticas
    img_resultado = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR) # Para dibujar en color
    cont_act = 0
    for i in range(1, num_lab):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        # UMBRAL DE ÁREA: Si el objeto es menor a 800 píxeles, es ruido/madera
        # Ajusta este valor (800) dependiendo de qué tan cerca esté la cámara
        if area > 800:
            cont_act += 1
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_resultado, f"#{cont_act}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return cont_act, img_resultado

#%% EJECUCIÓN
ruta = "C:/Users/Daniela/OneDrive/Documentos/8 SEMESTRE/VISION ROBOTICA/PROYECTO 2/semillas2.jpg"
img = cv2.imread(ruta)

if img is not None:
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_log = transformacion_log(img_g)
    
    # 3. FILTRO DE MEDIANA (Ajuste a tam=3 para nitidez máxima)
    img_filtrada = filtro_espacial(img_log, tam = 5, sigma = 0.3 )
    
   # 4. SEGMENTACIÓN
    binaria, val_u = umbralizacion(img_filtrada)
    
    # 5. CONTEO Y DIBUJO
    piezas_fin, img_final = conteo(binaria, img_g)
    
    print(f"Piezas detectadas: {piezas_fin}")

    # Visualización completa para el reporte
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1), plt.imshow(img_g, cmap='gray'), plt.title("1. Original")
    plt.subplot(2, 3, 2), plt.imshow(img_log, cmap='gray'), plt.title("2. Logarítmica")
    plt.subplot(2, 3, 3), plt.imshow(img_filtrada, cmap='gray'), plt.title("3. Filtro Gaussiano")
    plt.subplot(2, 3, 4), plt.imshow(binaria, cmap='gray'), plt.title("4. Umbralización Otsu")
    plt.subplot(2, 3, 5), plt.imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)), plt.title(f"5. Detección Final: {piezas_fin}")
    
    plt.tight_layout()
    plt.show()