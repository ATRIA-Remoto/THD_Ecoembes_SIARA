from XimeaCam_lib import XimeaCam
import cv2
import time
from datetime import datetime
import PIL.Image
import os
import xml.etree.ElementTree as ET
from ctypes import *
from ctypes.wintypes import *
from skimage.metrics import structural_similarity

# Fichero con los parametros de la camara
XML = "D:/Proyectos/THD_Ecoembes/codigo/THD/TomaFotos/params.xicamera"

# Carpeta donde guardar las imagenes
FOLDER = 'D:/Proyectos/THD_Ecoembes/codigo/THD/Fotos'

# Compara dos imagenes, si superan un determinado threshold se consideran iguales y devuelve true
def compareImages(img1, img2, threshold=0.7):
    (score, diff) = structural_similarity(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 
            cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), full=True)
    # print(score)
    if score >= threshold:
        return True
    else:
        return False

if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)

#create instance for first connected camera
cam = XimeaCam(XML=XML)

try:
    lastImg = cam.takePhoto()
    print('Starting video. Press CTRL+C to exit.')
    t0 = time.time()
    while True:
        date = datetime.today()
        #get data and pass them from camera to img
        img = cam.takePhoto()
        
        ##PARA GUARDAR FOTOS EN CARPETAS

        pathToSave = FOLDER + '/' + date.strftime('%Y-%m-%d')
        if not os.path.exists(pathToSave):  # Se crea el directorio si no existe
            os.mkdir(pathToSave)

        # Compara las dos ultimas imagenes, si son muy parecidas, no guarda una nueva
        if not compareImages(img, lastImg):
            fileName = pathToSave + '/' + date.strftime('%Y%m%d_%H-%M-%S') + '.jpg'

            cv2.imwrite(fileName, img)

        # Muestra la ultima imagen obtenida
        cv2.imshow('img_ximea', cv2.resize(img, (640,480)))
        lastImg = img       # Se guarda como ultima imagen
        cv2.waitKey(3000)   # Se esperan 3 segundos para tomar la siguiente imagen
        
# Si se para la ejecución por teclado se cierran las ventanas
except KeyboardInterrupt:   
    cv2.destroyAllWindows()

# Se detiene la camara
print('Stopping acquisition...')
cam.close()

print('Done.')

