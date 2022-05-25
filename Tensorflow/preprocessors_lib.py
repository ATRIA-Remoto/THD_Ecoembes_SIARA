# En este fichero se añaden todos los preprocesos que se le hagan a una imagen antes de meterla a una red o de ser usada con 
# funciones de Vision por Computador, todas las clase se aplican con la funcion preprocess para facilitar su llamada.


import cv2
from tensorflow.keras.preprocessing.image import img_to_array


# Convierte las imagenes en un array para ser usadas por keras
class ImageToArrayPreprocessor:

    def __init__(self, dataFormat=None, dataType=None):
        self.dataFormat = dataFormat
        self.dataType = dataType

    def preprocess(self, img):
        arrayImg = img_to_array(img, data_format=self.dataFormat, dtype=self.dataType)
        return arrayImg


# Preprocesa imagenes para modificar su tamaño
class SizePreprocessor:

    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def preprocess(self, img):
        resizeImg = cv2.resize(img, (self.width, self.height), interpolation=self.interpolation)
        return resizeImg

# Preprocesa imagenes para modificar su formato
class FormatPreprocessor:

    def __init__(self, color=cv2.COLOR_BGR2GRAY):
        self.color = color

    def preprocess(self, img):
        colorImg = cv2.cvtColor(img, self.color)
        return colorImg

# Preprocesa imagenes obteniendo un roi especifico
class RoiPreprocessor:

    def __init__(self, y1, y2, x1, x2):
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2

    def preprocess(self, img):
        roi = img[self.y1:self.y2, self.x1:self.x2]
        return roi