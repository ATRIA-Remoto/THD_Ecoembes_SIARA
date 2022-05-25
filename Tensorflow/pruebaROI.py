import cv2, os
import numpy as np
from pushKey import *
from preprocessors_lib import RoiPreprocessor

# Dibuja un rectangulo en los puntos indicados.
def pintarRectangulo(frame, bbox, color, tag=None, grosor=2):
    (x, y) = (int(bbox[0]), int(bbox[1]))
    (w, h) = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, grosor)
    if tag is not None:
        cv2.putText(frame, tag, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, grosor)
    return frame

DIR = "D:/Proyectos/THD_Ecoembes/Labels/YOLO/Train"

listFiles = os.listdir(DIR)
imgName = None
for f in listFiles:
    if f.split('.')[1].lower() == 'jpg':
        imgName = f
        break

img = cv2.imread(os.path.join(DIR, listFiles[0]))

recorte = {
    'y1': 0,
    'y2': np.shape(img)[0],
    'x1': 150,
    'x2': 1050
}

ROI = RoiPreprocessor(recorte.get('y1'), recorte.get('y2'), recorte.get('x1'), recorte.get('x2'))

for f in listFiles:
    ext = f.split(".")[1]
    if ext.lower() == 'jpg':
        fPath = os.path.join(DIR, f)
        img = cv2.imread(fPath)
        imgROI = ROI.preprocess(img.copy())
        (H, W, D) = np.shape(img)
        
        lPath = fPath.split('.')[0] + '.txt'
        label = open(lPath, 'r')
        bboxes = []
        for line in label:
            if line != "":
                (c, centerX, centerY, width, height) = line.split(' ')
                centerX = float(centerX)
                centerY = float(centerY)
                width = float(width)
                height = float(height)
                x = centerX - width/2
                y = centerY - height/2
                bbox = [x*W, y*H, width*W, height*H]
                bboxes.append(bbox)
        label.close()
        for bbox in bboxes:
            img = pintarRectangulo(img, bbox, (255,0,0))
            # Se adapta a la ROI
            (x, y, w, h) = bbox
            if x > recorte.get('x2'):
                pass
            else:
                x = x - recorte.get('x1')
                if x < 0.0:
                    w = w - (-x)
                    x = 0.0
                
                if (x+w) > np.shape(imgROI)[1]:
                    if x < np.shape(imgROI)[1]:
                        w = np.shape(imgROI)[1] - x

            imgROI = pintarRectangulo(imgROI, [x,y,w,h], (255,0,0))

        cv2.imshow('original', img)
        cv2.imshow('ROI', imgROI)
        k = cv2.waitKey(0) & 0xff
        if k == ESC:
            break
cv2.destroyAllWindows()