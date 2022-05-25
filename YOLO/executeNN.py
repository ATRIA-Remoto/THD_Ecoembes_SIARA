from executeImageNN_lib import YOLONet
import os, cv2, time
from pushKey import *
from preprocessors_lib import SizePreprocessor
import progressbar

DETECTION_DIR = 'detection_test_YOLOv3_20201125'

# COmprueba una ruta especicada y si no existe la crea.
def checkCreateFilePath(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)

# Dibuja un rectangulo en los puntos indicados.
def pintarRectangulo(frame, bbox, color, tag=None, grosor=2):
    (x, y) = (int(bbox[0]), int(bbox[1]))
    (w, h) = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, grosor)
    if tag is not None:
        cv2.putText(frame, tag, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, grosor)
    return frame

TAGS = ['Barquilla', 'Carton', "Chatarra", "Film", "Raee"]

imagesPath = "D:/Proyectos/THD_Ecoembes/Imagenes/2020-09-01"
weightsPath = "D:\Proyectos\THD_Ecoembes\codigo\THD\YOLO/yolov3_20000.weights"
configPath = "D:\Proyectos\THD_Ecoembes\codigo\THD\YOLO/yolov3.cfg"
checkCreateFilePath(DETECTION_DIR)

net = YOLONet(configPath, weightsPath)


widgets = ["Detecting THD objects: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(os.listdir(imagesPath)), widgets=widgets).start()

for (j, imgFile) in enumerate(os.listdir(imagesPath)):
    imgPath = os.path.join(imagesPath, imgFile)
    img = cv2.imread(imgPath)
    (boxes, classIDs, confidences) = net.execute(img)
    k = None
    save = False
    for i, box in enumerate(boxes):
        if confidences[i] > 0.7:
            save = True
            img = pintarRectangulo(img, box, (0,255,0), tag=TAGS[classIDs[i]])
    if save:
        cv2.imwrite(os.path.join(DETECTION_DIR, imgFile), img)
    cv2.imshow('Detection', img)
    k = waitAnyKey(time=1)
    if k == ESC:
        break
    # if isEscPushed():
    #     break
    pbar.update(j)
pbar.finish()
cv2.destroyAllWindows()
    