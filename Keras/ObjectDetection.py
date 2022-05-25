from executeImageNN_lib import KerasNet, ObjDetector
import os, cv2
from pushKey import *
from preprocessors_lib import *

MODEL = "D:\Proyectos\THD_Ecoembes\codigo\modelo.h5"
LABELS = ["barquilla", "carton", "chatarra", "film", "raee"]
IMG_DIR = "D:\Proyectos\THD_Ecoembes\Imagenes/2020-09-07"

detector = ObjDetector(MODEL, LABELS, preprocessors=[SizePreprocessor(224, 224), FormatPreprocessor(color=cv2.COLOR_BGR2RGB), ImageToArrayPreprocessor()])

for imageName in os.listdir(IMG_DIR):
    imagePath = os.path.join(IMG_DIR, imageName)
    (preds, img) = detector.detect(cv2.imread(imagePath))
    if len(preds) > 0:
        cv2.imshow("preds", cv2.resize(img, (1024, 720)))
        k = waitAnyKey()
        if k == ESC:
            break
cv2.destroyAllWindows()
