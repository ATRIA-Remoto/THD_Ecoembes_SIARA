from executeImageNN_lib import YOLONet
from pushKey import keyPushed, ESC
import os, time
import cv2
from preprocessors_lib import SizePreprocessor, FormatPreprocessor
import imutils

# Ficheros de configuracion
PATH_TO_SAVED_MODEL = "D:/Proyectos/THD_Ecoembes/codigo/THD/YOLO/backup_train/yolov4-leaky-416_best5_map-82-86.weights"
CONFIG_PATH = "D:/Proyectos/THD_Ecoembes/codigo/THD/YOLO/data/yolov4-leaky-416.cfg"
LABELS_NAMES_PATH = "D:/Proyectos/THD_Ecoembes/codigo/THD/YOLO/data/THD.names"
PATH_IMAGES = "D:/Proyectos/THD_Ecoembes/Labels/YOLO/NoBorders/3_classes/Test"

SCORE_THRESHOLD = 0.5   # Minima puntuacion para aceptar la deteccion

# Tamaño de laimagen
SIZE_IMSHOW_X = 1280
SIZE_IMSHOW_Y = 960

DIMS_IMG = (SIZE_IMSHOW_X, SIZE_IMSHOW_Y)

EXTENSIONS = ["jpg", "png", "bmp"]

# Carga del modelo
print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
preProcessors = [SizePreprocessor(416, 416)]
detect_fn = YOLONet(CONFIG_PATH, PATH_TO_SAVED_MODEL, namesPath=LABELS_NAMES_PATH, confidence=SCORE_THRESHOLD, preprocesors=preProcessors)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

is_file = False

if os.path.isdir(PATH_IMAGES):
    listFiles = os.listdir(PATH_IMAGES)
    listFiles = [f for f in listFiles if f.split(".")[1] in EXTENSIONS]
    key = None
    for imgFile in listFiles:
        imgPath = os.path.join(PATH_IMAGES, imgFile)
        if os.path.isfile(imgPath):
            img = cv2.imread(imgPath)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            (boxes, classes, scores) = detect_fn.execute(img)

            if len([s for s in scores if s >= SCORE_THRESHOLD]) > 0:
                print(imgPath)
                imgResults = detect_fn.showResults(img, boxes, classes, scores, min_score_thresh=SCORE_THRESHOLD)
                print(scores)
                cv2.imshow('image',cv2.resize(imgResults, DIMS_IMG))
                key = keyPushed(time=0)
            else:
                cv2.imshow('image',cv2.resize(img, DIMS_IMG))
                key = keyPushed(time=1)
            if key == ESC:
                break
else:
    img = cv2.imread(PATH_IMAGES)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    (boxes, classes, scores) = detect_fn.execute(img)

    if len([s for s in scores if s >= SCORE_THRESHOLD]) > 0:
        img = detect_fn.showResults(img, boxes, classes, scores, min_score_thresh=SCORE_THRESHOLD)
    cv2.imshow('image',cv2.resize(img, DIMS_IMG))
    key = keyPushed(time=0)

cv2.destroyAllWindows()