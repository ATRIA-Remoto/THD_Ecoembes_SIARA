import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from preprocessors_lib import ImageToArrayPreprocessor
import cv2, os
from PIL import Image
import numpy as np
from FileManagement_lib import checkCreateFilePath
import progressbar
from executeImageNN_lib import TFODNeT
from pushKey import keyPushed, ESC

# Extrae las imagenes de los TFRecords
def extractTFRecordImages(tfrecord):
    images = []
    boxes = []
    categoriesDecoded = []
    record_iterator = tf.compat.v1.python_io.tf_record_iterator(tfrecord)
    k = 0
    for string_record in record_iterator:
        k = k + 1
    record_iterator = tf.compat.v1.python_io.tf_record_iterator(tfrecord)
    widgets = ["Extracting images from tfrecord: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=k, widgets=widgets).start()
    f = None
    for (j, string_record) in enumerate(record_iterator):
        example = tf.train.Example()
        example.ParseFromString(string_record)

        categories = example.features.feature["image/object/class/text"].bytes_list.value
        if len(categories) > 0:
            box = []
            for (k, xmin) in enumerate(example.features.feature['image/object/bbox/xmin'].float_list.value):
                box.append([xmin, example.features.feature['image/object/bbox/ymin'].float_list.value[k], 
                    example.features.feature['image/object/bbox/xmax'].float_list.value[k],
                    example.features.feature['image/object/bbox/ymax'].float_list.value[k]])

            auxCategories = []
            for c in categories:
                auxCategories.append(c.decode('utf-8'))

            categoriesDecoded.append(auxCategories)

            image = example.features.feature["image/encoded"].bytes_list.value[0]
            f=open('./tf.txt', 'w')
            print(str(example.features.feature), file=f)
            f.close()
            decoded = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
            images.append(decoded)
            boxes.append(box)
    return images, categoriesDecoded, boxes

# Rutas a los ficheros de configuracion
PATH_TO_SAVED_MODEL = "D:/Proyectos/THD_Ecoembes/codigo/THD/Tensorflow/experiments/faster_resnet50_adam/saved_model"
PATH_IMAGES = "D:/Proyectos/THD_Ecoembes/Labels/TF_ALT/3classesV/Test"
PATH_TO_LABELS = "D:/Proyectos/THD_Ecoembes/Labels/TF_ALT/3classesV/label_map.pbtxt"

SCORE_THRESHOLD = 0.5   # Minima scores de confianza para aceptar la deteccion
# tamaño de la imagen
SIZE_IMSHOW_X = 640
SIZE_IMSHOW_Y = 480

DIMS_IMG = (SIZE_IMSHOW_X, SIZE_IMSHOW_Y)

# Se carga el modelo
print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = TFODNeT(PATH_TO_SAVED_MODEL, PATH_TO_LABELS)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


is_file = False
if len(PATH_IMAGES.split('.')) > 1  and PATH_IMAGES.split('.')[1].lower() == 'tfrecord':
    is_file = True

# Si es solo un fichero se gestionan las imagenes del mismo
if is_file:
    tfrecord = PATH_IMAGES
    (images, categories, boxes) = extractTFRecordImages(tfrecord)
    # Se comprueban las imagenes contenidas en el TFRecord
    for (x, image) in enumerate(images):
        key = None
        # Se procesa la imagen con la red y se obtienen resultados
        (boxes, classes, scores) = detect_fn.execute(image)

        # Se consideran solo los resultados que superan un threshold determinado
        if len([s for s in scores if s >= SCORE_THRESHOLD]) > 0:
            # Se marcan los objetos detectados
            image_np_with_detections = detect_fn.showResults(image, boxes, classes, scores, min_score_thresh=SCORE_THRESHOLD)
            # Se muestra la imagen con los objetos señalados
            cv2.imshow('image',cv2.resize(image_np_with_detections, DIMS_IMG))
            key = keyPushed(time=0)
        else:
            # Se muestra la imagen aunque no se haya detectado nada
            cv2.imshow('image', cv2.resize(image, DIMS_IMG))
            key = keyPushed(time=1)
        if key == ESC:
            break
else:
    # Se itera sobre los TFRecords
    for (i, tfrecord) in enumerate(os.listdir(PATH_IMAGES)):
        tfrecord = os.path.join(PATH_IMAGES, tfrecord)
        (images, categories, boxes) = extractTFRecordImages(tfrecord)
        key = None
        # Se extrae el resultado de procesar cada imagen
        for (x, image) in enumerate(images):
            # Se procesa la imagen con la red
            (boxes, classes, scores) = detect_fn.execute(image)

            # Se consideran solo los resultados que superan un threshold determinado
            if len([s for s in scores if s >= SCORE_THRESHOLD]) > 0:
                # Se marcan los objetos detectados
                image_np_with_detections = detect_fn.showResults(image, boxes, classes, scores, min_score_thresh=SCORE_THRESHOLD)
                # Se muestra la imagen con los objetos señalados
                cv2.imshow('image',cv2.resize(image_np_with_detections, DIMS_IMG))
                key = keyPushed(time=0)
            else:
                # Se muestra la imagen aunque no se haya detectado nada
                cv2.imshow('image',cv2.resize(image, DIMS_IMG))
                key = keyPushed(time=1)
            # El programa finaliza al pulsar esc
            if key == ESC:
                break
        if key == ESC:
            break

cv2.destroyAllWindows()
