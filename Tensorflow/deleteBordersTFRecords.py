# Recorta los bordes de las imagenes de THD

import cv2, os
import numpy as np
from pushKey import *
from preprocessors_lib import RoiPreprocessor
import tensorflow as tf
import progressbar

# Diccionario para los atributos de TFRecord
RECORD_VALUES = {
    "image":    'image/encoded',
    "xmin":     'image/object/bbox/xmin',
    "ymin":     'image/object/bbox/ymin',
    "xmax":     'image/object/bbox/xmax',
    "ymax":     'image/object/bbox/ymax',
    "category": 'image/object/class/text',
    "labels":   'image/object/class/label',
    "height":   'image/height',
    "width":    'image/width',
    "source_id":'image/source_id',
    "format":   'image/format',
    "filename": 'image/filename',
    "depth":    'image/depth',
}

# Recorte a realizar sobre todas las imagenes
recorte = {
    'y1': 0,
    'y2': 1024,
    'x1': 150,
    'x2': 1050
}

# Preprocesamiento con el recorte de la imagen
ROI = RoiPreprocessor(recorte.get('y1'), recorte.get('y2'), recorte.get('x1'), recorte.get('x2'))

# Comprueba una ruta especicada y si no existe la crea.
def checkCreateFilePath(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)

# Funcion que recorta los bordes de la imagen
def deleteBorders(img, bboxes, labels, categories):
    imgROI = ROI.preprocess(img)
    newBboxes = []
    newLabels = []
    newCategories = []
    for i, bbox in enumerate(bboxes):
        isValidBbox = True
        # Se adapta a la ROI
        (x, y, xbr, ybr) = bbox
        w = xbr - x
        h = ybr - y
        if x > recorte.get('x2'):
            isValidBbox = False
        else:
            x = x - recorte.get('x1')
            if x < 0.0:
                w = w - (-x)
                x = 0.0
            
            if (x+w) > np.shape(imgROI)[1]:
                if x < np.shape(imgROI)[1]:
                    w = np.shape(imgROI)[1] - x
                else:
                    isValidBbox = False
        if isValidBbox:
            newBboxes.append([x, y, x+w, y+h])
            newLabels.append(labels[i])
            newCategories.append(categories[i])
    return imgROI, newBboxes, newLabels, newCategories

# Extrae y devuelve los atributos de un fichero TFRecord
def extractTFRecordData(tfrecord):
    images = []
    boxes = []
    all_categories = []
    labels = []
    heights = []
    widths = []
    source_ids = []
    formats = []
    filenames = []
    depths = []
    record_iterator = tf.compat.v1.python_io.tf_record_iterator(tfrecord)
    k = 0
    for string_record in record_iterator:
        k = k + 1
    record_iterator = tf.compat.v1.python_io.tf_record_iterator(tfrecord)
    f = None
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        categories = example.features.feature[RECORD_VALUES['category']].bytes_list.value
        if len(categories) > 0:
            box = []
            for (k, xmin) in enumerate(example.features.feature[RECORD_VALUES['xmin']].float_list.value):
                box.append([xmin, example.features.feature[RECORD_VALUES['ymin']].float_list.value[k], 
                    example.features.feature[RECORD_VALUES['xmax']].float_list.value[k],
                    example.features.feature[RECORD_VALUES['ymax']].float_list.value[k]])

            all_categories.append(categories)

            image = example.features.feature[RECORD_VALUES['image']].bytes_list.value[0]
            images.append(image)
            boxes.append(box)
            labels.append(example.features.feature[RECORD_VALUES["labels"]].int64_list.value)
            heights.append(example.features.feature[RECORD_VALUES["height"]].int64_list.value[0])
            widths.append(example.features.feature[RECORD_VALUES["width"]].int64_list.value[0])
            source_ids.append(example.features.feature[RECORD_VALUES["source_id"]].bytes_list.value[0])
            formats.append(example.features.feature[RECORD_VALUES["format"]].bytes_list.value[0])
            filenames.append(example.features.feature[RECORD_VALUES["filename"]].bytes_list.value[0])
            depths.append(example.features.feature[RECORD_VALUES["depth"]].int64_list.value[0])

    return images, all_categories, boxes, labels, heights, widths, depths, source_ids, formats, filenames

# Genera una nueva imagen con los margenes recortados
def generateModifiedTFRecord(pathTF):
    tf_examples = []
    # Se extrae la informacion del antiguo tfrecord
    (images, all_categories, boxes, labels, heights, widths, depths, source_ids, formats, filenames) = extractTFRecordData(pathTF)
    
    # Se obtienen las etiquetas de cada una de las imagenes
    for (i, image) in enumerate(images):
        imageDecoded = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        xtls = []
        ytls = []
        xbrs = []
        ybrs = []
        boxesReal = []
        for box in boxes[i]:
            (xtl, ytl, xbr, ybr) = box
            xtl = xtl*widths[i]
            ytl = ytl*heights[i]
            xbr = xbr*widths[i]
            ybr = ybr*heights[i]
            boxesReal.append([xtl, ytl, xbr, ybr])
            # Se eliminan los bordes de la imagen
        (newImage, newBoxesReal, newLabels, newCategories) = deleteBorders(imageDecoded, boxesReal, labels[i], all_categories[i])
        newHeight, newWidth, newDepth = np.shape(newImage)
        # Se adaptan las coordenadas al nuevo tamaño de la imagen
        for box in newBoxesReal:
            (xtl, ytl, xbr, ybr) = box
            xtl = xtl/newWidth
            ytl = ytl/newHeight
            xbr = xbr/newWidth
            ybr = ybr/newHeight
            xtls.append(xtl)
            ytls.append(ytl)
            xbrs.append(xbr)
            ybrs.append(ybr)
        # Se guardan los datos en el nuevo archivo
        cv2.imwrite('./temp.'+formats[i].decode('utf-8'), newImage)
        image_string = open('./temp.'+formats[i].decode('utf-8'), 'rb').read()
        tf_example = image_example(source_ids[i], image_string, newLabels, xtls, xbrs, ytls, ybrs, filenames[i], newCategories, formats[i])
        tf_examples.append(tf_example)

    record_file = os.path.join(NEW_TF_DIR, os.path.basename(pathTF))
    with tf.io.TFRecordWriter(record_file) as writer:
        for tf_example in tf_examples:
            writer.write(tf_example.SerializeToString())

# Funciones para transformar y decodificar los atributos del TFRecord
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_list_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# Create a dictionary with features that may be relevant.
def image_example(source_id, image_string, labels, xmins, xmaxs, ymins, ymaxs, filename, labelText, form):
    image_shape = tf.image.decode_jpeg(image_string).shape  
    feature = {
        'image/source_id': _bytes_feature(source_id),
        'image/height': _int64_feature([image_shape[0]]),
        'image/width': _int64_feature([image_shape[1]]),
        'image/depth': _int64_feature([image_shape[2]]),
        'image/object/class/label': _int64_feature(labels),
        "image/object/class/text": _bytes_list_feature(labelText),
        'image/object/bbox/xmin': _float_feature(xmins),     
        'image/object/bbox/xmax': _float_feature(xmaxs),
        'image/object/bbox/ymin': _float_feature(ymins),
        'image/object/bbox/ymax': _float_feature(ymaxs),
        'image/filename': _bytes_feature(filename),
        'image/format': _bytes_feature(form),
        'image/encoded': _bytes_feature(image_string)
    } 
    return tf.train.Example(features=tf.train.Features(feature=feature))  

# Ruta a fichero(s) tfrecords
TF_DIR = "D:/Proyectos/THD_Ecoembes/Labels/TF_ALT/3classesV/All"
# Ruta de los nuevos ficheros tfrecord generados
NEW_TF_DIR = "D:/Proyectos/THD_Ecoembes/Labels/TF_ALT/3classesV/prueba2"

# Se crea el directorio de los ficheros nuevos si no existe
checkCreateFilePath(NEW_TF_DIR) 

# Se comprueba si se trabsforma un unico fichero o un directorio con uno o mas
is_file = False
if os.path.isfile(TF_DIR):
    is_file = True

if is_file: # Si solo es un fichero
    # Barra de progreso
    widgets = ["Creating new tfrecords files: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=1, widgets=widgets).start()
    generateModifiedTFRecord(TF_DIR)    # Funcion que modifica los datos y los guarda
    pbar.update(1)
    pbar.finish()
else:   # Si es un directorio
    # Se listan los ficheros
    listFiles = os.listdir(TF_DIR)
    widgets = ["Creating new tfrecords files: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(listFiles), widgets=widgets).start()
    # Para cada fichero, se obtiene su ruta completa y se modifican los datos
    for (j, record) in enumerate(listFiles):
        recordPath = os.path.join(TF_DIR, record)
        generateModifiedTFRecord(recordPath)
        pbar.update(j)
    pbar.finish()
