import tensorflow as tf
import cv2, os
import numpy as np
import progressbar

# COmprueba una ruta especicada y si no existe la crea.
def checkCreateFilePath(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)


DIR = 'A:/11_Proyectos Financiados/02_Proyectos Presentados/01_Aceptados/07_2019_03_THD_Ecoembes_SIARA/04_PROYECTO/05_Documentos de trabajo/03_Imágenes/Records/3_classes/NoBorders/Test'
ONLY_LABELED = True
SAVE_DIR = "D:/Proyectos/THD_Ecoembes/Labels/YOLO/NoBorders/3_classes/Test3"
ANNOTATIONS_DIR = "All"
ANNOTATIONS_DIR = os.path.join(SAVE_DIR, ANNOTATIONS_DIR)

listFile = "test2.txt"
listFile = os.path.join(SAVE_DIR, listFile)



# Create a dictionary describing the features.
image_feature_description = {
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string)
}


files = os.listdir(DIR)
filesPath=[]
save_dir = []
for f in files:
    name = ''
    extension = ''
    try:
        (name, extension) = f.split('.')
    except Exception as e:
        pass
    if extension == 'tfrecord':
        filesPath.append(os.path.join(DIR, f))
        save_dir.append(os.path.join(SAVE_DIR, name))
# for save in save_dir:
#     checkCreateFilePath(save)
checkCreateFilePath(SAVE_DIR)
checkCreateFilePath(ANNOTATIONS_DIR)
files = filesPath
filesTotal = len(files)

# Se crea el fichero que listara las imagenes y etiquetas
l = open(listFile, "w")
# Se realiza la accion con cada tfrecord del directorio
for (i, tfrecord) in enumerate(files):
    name = ''
    extension = ''
    try:
        (name, extension) = tfrecord.split('.')
    except Exception as e:
        pass
    # Se comprueba que el fichero sea de tipo tfrecord
    if extension == 'tfrecord':
        # Se lee el fichero y se empiezan a tratar sus datos
        tfrecord = os.path.join(DIR, tfrecord)
        record_iterator = tf.compat.v1.python_io.tf_record_iterator(tfrecord)
        k = 0
        for string_record in record_iterator:
            k = k + 1
        record_iterator = tf.compat.v1.python_io.tf_record_iterator(tfrecord)
        widgets = ["Creating YOLO annotations (file " + str(i+1) + "/" + str(filesTotal) + "): ", 
        progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=k, widgets=widgets).start()
        f = None
        # Por cada dato del tfrecord, se traduce al formato YOLO
        for (j, string_record) in enumerate(record_iterator):
            example = tf.train.Example()
            example.ParseFromString(string_record)
            save_annotation = True
            # Se comprueba si hay datos de interes
            if ONLY_LABELED:
                label = example.features.feature["image/object/class/text"].bytes_list.value
                if len(label) == 0:
                    save_annotation = False
            # En caso de que haya datos de interes se guarda la informacion n formato YOLO
            if save_annotation:
                # Se extraen los datos (imagen, nombre del archivo, etiqueta, coordendas...)
                filename = example.features.feature["image/filename"].bytes_list.value[0].decode('utf-8')
                print(os.path.join(ANNOTATIONS_DIR, filename), file=l)
                image = example.features.feature["image/encoded"].bytes_list.value[0]
                annotationName = filename.split('.')[0] + '.txt'
                ann = open(os.path.join(ANNOTATIONS_DIR, annotationName), 'w')
                # Se extraen las coordenadas y se traducen al formato de YOLO
                for (k, value) in enumerate(example.features.feature["image/object/class/label"].int64_list.value):
                    xmin = example.features.feature["image/object/bbox/xmin"].float_list.value[k]
                    ymin = example.features.feature["image/object/bbox/ymin"].float_list.value[k]
                    w = example.features.feature["image/object/bbox/xmax"].float_list.value[k] - xmin
                    h = example.features.feature["image/object/bbox/ymax"].float_list.value[k] - ymin
                    # En YOLO x e y indican el centro de la etiqueta
                    x = xmin + w/2
                    y = ymin + h/2
                    # Se escriben las coordenadas y la etiqueta en un fichero de texto con el mismo nombre que la imagen
                    annotation = str(value-1) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h)
                    print(annotation, file=ann)
                ann.close()                 
                # Se decodifica y guarda la imagen
                decoded = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
                cv2.imwrite(os.path.join(ANNOTATIONS_DIR, filename), decoded)
            pbar.update(j)
        pbar.finish()
l.close()
