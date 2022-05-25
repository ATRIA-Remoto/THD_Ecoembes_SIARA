import tensorflow as tf
import cv2, os
import numpy as np
import progressbar

# COmprueba una ruta especicada y si no existe la crea.
def checkCreateFilePath(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)


ONLY_LABELED = False
SAVE_IMAGES = "D:\Proyectos\THD_Ecoembes\Imagenes/"

SIZE = [640, 640]

# Create a dictionary describing the features.
image_feature_description = {
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string)
}
DIR = 'D:\Proyectos\THD_Ecoembes\Labels\Records/New'
# DIR='A:/00_ATRIA/11_Proyectos Financiados/02_Proyectos Presentados/2019_03_THD_Ecoembes_SIARA/04_PROYECTO/05_Documentos de trabajo/03_Imágenes\Records'
files = [
    "20200908_1.tfrecord"
]

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
        save_dir.append(os.path.join(SAVE_IMAGES, name))
for save in save_dir:
    checkCreateFilePath(save)
checkCreateFilePath(SAVE_IMAGES)
files = filesPath
filesTotal = len(files)

for (i, tfrecord) in enumerate(files):
    name = ''
    extension = ''
    try:
        (name, extension) = tfrecord.split('.')
    except Exception as e:
        pass
    if extension == 'tfrecord':
        tfrecord = os.path.join(DIR, tfrecord)
        record_iterator = tf.compat.v1.python_io.tf_record_iterator(tfrecord)
        k = 0
        for string_record in record_iterator:
            k = k + 1
        record_iterator = tf.compat.v1.python_io.tf_record_iterator(tfrecord)
        widgets = ["Resizing images (file " + str(i+1) + "/" + str(filesTotal) + "): ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=k, widgets=widgets).start()
        f = None
        for (j, string_record) in enumerate(record_iterator):
            example = tf.train.Example()
            example.ParseFromString(string_record)
            save_image = True
            if ONLY_LABELED:
                label = example.features.feature["image/object/class/text"].bytes_list.value
                if len(label) == 0:
                    save_image = False

            if save_image:
                filename = example.features.feature["image/filename"].bytes_list.value[0].decode('utf-8')
                image = example.features.feature["image/encoded"].bytes_list.value[0]
                decoded = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
                # cv2.imwrite(os.path.join(SAVE_IMAGES, filename), decoded)
                cv2.imwrite(os.path.join(save_dir[i], filename), decoded)
            pbar.update(j)
        pbar.finish()
