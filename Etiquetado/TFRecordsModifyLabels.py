import tensorflow as tf
import cv2, os, time
import numpy as np
import progressbar

# COmprueba una ruta especicada y si no existe la crea.
def checkCreateFilePath(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)


ONLY_LABELED = True
LABELS_TO_EXTRACT = {
  'barquilla': 1,
  'carton': 2,
  'film': 3
}

SAVE_DIR = "A:/00_ATRIA/11_Proyectos Financiados/02_Proyectos Presentados/01_Aceptados/07_2019_03_THD_Ecoembes_SIARA/04_PROYECTO/05_Documentos de trabajo/03_Imágenes\Records/3_classes"


DIR = 'D:\Proyectos\THD_Ecoembes\Labels\Records/New'
DIR = 'A:/00_ATRIA/11_Proyectos Financiados/02_Proyectos Presentados/01_Aceptados/07_2019_03_THD_Ecoembes_SIARA/04_PROYECTO/05_Documentos de trabajo/03_Imágenes\Records/Complete'
files = [
    "20200817_1.tfrecord"
]
files = os.listdir(DIR)

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

# Create a dictionary describing the features.
image_feature_description = {
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string)
}


# for save in save_dir:
#     checkCreateFilePath(save)
checkCreateFilePath(SAVE_DIR)
filesTotal = len(files)

for (i, tfrecord) in enumerate(files):
    fileOrigin = tfrecord
    name = ''
    extension = ''
    try:
        (name, extension) = tfrecord.split('.')
    except Exception as e:
        pass
    # fTF=open("D:/Proyectos/THD_Ecoembes/codigo/THD/patata.txt", "w")
    if extension == 'tfrecord': 
        tfrecord = os.path.join(DIR, tfrecord)
        record_iterator = tf.compat.v1.python_io.tf_record_iterator(tfrecord)
        k = 0
        for string_record in record_iterator:
            k = k + 1
        record_iterator = tf.compat.v1.python_io.tf_record_iterator(tfrecord)
        widgets = ["Creating new tfrecords (file " + str(i+1) + "/" + str(filesTotal) + "): ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=k, widgets=widgets).start()
        f = None
        tf_examples = []
        for (j, string_record) in enumerate(record_iterator):
            example = tf.train.Example()
            example.ParseFromString(string_record)
            save_annotation = True
            label = example.features.feature["image/object/class/text"].bytes_list.value
            if ONLY_LABELED:
                if len(label) == 0:
                    save_annotation = False

            if len(LABELS_TO_EXTRACT) > 0:
              extract = False
              for l in label:
                if l.decode('utf-8') in LABELS_TO_EXTRACT:
                  extract = True
                  break
              if not extract:
                save_annotation = False

            if save_annotation:
                filename = example.features.feature["image/filename"].bytes_list.value[0]
                form = example.features.feature['image/format'].bytes_list.value[0]
                image = example.features.feature["image/encoded"].bytes_list.value[0]
                source_id = example.features.feature["image/source_id"].bytes_list.value[0]

                labels = []
                labelTexts = []
                xmins = []
                xmaxs = []
                ymins = []
                ymaxs = []
                decoded = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
                label_id = 0
                for (k, value) in enumerate(example.features.feature["image/object/class/label"].int64_list.value):
                  classText = example.features.feature["image/object/class/text"].bytes_list.value[k].decode('utf-8')
                  if len(LABELS_TO_EXTRACT) == 0 or (len(LABELS_TO_EXTRACT) > 0 and classText in LABELS_TO_EXTRACT):
                    label_id = value
                    if len(LABELS_TO_EXTRACT) > 0:
                      label_id = LABELS_TO_EXTRACT[classText]
                    labels.append(label_id)
                    labelTexts.append(example.features.feature["image/object/class/text"].bytes_list.value[k])
                    xmins.append(example.features.feature["image/object/bbox/xmin"].float_list.value[k])
                    ymins.append(example.features.feature["image/object/bbox/ymin"].float_list.value[k])
                    xmaxs.append(example.features.feature["image/object/bbox/xmax"].float_list.value[k])
                    ymaxs.append(example.features.feature["image/object/bbox/ymax"].float_list.value[k])
                    
                cv2.imwrite('./temp.jpg', decoded)
                image_string = open('./temp.jpg', 'rb').read()
                tf_example = image_example(source_id, image_string, labels, xmins, xmaxs, ymins, ymaxs, filename, labelTexts, form)
                tf_examples.append(tf_example)
                # for line in str(tf_example).split('\n'):
                #     print(line, file=fTF)
            pbar.update(j)
        pbar.finish()
        
        record_file = os.path.join(SAVE_DIR,fileOrigin)
        with tf.io.TFRecordWriter(record_file) as writer:
            for tf_example in tf_examples:
                writer.write(tf_example.SerializeToString())
