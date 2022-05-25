import tensorflow as tf
import cv2
import numpy as np

# Clase de gestion de ficherps tfrecord
class TFRecord:
    # Inicializacion de tfrecords con sus atributos declarados
    def __init__(self, path=None):
        # Ruta del fichero
        self.file = path

        # Atributos y caracteristicas
        self.RECORD_VALUES = {
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

        # Se lee y extraen los atributos del fichero
        if self.file is not None:
            (self.images, self.all_categories, self.boxes, self.labels, self.heights, 
                self.widths, self.depths, self.source_ids, self.formats, self.filenames) = self.extractTFRecordData(self.file)
        else:   # Si no se ha declarado el fichero, solamente se inicializan los atributos
            self.images = None
            self.all_categories = None
            self.boxes = None
            self.labels = None
            self.heights = None
            self.widths = None
            self.depths = None
            self.source_ids = None
            self.formats = None
            self.filenames = None


    # Funciones para la transformacion de los atributos leidos
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _bytes_list_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    # Crea un diiccionario de los atributos contenidos en el fichero
    def image_example(self, source_id, image_string, labels, xmins, xmaxs, ymins, ymaxs, filename, labelText, form):
        image_shape = tf.image.decode_jpeg(image_string).shape  
        feature = {
            'image/source_id': self._bytes_feature(source_id),
            'image/height': self._int64_feature([image_shape[0]]),
            'image/width': self._int64_feature([image_shape[1]]),
            'image/depth': self._int64_feature([image_shape[2]]),
            'image/object/class/label': self._int64_feature(labels),
            "image/object/class/text": self._bytes_list_feature(labelText),
            'image/object/bbox/xmin': self._float_feature(xmins),     
            'image/object/bbox/xmax': self._float_feature(xmaxs),
            'image/object/bbox/ymin': self._float_feature(ymins),
            'image/object/bbox/ymax': self._float_feature(ymaxs),
            'image/filename': self._bytes_feature(filename),
            'image/format': self._bytes_feature(form),
            'image/encoded': self._bytes_feature(image_string)
        } 
        return tf.train.Example(features=tf.train.Features(feature=feature))  

    # Itera sobre los datos del fichero, y extrae los atributos principales
    def extractTFRecordData(self, tfrecord):
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

            categories = example.features.feature[self.RECORD_VALUES['category']].bytes_list.value
            if len(categories) > 0:
                box = []
                for (k, xmin) in enumerate(example.features.feature[self.RECORD_VALUES['xmin']].float_list.value):
                    box.append([xmin, example.features.feature[self.RECORD_VALUES['ymin']].float_list.value[k], 
                        example.features.feature[self.RECORD_VALUES['xmax']].float_list.value[k],
                        example.features.feature[self.RECORD_VALUES['ymax']].float_list.value[k]])

                all_categories.append(categories)

                image = example.features.feature[self.RECORD_VALUES['image']].bytes_list.value[0]
                images.append(cv2.imdecode(np.frombuffer(image, np.uint8), -1))
                boxes.append(box)
                labels.append(example.features.feature[self.RECORD_VALUES["labels"]].int64_list.value)
                heights.append(example.features.feature[self.RECORD_VALUES["height"]].int64_list.value[0])
                widths.append(example.features.feature[self.RECORD_VALUES["width"]].int64_list.value[0])
                source_ids.append(example.features.feature[self.RECORD_VALUES["source_id"]].bytes_list.value[0])
                formats.append(example.features.feature[self.RECORD_VALUES["format"]].bytes_list.value[0])
                filenames.append(example.features.feature[self.RECORD_VALUES["filename"]].bytes_list.value[0])
                depths.append(example.features.feature[self.RECORD_VALUES["depth"]].int64_list.value[0])

        return images, all_categories, boxes, labels, heights, widths, depths, source_ids, formats, filenames

    # Devuelve los atributos leidos del fichero
    def getData(self):
        return (self.images, self.all_categories, self.boxes, self.labels, self.heights, 
                self.widths, self.depths, self.source_ids, self.formats, self.filenames)
