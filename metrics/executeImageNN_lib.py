import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from tensorflow import keras
import cv2
from utils import label_map_util
import numpy as np
from preprocessors_lib import *
import darknet

# Libreria para carga y ejecucion de redes ya entrenadas
# Se pretende que esta libreria sirva para toda clase de redes (YOLO, tensorflow, Keras...)
# Los nombres de las funciones de las clases debe ser el mismo para cada tipo de red 
# (Por ejemplo el metodo de ejecucion debe llamarse siempre "execute" 
# y debe recibir por parametro solamente la imagen sobre la que apicar la red) de esta forma sera mas facil
# modificar otros codigos.
# Por la misma razon se debe intentar que todas las funciones reciban los mismo parametros y devuelvan los mismos resultados
# y en el mismo orden, en el caso en que sea posible (y si no lo es hacerlo lo mas parecido que se pueda)

# Clase para utilizar una red de YOLO
class YOLONet:

    # Se carga la red y las funciones para preprocesar imagenes, recibe la configuracion de la red y sus pesos como parametros
    def __init__(self, configPath, weightsPath, dataPath, preprocesors=[], confidence=0.5):
        self.configPath = configPath
        self.weightsPath = weightsPath
        self.dataPath = dataPath
        self.network, self.labels, self.class_colors = darknet.load_network(
            self.configPath,
            self.dataPath,
            self.weightsPath,
            batch_size=1
        )
        
        self.preprocesors = preprocesors
        self.confidence = confidence
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)    

        self.darknet_image = darknet.make_image(self.width, self.height, 3)
        self.thresh = confidence
    
        # self.labels = []
        # self.category_index = None
        # if namesPath is not None:
        #     f = open(namesPath, "r")
        #     for line in f.readlines():
        #         self.labels.append(line.replace("\n",""))
            
        #     self.category_index = {}
        #     for i, label in enumerate(self.labels):
        #         self.category_index[i] = {"id":i, "name":label}
        # else:
        #     self.labels = None

    # Funcion de ejecucion de la red, recibe una imagen como parametro y devuelve las boxes detectadas con sus clases y scores
    def execute(self, img):
        for p in self.preprocesors:
            img = p.preprocess(img)

        

        darknet.copy_image_from_bytes(self.darknet_image, img.tobytes())
        detections = darknet.detect_image(self.network, self.labels, self.darknet_image, thresh=self.thresh)

        return detections

    def showResults(self, img, detections):
        
        img = darknet.draw_boxes(detections, img, self.class_colors)

        return img
# Clase para ejecutar uan red generada con Tensorflow Object Detection API
class TFODNeT:

    # Se carga la red y las etiquetas que ha aprendido asi como las funciones de preprocesado de las imagenes,
    # recibe como parametros la red y las etiquetas, asi como las funciones de preprocesado
    def __init__(self, modelPath, labels, preprocesors=[]):
        # Load a (frozen) Tensorflow model into memory.
        self.modelPath = modelPath
        self.labelsPath = labels
        self.preprocesors = preprocesors

        self.detector = tf.saved_model.load(self.modelPath)

        self.category_index = label_map_util.create_category_index_from_labelmap(self.labelsPath, use_display_name=True)

        self.labels_names = []
        for idx in self.category_index:
            self.labels_names.append(self.category_index[idx]['name'])

    # Ejecuta la red con una imagen recibida mediante parametro (tras aplicarle las funciones de preprocesado)
    # Devuelve las bounding boxes y las clases detectadas asi como las scores
    def execute(self, img):
        for preprocessor in self.preprocesors:
            img = preprocessor.preprocess(img)
        input_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)
        # input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0))
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.detector(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        boxes = detections['detection_boxes']
        classes = detections['detection_classes']
        scores = detections['detection_scores']

        return (boxes, classes, scores)

    def showResults(self, img, boxes, classes, scores):
        
        image_np_with_detections = img.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            boxes,
            classes,
            scores,
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=0.7,
              agnostic_mode=False)

        return image_np_with_detections

# Clase para utilizar una red de clasificacion con Keras
class KerasNet:
    
    # Se carga la red y sus funciones de preprocesado, las cuales son recibidas por parametro
    def __init__(self, file, preprocesors=[]):
        self.fileNN = file
        self.NN = keras.models.load_model(file)
        self.preprocesors = preprocesors

    # Se ejecuta la red ocn una imagen recibida por parametro y se devuelve el resultado mas probable
    def execute(self, img):
        for p in self.preprocesors:
            img = p.preprocess(img)

        data = img.astype("float") / 255.0
        prediction = self.NN.predict(np.expand_dims(data, axis=0))[0]
        return prediction
        

class ObjDetector:

    def __init__(self, modelFile, labels, max_rectangles=5, threshold=0.9, preprocessors=[]):        
        self.fileNN = modelFile
        self.MAX_RECTANGLES = max_rectangles
        self.LABELS = labels
        self.threshold = threshold
        self.preprocessors = preprocessors
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.MODEL = KerasNet(self.fileNN, preprocesors=self.preprocessors)

    # Dibuja un rectangulo en los puntos indicados.
    def pintarRectangulo(self, frame, bbox, color, tag=None, grosor=2):
        (xtl, ytl) = (int(bbox[0]), int(bbox[1]))
        (xbr, ybr) = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), color, grosor)
        if tag is not None:
            cv2.putText(frame, tag, (xtl, ytl + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, grosor)
        return frame


    def selectiveSearch(self, image):
        self.ss.setBaseImage(image)
        self.ss.switchToSelectiveSearchFast()
        rects = self.ss.process()

        proposedRects = []

        for (x,y,w,h) in rects:
            proposedRects.append((x, y, x + w, y + h))
        return proposedRects


    def detect(self, image):
        rects = self.selectiveSearch(image)
        predictions = []
        for rect in rects:
            if len(predictions) >= self.MAX_RECTANGLES:
                break
            else:
                (xtl, ytl, xbr, ybr) = rect
                roi = image[ytl:ybr, xtl:xbr]
                if np.shape(roi)[0] > np.shape(image)[0]/10 and np.shape(roi)[1] > np.shape(image)[1]/10:
                    probs = self.MODEL.execute(roi)
                    index = list(probs).index(max(probs))
                    if probs[index] >= self.threshold:
                        pred = self.LABELS[index]
                        predictions.append((pred, probs[index], rect))
        for result in predictions:
            (pred, prob, rect) = result
            image = self.pintarRectangulo(image, rect, (0, 255, 0), tag=pred)

        return (predictions, image)