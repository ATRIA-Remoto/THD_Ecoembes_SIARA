import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from tensorflow import keras
import cv2
from utils import label_map_util
import numpy as np
from preprocessors_lib import *
# Libreria para carga y ejecucion de redes ya entrenadas
# Se pretende que esta libreria sirva para toda clase de redes (YOLO, tensorflow, Keras...)
# Los nombres de las funciones de las clases debe ser el mismo para cada tipo de red 
# (Por ejemplo el metodo de ejecucion debe llamarse siempre "execute" 
# y debe recibir por parametro solamente la imagen sobre la que apicar la red) de esta forma sera mas facil
# modificar otros codigos.
# Por la misma razon se debe intentar que todas las funciones reciban los mismo parametros y devuelvan los mismos resultados
# y en el mismo orden, en el caso en que sea posible (y si no lo es hacerlo lo mas parecido que se pueda)

# Dibuja un rectangulo en los puntos indicados.
def pintarRectangulo(frame, bbox, color, tag=None, grosor=2):
    (x, y) = (int(bbox[0]), int(bbox[1]))
    (w, h) = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, grosor)
    if tag is not None:
        cv2.putText(frame, tag, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, grosor)
    return frame

# Clase para utilizar una red de YOLO
class YOLONet:

    # Se carga la red y las funciones para preprocesar imagenes, recibe la configuracion de la red y sus pesos como parametros
    def __init__(self, configPath, weightsPath, namesPath=None, preprocesors=[], confidence=0.5):
        self.configPath = configPath
        self.weightsPath = weightsPath
        self.NN = cv2.dnn.readNet(configPath, weightsPath)
        self.preprocesors = preprocesors
        self.confidence = confidence
        self.labels = []
        self.category_index = None
        if namesPath is not None:
            f = open(namesPath, "r")
            for line in f.readlines():
                self.labels.append(line.replace("\\n",""))
            
            self.category_index = {}
            for i, label in enumerate(self.labels):
                self.category_index[i] = {"id":i, "name":label}
        else:
            self.labels = None

    # Funcion de ejecucion de la red, recibe una imagen como parametro y devuelve las boxes detectadas con sus clases y scores
    def execute(self, img):
        for p in self.preprocesors:
            img = p.preprocess(img)
        # determine only the *output* layer names that we need from YOLO

        ln = self.NN.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.NN.getUnconnectedOutLayers()]
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (img.shape[1], img.shape[0]), crop=False)
        self.NN.setInput(blob)
        
        layerOutputs = self.NN.forward(ln)

        (H, W) = img.shape[:2]
        boxes = []
        classIDs = []
        confidences = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    xtl = int(centerX - (width / 2))
                    ytl = int(centerY - (height / 2))
                    xbr = int(centerX + (width / 2))
                    ybr = int(centerY + (height / 2))
                    box = np.array([ytl, xtl, ybr, xbr]) / np.array([H, W, H, W])
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        return (boxes, classIDs, confidences)

    # Muestra sobre una imagen la posicion de un objeto y su clasificacion
    def showResults(self, img, boxes, classes, scores, max_boxes_to_draw=200, min_score_thresh=0.7):
        
        image_np_with_detections = img.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            np.array(boxes),
            np.array(classes),
            np.array(scores),
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_score_thresh,
              agnostic_mode=False)

        return image_np_with_detections

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

    # Muestra sobre una imagen la posicion de un objeto y su clasificacion
    def showResults(self, img, boxes, classes, scores, max_boxes_to_draw=200, min_score_thresh=0.7):
        
        image_np_with_detections = img.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            boxes,
            classes,
            scores,
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_score_thresh,
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
        
# Detector de objetos generico, utiliza como base una red clasificacion estandar, empleando OpenCV para encontrar candidatos a objetos a clasificar
class ObjDetector:

    # Inicializacion de las fuciones y modelo de la red
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

    # Algoritmo de busqueda de candidatos
    def selectiveSearch(self, image):
        self.ss.setBaseImage(image)
        self.ss.switchToSelectiveSearchFast()
        rects = self.ss.process()

        proposedRects = []

        for (x,y,w,h) in rects:
            proposedRects.append((x, y, x + w, y + h))
        return proposedRects

    # Se ejecuta la busqueda de objetos y la clasificacion de la red
    def execute(self, image):
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