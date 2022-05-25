from executeImageNN_lib import TFODNeT
from metricsFunctions import intersectionOverUnion, saveConfusionMatrix, calcLRP
import numpy as np
import progressbar
from TFRecord import TFRecord
from evaluationOD import *
import os, time

# Rutas a las carpetas con tfrecords, mdoelo y etiquetas
# TFRECORDS_PATH = "D:/Proyectos/THD_Ecoembes/Labels/TF_ALT/3classesV/NoBorders/Test"
TFRECORDS_PATH = "A:/11_Proyectos Financiados/02_Proyectos Presentados/01_Aceptados/07_2019_03_THD_Ecoembes_SIARA/04_PROYECTO/05_Documentos de trabajo/03_Imágenes/Records/3_classes/NoBorders/Test"
MODEL_PATH = "D:\Proyectos\THD_Ecoembes\Modelos\Mobilenet\exported_model\saved_model"
LABEL_MAP_PATH = "D:/Proyectos/THD_Ecoembes/Labels/TF_ALT/3classesV/label_map.pbtxt"

# Threshold para considerar imagenes como validas al evaluar
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# Se carga el modelo de Tensorflow
print("[INFO] Loading model")
tfDetector = TFODNeT(MODEL_PATH, LABEL_MAP_PATH)
print("[INFO] Model loaded")

# Se comprueba si se trata una ruta a un fichero
is_file = False
if len(TFRECORDS_PATH.split('.')) > 1  and TFRECORDS_PATH.split('.')[1].lower() == 'tfrecord':
    is_file = True

# Se preparan las variables que se utilizaran para contener los resultados
dataResults = {}

confusionMatrix = []
for c in tfDetector.category_index:
    confusionMatrix.append([])
    for c_pred in tfDetector.category_index:
        confusionMatrix[c-1].append(0)

confusionMatrix = np.array(confusionMatrix)

for c in tfDetector.category_index:
    dataResults[tfDetector.category_index[c]['name']] = {}
    dataResults[tfDetector.category_index[c]['name']]['predictions'] = []
    dataResults[tfDetector.category_index[c]['name']]['scores'] = []
    dataResults[tfDetector.category_index[c]['name']]['ious'] = []
    dataResults[tfDetector.category_index[c]['name']]['pred_classes'] = []

total_gt_categories = []

meanTimeDetection = 0.0
numImages = 0

# Se trata los tfrecords segun se trate de un directorio con varios archivos o solo un archivo
if is_file:
    # Se extraen los datos del tfrecord
    recordData = TFRecord(TFRECORDS_PATH)
    (images, all_categories, qt_boxes, labels, heights, widths, depths, source_ids, formats, filenames) = recordData.getData()

    # Se obtienen los resultados y se añaden a las variables previamente creadas
    detections, detectionTimeTotal = processImages(tfDetector, images)

    if len(detections) > 0:
    
        (pred_boxes_record, pred_classes_record, pred_scores_record) = detections
        if len(detections) > 0:

            all_gt_boxes = []
            labels = []
            all_pred_boxes = []
            labelsPredicted = []
            all_scores = []

            classes = tfDetector.labels_names

            dimensions = {
                "width": np.shape(images[0])[1],
                "height": np.shape(images[0])[0]
            }

            for detection in detections:
                (pred_boxes_img, pred_classes_img, pred_scores_img) = detection
                pred_classes_img_str = []
                for i, pred_class in enumerate(pred_classes_img):
                    pred_classes_img_str.append(classes[pred_class])
                all_pred_boxes.append(pred_boxes_img)
                labelsPredicted.append(pred_classes_img_str)
                all_scores.append(pred_scores_img)
            (dataResults, confusionMatrix) = extractResults(qt_boxes, labels, all_pred_boxes, labelsPredicted, all_scores, tfDetector.labels_names, dimensions)
            

    # for categories in all_categories:
    #     total_gt_categories.extend(categories)
    numImages = len(images)
    # meanTimeDetection = detectionTimeTotal / numImages
else:
    # Se listan los ficheros del directorio
    listFiles = os.listdir(TFRECORDS_PATH)
    detectionTimeTotal = 0.0
    widgets = ["Evaluando los datos de los tfrecords: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(listFiles), widgets=widgets).start()
    # Para cada fichero se calculan las metricas
    for (record_idx, record) in enumerate(listFiles):
        recordPath = os.path.join(TFRECORDS_PATH, record)
        recordData = TFRecord(recordPath)
        # Se extraen los datos
        (images, all_categories, qt_boxes, labels, heights, widths, depths, source_ids, formats, filenames) = recordData.getData()
        numImages = numImages + len(images)
        for categories in all_categories:
            total_gt_categories.extend(categories)
        # Se obtienen las metricas y se añaden a los datos generales
        detections, detectionTimeTF = processImages(tfDetector, images, bar=False)

        if len(detections) > 0:

            all_gt_boxes = []
            labels_str = []
            all_pred_boxes = []
            labelsPredicted = []
            all_scores = []

            classes = tfDetector.labels_names

            dimensions = {
                "width": np.shape(images[0])[1],
                "height": np.shape(images[0])[0]
            }

            for i, detection in enumerate(detections):
                (pred_boxes_img, pred_classes_img, pred_scores_img) = detection

                pred_classes_img_str = []
                for pred_class in pred_classes_img:
                    pred_classes_img_str.append(classes[pred_class-1])
                labelsPredicted.append(pred_classes_img_str)

                categories_aux = []
                categories = all_categories[i]
                for categorie in categories:
                    categories_aux.append(categorie.decode())
                labels_str.append(categories_aux)

                pred_boxes_img_aux = []
                for pred_box in pred_boxes_img:
                    (ymin, xmin, ymax, xmax) = pred_box
                    pred_boxes_img_aux.append([xmin, ymin, xmax, ymax])
                all_pred_boxes.append(pred_boxes_img_aux)
                all_scores.append(pred_scores_img)
            (dataResults_record, confusionMatrixRecord) = extractResults(qt_boxes, labels_str, all_pred_boxes, labelsPredicted, all_scores, tfDetector.labels_names, dimensions)
            
            for c in tfDetector.category_index:
                dataResults[tfDetector.category_index[c]['name']]['predictions'].extend(dataResults_record[tfDetector.category_index[c]['name']]['predictions'])
                dataResults[tfDetector.category_index[c]['name']]['scores'].extend(dataResults_record[tfDetector.category_index[c]['name']]['scores'])
                dataResults[tfDetector.category_index[c]['name']]['ious'].extend(dataResults_record[tfDetector.category_index[c]['name']]['ious'])
                dataResults[tfDetector.category_index[c]['name']]['pred_classes'].extend(dataResults_record[tfDetector.category_index[c]['name']]['pred_classes'])
            
            confusionMatrix = confusionMatrix + np.array(confusionMatrixRecord)

        detectionTimeTotal = detectionTimeTotal + detectionTimeTF

        pbar.update(record_idx)
    pbar.finish()
    detectionTimeTotal = detectionTimeTotal / len(listFiles)
    

# Se muestra informacion general
print("Imagenes evaluadas: " + str(numImages))
print("Total Etiquetados:")
for category in tfDetector.labels_names:
    print(category + ": " + str(len([True for label in total_gt_categories if label.decode()==category])))
print()

# Se calculan y muestran las metricas extraidas
print("Tiempo medio deteccion de objetos: " + str(detectionTimeTotal) + " segundos")

FN_all = 0
for l in tfDetector.labels_names:
    TP = len([i for i in dataResults[l]['predictions'] if i == "tp"])
    FP = len([i for i in dataResults[l]['predictions'] if i == "fp"])
    FN = len([i for i in dataResults[l]['predictions'] if i == "fn"])
    print(l + ": [TP: " + str(TP) + ", FP: " + str(FP) + "]")
    FN_all = FN_all + FN
print("FN: " + str(FN_all))


# Se terminan de calcular las difewrentes metricas con los resultados obtenidos
(precisions_inter, precisions, recalls) = calculatePrecisionsRecallsOD(dataResults)
ap = calculateAveragePrecision(precisions_inter, recalls)
mAp = 0.0
for category in ap:
    mAp += ap[category]
mAp = mAp / len(ap)

for category in ap:
    print("AP@" + str(category) + " = " + str(ap[category]))
print("mAP = " + str(mAp))

# Se calcula y muestra la metrica LRP
lrp = {}
for c in tfDetector.category_index:
    dataResults_class = dataResults[tfDetector.category_index[c]['name']]
    nTP = len([prediction for i, prediction in enumerate(dataResults_class['predictions']) if prediction == 'tp' and dataResults_class["ious"][i] is not None and dataResults_class["ious"][i] > IOU_THRESHOLD and dataResults_class["scores"][i] is not None and dataResults_class["scores"][i] > SCORE_THRESHOLD])
    nFP = len([prediction for i, prediction in enumerate(dataResults_class['predictions']) if prediction == 'fp' and dataResults_class["ious"][i] is not None and dataResults_class["ious"][i] > IOU_THRESHOLD and dataResults_class["scores"][i] is not None and dataResults_class["scores"][i] > SCORE_THRESHOLD])
    nFN = len([prediction for i, prediction in enumerate(dataResults_class['predictions']) if prediction == 'fn' and dataResults_class["ious"][i] is not None and dataResults_class["ious"][i] > IOU_THRESHOLD and dataResults_class["scores"][i] is not None and dataResults_class["scores"][i] > SCORE_THRESHOLD])
    iousTP = [iou for i, iou in enumerate(dataResults_class["ious"]) if dataResults_class['predictions'][i] == 'tp' and iou is not None and iou > IOU_THRESHOLD and dataResults_class["scores"][i] is not None and dataResults_class["scores"][i] > SCORE_THRESHOLD]
    lrp[tfDetector.category_index[c]['name']] = calcLRP(nTP, nFP, nFN, iousTP, IOU_THRESHOLD)

mLRP = 0.0
for category in lrp:
    print("LRP@" + str(category) + " = " + str(lrp[category]))
    mLRP += lrp[category]
mLRP = mLRP / len(lrp)

print("mLRP = " + str(mLRP))



for j, c in enumerate(tfDetector.category_index):
    dataResults_class = dataResults[tfDetector.category_index[c]['name']]
    for i, pred in enumerate(dataResults_class["pred_classes"]):
        if dataResults_class["scores"][i] > SCORE_THRESHOLD and (dataResults_class["ious"][i] is not None and dataResults_class["ious"][i] > IOU_THRESHOLD):
            confusionMatrix[j][tfDetector.labels_names.index(pred)] += 1

saveConfusionMatrix(np.array(confusionMatrix), tfDetector.labels_names)