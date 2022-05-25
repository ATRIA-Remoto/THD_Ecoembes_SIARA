from executeImageNN_lib import TFODNeT
from metricsFunctions import intersectionOverUnion, saveConfusionMatrix, calcLRP
import numpy as np
import progressbar
from TFRecord import TFRecord
import os, time

# Rutas a las carpetas con tfrecords, mdoelo y etiquetas
TFRECORDS_PATH = "D:/Proyectos/THD_Ecoembes/Labels/TF_ALT/3classesV/NoBorders/Test"
MODEL_PATH = "D:/Proyectos/THD_Ecoembes/codigo/THD/Tensorflow/experiments/faster_resnet50_adam/saved_model"
LABEL_MAP_PATH = "D:/Proyectos/THD_Ecoembes/Labels/TF_ALT/3classesV/label_map.pbtxt"

# Threshold para considerar imagenes como validas al evaluar
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# Funcion principal de procesamiento de la simagenes, carga las imagenes, las pasa por el detector y extrae
# scores, IoU, TP, FP, TN, FN... 
def processImages(tfDetector, images, all_gt_boxes, labels):
    dataResults = {}
    confusionMatrix = []
    detectionTime = 0.0     # Time expended in detection task
    category_index = tfDetector.category_index
    for c in category_index: 
        dataResults[category_index[c]['name']] = {}
        dataResults[category_index[c]['name']]['predictions'] = []
        dataResults[category_index[c]['name']]['scores'] = []
        dataResults[category_index[c]['name']]['ious'] = []
        dataResults[category_index[c]['name']]['pred_classes'] = []
        confusionMatrix.append([])
        for c_pred in tfDetector.category_index:
            confusionMatrix[c-1].append(0)

    for i, img in enumerate(images):
        labels_img = labels[i]
        gt_boxes_img = all_gt_boxes[i]
        tInit = time.time()
        (pred_boxes_img, pred_classes_img, scores_img) = tfDetector.execute(img)
        tEnd = time.time()
        detectionTime = detectionTime + (tEnd - tInit)
        (h, w, d) = np.shape(img)
        
        for c in category_index:
            gt_idx_class = [i for i, x in enumerate(labels_img) if x==category_index[c]['id']]
            # pred_idx_class = [i for i, x in enumerate(pred_classes_img) if x==category_index[c]['id']]
            pred_idx_class = [i for i, x in enumerate(pred_classes_img) if scores_img[i] > SCORE_THRESHOLD]
            not_matches = {}
            for j in gt_idx_class:
                fn_detections = []
                gt_box = gt_boxes_img[j]
                (xmin, ymin, xmax, ymax) = gt_box
                gt_box = [xmin*w, ymin*h, xmax*w, ymax*h]
                ious_aux = []
                pred_classes_aux = []
                pred_classes_ious = []
                if len(pred_boxes_img) == 0:
                    dataResults[category_index[c]['name']]['predictions'].append('fn')
                    dataResults[category_index[c]['name']]['ious'].append(None)
                    dataResults[category_index[c]['name']]['scores'].append(0.0)
                    dataResults[category_index[c]['name']]['pred_classes'].append(None)
                else:
                    # for x, pred_box in enumerate(pred_boxes_img):
                    for x in pred_idx_class:
                        pred_box = pred_boxes_img[x]
                        (ymin, xmin, ymax, xmax) = pred_box
                        pred_box = [xmin*w, ymin*h, xmax*w, ymax*h]
                        iou = intersectionOverUnion(gt_box, pred_box)
                        if iou > IOU_THRESHOLD and pred_classes_img[x]==category_index[c]['id']:
                            ious_aux.append(iou)
                            if scores_img[x] > SCORE_THRESHOLD:
                                pred_classes_aux.append(pred_classes_img[x])
                                pred_classes_ious.append(iou)
                            dataResults[category_index[c]['name']]['scores'].append(scores_img[x])
                            dataResults[category_index[c]['name']]['pred_classes'].append(pred_classes_img[x])
                        elif iou > IOU_THRESHOLD:
                            fn_detections.append((iou, scores_img[x], pred_classes_img[x]))
                            if scores_img[x] > SCORE_THRESHOLD:
                                pred_classes_aux.append(pred_classes_img[x])
                                pred_classes_ious.append(iou)
                        else:
                            if not_matches.get(x) is None:
                                not_matches[x] = 0
                            not_matches[x] += 1
                    if len(ious_aux) > 0:
                        max_iou = max(ious_aux)
                        for iou in ious_aux:
                            if max_iou == iou:
                                dataResults[category_index[c]['name']]['predictions'].append('tp')
                            else:
                                dataResults[category_index[c]['name']]['predictions'].append('fp')
                        dataResults[category_index[c]['name']]['ious'].extend(ious_aux)
                    if len(pred_classes_aux) > 0:
                        max_iou = max(pred_classes_ious)
                        for x, iou in enumerate(pred_classes_ious):
                            if max_iou == iou:
                                confusionMatrix[c-1][pred_classes_aux[x]-1] += 1
                    for fn_detection in fn_detections:
                        dataResults[category_index[c]['name']]['predictions'].append('fn')
                        dataResults[category_index[c]['name']]['ious'].append(fn_detection[0])
                        dataResults[category_index[c]['name']]['scores'].append(fn_detection[1])
                        dataResults[category_index[c]['name']]['pred_classes'].append(fn_detection[2])

            for x in not_matches:
                if not_matches[x] == len(gt_idx_class):
                    dataResults[category_index[c]['name']]['predictions'].append('fp')
                    dataResults[category_index[c]['name']]['ious'].append(None)
                    dataResults[category_index[c]['name']]['scores'].append(scores_img[x])
                    dataResults[category_index[c]['name']]['pred_classes'].append(pred_classes_img[x])
    return dataResults, confusionMatrix, detectionTime

# Calcula la Precision y el Recall para un detector de objetos
def calculatePrecisionsRecallsOD(dataResults):
    # Se preparan las variables
    precisions = {}
    recalls = {}
    precisions_inter = {}
    for category in dataResults:
        total_tp = len([1 for j in dataResults[category]["predictions"] if j == "tp"])
        total_fn = len([1 for j in dataResults[category]["predictions"] if j == "fn"])
        tp = 0
        fp = 0
        precisions[category] = []
        recalls[category] = []
        precisions_inter[category] = []
        sort_data_idx = np.argsort(dataResults[category]["scores"])
        for idx in sort_data_idx:
            if dataResults[category]["predictions"][idx] != "fn":
                if dataResults[category]["predictions"][idx] == "tp":
                    tp += 1
                elif dataResults[category]["predictions"][idx] == "fp":
                    fp += 1
                precision = 0.0
                if tp > 0 or fp > 0:
                    precision = tp / (tp + fp)
                recall = 0.0
                if total_tp > 0:
                    recall = tp / (total_tp + total_fn)
                precisions[category].append(precision)
                recalls[category].append(recall)

        for i, recall in enumerate(recalls[category]):
            precisions_inter[category].append(max(precisions[category][i:]))
    return precisions_inter, precisions, recalls

# Calcula la AP
def calculateAveragePrecision(precisions_inter, recalls):
    recall_points = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ap = {}
    for category in precisions_inter:
        sum_precisions = 0.0
        if len(recalls[category]) > 0:
            for point in recall_points:
                idx = -1
                for i, recall in enumerate(recalls[category]):
                    if recall >= point:
                        idx = i
                        break
                sum_precisions += precisions_inter[category][idx]
        ap[category] = sum_precisions / len(recall_points)
    return ap


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
    widgets = ["Evaluando datos del tfrecord: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=1, widgets=widgets).start()
    # Se extraen los datos del tfrecord
    recordData = TFRecord(TFRECORDS_PATH)
    (images, all_categories, boxes, labels, heights, widths, depths, source_ids, formats, filenames) = recordData.getData()

    # Se obtienen los resultados y se añaden a las variables previamente creadas
    dataResults, confusionMatrix, detectionTimeTotal = processImages(tfDetector, images, boxes, labels)
    for categories in all_categories:
        total_gt_categories.extend(categories)
    pbar.update(1)
    pbar.finish()
    numImages = len(images)
    meanTimeDetection = detectionTimeTotal / numImages
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
        (images, all_categories, boxes, labels, heights, widths, depths, source_ids, formats, filenames) = recordData.getData()
        numImages = numImages + len(images)
        for categories in all_categories:
            total_gt_categories.extend(categories)
        # Se obtienen las metricas y se añaden a los datos generales
        dataResults_record, confusionMatrixRecord, detectionTimeTF = processImages(tfDetector, images, boxes, labels)
        for c in tfDetector.category_index:
            dataResults[tfDetector.category_index[c]['name']]['predictions'].extend(dataResults_record[tfDetector.category_index[c]['name']]['predictions'])
            dataResults[tfDetector.category_index[c]['name']]['scores'].extend(dataResults_record[tfDetector.category_index[c]['name']]['scores'])
            dataResults[tfDetector.category_index[c]['name']]['ious'].extend(dataResults_record[tfDetector.category_index[c]['name']]['ious'])
            dataResults[tfDetector.category_index[c]['name']]['pred_classes'].extend(dataResults_record[tfDetector.category_index[c]['name']]['pred_classes'])
        
        confusionMatrix = confusionMatrix + np.array(confusionMatrixRecord)

        detectionTimeTotal = detectionTimeTotal + detectionTimeTF

        pbar.update(record_idx)
    pbar.finish()
    meanTimeDetection = detectionTimeTotal / numImages
    
# Se terminan de calcular las difewrentes metricas con los resultados obtenidos
(precisions_inter, precisions, recalls) = calculatePrecisionsRecallsOD(dataResults)
ap = calculateAveragePrecision(precisions_inter, recalls)
mAp = 0.0
for category in ap:
    mAp += ap[category]
mAp = mAp / len(ap)

# Se muestra informacion general
print("Imagenes evaluadas: " + str(numImages))
print("Total Etiquetados:")
for category in tfDetector.labels_names:
    print(category + ": " + str(len([True for label in total_gt_categories if label.decode()==category])))
print()

# Se calculan y muestran las metricas extraidas
print("Tiempo medio deteccion de objetos: " + str(meanTimeDetection) + " segundos")

for category in ap:
    print("AP@" + str(category) + " = " + str(ap[category]))
print("mAP = " + str(mAp))

for c in tfDetector.category_index:
    dataResults_class = dataResults[tfDetector.category_index[c]['name']]
    for i, pred in enumerate(dataResults_class["pred_classes"]):
        if dataResults_class["scores"][i] > SCORE_THRESHOLD and (dataResults_class["ious"][i] is not None and dataResults_class["ious"][i] > IOU_THRESHOLD):
            confusionMatrix[c-1][pred-1] += 1

saveConfusionMatrix(np.array(confusionMatrix), tfDetector.labels_names)

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

FN_all = 0
for l in tfDetector.labels_names:
    TP = len([i for i in dataResults[l]['predictions'] if i == "tp"])
    FP = len([i for i in dataResults[l]['predictions'] if i == "fp"])
    FN = len([i for i in dataResults[l]['predictions'] if i == "fn"])
    print(l + ": [TP: " + str(TP) + ", FP: " + str(FP) + "]")
    FN_all = FN_all + FN
print("FN: " + str(FN_all))