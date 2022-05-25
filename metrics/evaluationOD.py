import os, cv2
import time
from metricsFunctions import intersectionOverUnion
import progressbar
import numpy as np

def processImagesFiles(detector, imagesList, imagesPath=None, bar=True):
    pbar= None
    if bar:
        widgets = ["Leyendo imagenes: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(imagesList), widgets=widgets).start()
    detections = []
    detectionTime = 0.0
    for i, f in enumerate(imagesList):
        if imagesPath is not None:
            f = os.path.join(imagesPath, f)
        img = cv2.imread(f)
        tIni = time.time()
        detection = detector.execute(img)
        detectionTime = detectionTime + (time.time() - tIni)
        detections.append(detection)
        if bar:
            pbar.update(i)
    if bar:
        pbar.finish()
    detectionTime = detectionTime/len(imagesList)
    return detections, detectionTime

def processImages(detector, images, bar=True):
    pbar= None
    if bar:
        widgets = ["Leyendo imagenes: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(images), widgets=widgets).start()
    detections = []
    detectionTime = 0.0
    for i, img in enumerate(images):
        tIni = time.time()
        detection = detector.execute(img)
        detectionTime = detectionTime + (time.time() - tIni)
        detections.append(detection)
        if bar:
            pbar.update(i)
    if bar:
        pbar.finish()
    detectionTime = detectionTime/len(images)
    return detections, detectionTime


# Funcion principal de procesamiento de la simagenes, carga las imagenes, las pasa por el detector y extrae
# scores, IoU, TP, FP, TN, FN... 
# all_gt_boxes: lista de listas. Cada lista contiene las gt_boxes de una imagen
# labels: lista de listas. Cada lista lleva las etiquetas (str) correspondientes a las gt_boxes.
# all_pred_boxes: lista de listas. Cada lista contiene las predicciones realizadas en una imagen
# labelsPredicted: lista de listas. Cada lista lleva las predicciones (string) correspondientes a las pred_boxes.
# all_scores: lista de listas. Cada lista contiene las score correspondientes a las predicciones.
# classes: Lista. Lista con todas las etiquetas posibles (string).
# domensions: Diccionario. Diccionario con los valores width y height de las imagenes.
def extractResults(all_gt_boxes, labels, all_pred_boxes, labelsPredicted, all_scores, classes, dimensions, score_threshold=0.5, iou_threshold=0.5):
    widgets = ["Evaluando imagenes: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(labels), widgets=widgets).start()
    dataResults = {}
    confusionMatrix = []
    for i, c in enumerate(classes): 
        dataResults[c] = {}
        dataResults[c]['predictions'] = []
        dataResults[c]['scores'] = []
        dataResults[c]['ious'] = []
        dataResults[c]['pred_classes'] = []
        confusionMatrix.append([])
        for c_pred in classes:
            confusionMatrix[i].append(0)

    for i, labels_img in enumerate(labels):
        gt_boxes_img = all_gt_boxes[i]
        pred_boxes_img = all_pred_boxes[i]
        pred_classes_img = labelsPredicted[i]
        scores_img = all_scores[i]
        
        for z, c in enumerate(classes):
            gt_idx_class = [i for i, x in enumerate(labels_img) if x==c]
            # pred_idx_class = [i for i, x in enumerate(pred_classes_img) if x==category_index[c]['id']]
            pred_idx_class = [i for i, x in enumerate(pred_classes_img) if scores_img[i] > score_threshold]
            not_matches = {}
            for j in gt_idx_class:
                fn_detections = []
                gt_box = gt_boxes_img[j]
                (xmin, ymin, xmax, ymax) = gt_box
                gt_box = [xmin*dimensions["width"], ymin*dimensions["height"], xmax*dimensions["width"], ymax*dimensions["height"]]
                ious_aux = []
                pred_classes_aux = []
                pred_classes_ious = []
                if len(pred_boxes_img) == 0:
                    dataResults[c]['predictions'].append('fn')
                    dataResults[c]['ious'].append(None)
                    dataResults[c]['scores'].append(0.0)
                    dataResults[c]['pred_classes'].append(None)
                else:
                    # for x, pred_box in enumerate(pred_boxes_img):
                    for x in pred_idx_class:
                        pred_box = pred_boxes_img[x]
                        (xmin, ymin, xmax, ymax) = pred_box
                        pred_box = [xmin*dimensions["width"], ymin*dimensions["height"], xmax*dimensions["width"], ymax*dimensions["height"]]
                        iou = intersectionOverUnion(gt_box, pred_box)
                        if iou > iou_threshold and pred_classes_img[x]==c:
                            ious_aux.append(iou)
                            if scores_img[x] > score_threshold:
                                pred_classes_aux.append(pred_classes_img[x])
                                pred_classes_ious.append(iou)
                            dataResults[c]['scores'].append(scores_img[x])
                            dataResults[c]['pred_classes'].append(pred_classes_img[x])
                        elif iou > iou_threshold:
                            fn_detections.append((iou, scores_img[x], pred_classes_img[x]))
                            if scores_img[x] > score_threshold:
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
                                dataResults[c]['predictions'].append('tp')
                            else:
                                dataResults[c]['predictions'].append('fp')
                        dataResults[c]['ious'].extend(ious_aux)
                    if len(pred_classes_aux) > 0:
                        max_iou = max(pred_classes_ious)
                        for x, iou in enumerate(pred_classes_ious):
                            if max_iou == iou:
                                confusionMatrix[z][classes.index(pred_classes_aux[x])] += 1
                    for fn_detection in fn_detections:
                        dataResults[c]['predictions'].append('fn')
                        dataResults[c]['ious'].append(fn_detection[0])
                        dataResults[c]['scores'].append(fn_detection[1])
                        dataResults[c]['pred_classes'].append(fn_detection[2])

            for x in not_matches:
                if not_matches[x] == len(gt_idx_class):
                    dataResults[c]['predictions'].append('fp')
                    dataResults[c]['ious'].append(None)
                    dataResults[c]['scores'].append(scores_img[x])
                    dataResults[c]['pred_classes'].append(pred_classes_img[x])
        pbar.update(i)
    pbar.finish()
    return dataResults, confusionMatrix

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