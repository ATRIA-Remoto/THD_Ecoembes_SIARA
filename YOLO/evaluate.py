from executeImageNN_lib import YOLONet
from metricsFunctions import intersectionOverUnion, saveConfusionMatrix, calcLRP
import numpy as np
import progressbar
import cv2, os, time


TEST_PATH = "A:/00_ATRIA/11_Proyectos Financiados/02_Proyectos Presentados/01_Aceptados/07_2019_03_THD_Ecoembes_SIARA/04_PROYECTO/05_Documentos de trabajo/03_Imágenes/Records/3_classes/test"#/20200905_1.tfrecord"
CONFIG_PATH = "D:\Proyectos\THD_Ecoembes\codigo\THD\Tensorflow\experiments/exported_model\saved_model"
WEIGHTS_PATH = "A:/00_ATRIA/11_Proyectos Financiados/02_Proyectos Presentados/01_Aceptados/07_2019_03_THD_Ecoembes_SIARA/04_PROYECTO/05_Documentos de trabajo/03_Imágenes/Records/3_classes/label_map.pbtxt"

SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


def extractYoloLabel(labelPath):
    boxes = []
    labels = []
    f = open(labelPath, "r")
    lines = f.readlines()
    for line in lines:
        line = line.replace("\\n","")
        (label, centerX, centerY, width, height) = line.split(" ")
        xtl = centerX - width/2
        ytl = centerY - height/2
        xbr = centerX + width/2
        ybr = centerY + height/2
        labels.append(label)
        boxes.append([xtl, ytl, xbr, ybr])
    f.close()
    return (boxes, labels)

def processImages(yoloDetector, imagesList):
    widgets = ["Evaluando los datos de los tfrecords: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(listFiles), widgets=widgets).start()
    dataResults = {}
    confusionMatrix = []
    detectionTime = 0.0     # Time expended in detection task
    labels = yoloDetector.labels
    for c in labels:
        dataResults[c] = {}
        dataResults[c]['predictions'] = []
        dataResults[c]['scores'] = []
        dataResults[c]['ious'] = []
        dataResults[c]['pred_classes'] = []
        confusionMatrix.append([])
        for i, c_pred in enumerate(yoloDetector.labels):
            confusionMatrix[i].append(0)

    for i, imgFile in enumerate(imagesList):
        imgPath = os.path.join(TEST_PATH, imgFile)
        labelPath = imgFile.split(".")[0] + ".txt"
        (gt_boxes_img, labels_img) = extractLabelYolo(labelPath)
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        tInit = time.time()
        (pred_boxes_img, pred_classes_img, scores_img) = yoloDetector.execute(img)
        tEnd = time.time()
        detectionTime = detectionTime + (tEnd - tInit)
        (h, w, d) = np.shape(img)
        
        for c in labels:
            gt_idx_class = [i for i, x in enumerate(labels_img) if x==c['id']]
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
                    dataResults[c]['predictions'].append('fn')
                    dataResults[c]['ious'].append(None)
                    dataResults[c]['scores'].append(0.0)
                    dataResults[c]['pred_classes'].append(None)
                else:
                    # for x, pred_box in enumerate(pred_boxes_img):
                    for x in pred_idx_class:
                        pred_box = pred_boxes_img[x]
                        (ymin, xmin, ymax, xmax) = pred_box
                        pred_box = [xmin*w, ymin*h, xmax*w, ymax*h]
                        iou = intersectionOverUnion(gt_box, pred_box)
                        if iou > IOU_THRESHOLD and pred_classes_img[x]==c['id']:
                            ious_aux.append(iou)
                            if scores_img[x] > SCORE_THRESHOLD:
                                pred_classes_aux.append(pred_classes_img[x])
                                pred_classes_ious.append(iou)
                            dataResults[c]['scores'].append(scores_img[x])
                            dataResults[c]['pred_classes'].append(pred_classes_img[x])
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
                                dataResults[c]['predictions'].append('tp')
                            else:
                                dataResults[c]['predictions'].append('fp')
                        dataResults[c]['ious'].extend(ious_aux)
                    if len(pred_classes_aux) > 0:
                        max_iou = max(pred_classes_ious)
                        for x, iou in enumerate(pred_classes_ious):
                            if max_iou == iou:
                                confusionMatrix[c-1][pred_classes_aux[x]-1] += 1
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
    return dataResults, confusionMatrix, detectionTime


def calculatePrecisionsRecallsOD(dataResults):
    precisions = {}
    recalls = {}
    precisions_inter = {}
    for category in dataResults:
        total_tp = len([1 for j in dataResults[category]["predictions"] if j == "tp"])
        total_fn = len([1 for j in dataResults[category]["predictions"] if j == "fn"])
        # print("total tp: " + str(total_tp))
        # print("total fn: " + str(total_fn))
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



print("[INFO] Loading model")
yoloDetector = YOLONet(CONFIG_PATH, WEIGHTS_PATH, confidence=SCORE_THRESHOLD)
print("[INFO] Model loaded")

dataResults = {}

confusionMatrix = []
for i, c in enumerate(yoloDetector.labels):
    confusionMatrix.append([])
    for c_pred in yoloDetector.labels:
        confusionMatrix[i].append(0)

confusionMatrix = np.array(confusionMatrix)

for i, c in enumerate(yoloDetector.labels):
    dataResults[c] = {}
    dataResults[c]['predictions'] = []
    dataResults[c]['scores'] = []
    dataResults[c]['ious'] = []
    dataResults[c]['pred_classes'] = []

total_gt_categories = []

meanTimeDetection = 0.0

listFiles = os.listdir(TEST_PATH)
numImages = len(listFiles)
detectionTimeTotal = 0.0

# for (idx, imgFile) in enumerate(listFiles):
#     imgPath = os.path.join(TEST_PATH, imgFile)
#     img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
#     (images, all_categories, boxes, labels, heights, widths, depths, source_ids, formats, filenames) = recordData.getData()
#     for categories in all_categories:
#         total_gt_categories.extend(categories)
#     dataResults_record, confusionMatrixRecord, detectionTimeTF = processImages(yoloDetector, images, boxes, labels)
#     for c in yoloDetector.labels:
#         dataResults[c]['predictions'].extend(dataResults_record[c]['predictions'])
#         dataResults[c]['scores'].extend(dataResults_record[c]['scores'])
#         dataResults[c]['ious'].extend(dataResults_record[c]['ious'])
#         dataResults[c]['pred_classes'].extend(dataResults_record[c]['pred_classes'])
    
#     confusionMatrix = confusionMatrix + np.array(confusionMatrixRecord)
#     detectionTimeTotal = detectionTimeTotal + detectionTimeTF
#     pbar.update(idx)
meanTimeDetection = detectionTimeTotal / numImages
    

(precisions_inter, precisions, recalls) = calculatePrecisionsRecallsOD(dataResults)
ap = calculateAveragePrecision(precisions_inter, recalls)
mAp = 0.0
for category in ap:
    mAp += ap[category]
mAp = mAp / len(ap)

print("Imagenes evaluadas: " + str(numImages))
print("Total Etiquetados:")
for category in yoloDetector.labels_names:
    print(category + ": " + str(len([True for label in total_gt_categories if label.decode()==category])))
print()

print("Tiempo medio deteccion de objetos: " + str(meanTimeDetection) + " segundos")

for category in ap:
    print("AP@" + str(category) + " = " + str(ap[category]))
print("mAP = " + str(mAp))

for c in yoloDetector.labels:
    dataResults_class = dataResults[c]
    for i, pred in enumerate(dataResults_class["pred_classes"]):
        if dataResults_class["scores"][i] > SCORE_THRESHOLD and (dataResults_class["ious"][i] is not None and dataResults_class["ious"][i] > IOU_THRESHOLD):
            confusionMatrix[c-1][pred-1] += 1

saveConfusionMatrix(np.array(confusionMatrix), yoloDetector.labels)

lrp = {}
for c in yoloDetector.labels:
    dataResults_class = dataResults[c]
    nTP = len([prediction for i, prediction in enumerate(dataResults_class['predictions']) if prediction == 'tp' and dataResults_class["ious"][i] is not None and dataResults_class["ious"][i] > IOU_THRESHOLD and dataResults_class["scores"][i] is not None and dataResults_class["scores"][i] > SCORE_THRESHOLD])
    nFP = len([prediction for i, prediction in enumerate(dataResults_class['predictions']) if prediction == 'fp' and dataResults_class["ious"][i] is not None and dataResults_class["ious"][i] > IOU_THRESHOLD and dataResults_class["scores"][i] is not None and dataResults_class["scores"][i] > SCORE_THRESHOLD])
    nFN = len([prediction for i, prediction in enumerate(dataResults_class['predictions']) if prediction == 'fn' and dataResults_class["ious"][i] is not None and dataResults_class["ious"][i] > IOU_THRESHOLD and dataResults_class["scores"][i] is not None and dataResults_class["scores"][i] > SCORE_THRESHOLD])
    iousTP = [iou for i, iou in enumerate(dataResults_class["ious"]) if dataResults_class['predictions'][i] == 'tp' and iou is not None and iou > IOU_THRESHOLD and dataResults_class["scores"][i] is not None and dataResults_class["scores"][i] > SCORE_THRESHOLD]
    lrp[c] = calcLRP(nTP, nFP, nFN, iousTP, IOU_THRESHOLD)

mLRP = 0.0
for category in lrp:
    print("LRP@" + str(category) + " = " + str(lrp[category]))
    mLRP += lrp[category]
mLRP = mLRP / len(lrp)

print("mLRP = " + str(mLRP))
