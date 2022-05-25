import cv2, os
import progressbar
from evaluationOD import *
from metricsFunctions import intersectionOverUnion, saveConfusionMatrix, calcLRP
from preprocessors_lib import FormatPreprocessor, SizePreprocessor
import darknet
from executeImageNN_lib import YOLONet

TEST_PATH = "D:/Proyectos/THD_Ecoembes/Labels/YOLO/NoBorders/3_classes/Test2"
CONFIG_PATH = "D:/Proyectos/THD_Ecoembes/codigo/THD/YOLO/data/yolov4-leaky-416.cfg"
WEIGHTS_PATH = "D:/Proyectos/THD_Ecoembes/codigo/THD/YOLO/backup_train/yolov4-leaky-416_best5_map-82-86.weights"
DATA_PATH = "D:/Proyectos/THD_Ecoembes/codigo/THD/YOLO/data/THD_test.data"
EXTENSIONS = ["jpg", "png", "bmp"]

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
        xtl = float(centerX) - float(width)/2
        ytl = float(centerY) - float(height)/2
        xbr = float(centerX) + float(width)/2
        ybr = float(centerY) + float(height)/2
        labels.append(int(label))
        boxes.append([xtl, ytl, xbr, ybr])
    f.close()
    return (boxes, labels)

print("[INFO] Loading model")
yoloDetector = YOLONet(CONFIG_PATH, WEIGHTS_PATH, DATA_PATH, confidence=SCORE_THRESHOLD)
print("[INFO] Model loaded")
# Se carga la red YOLO, indicando el tamaño de redimension de las imagenes y ficheros de configuracion
preProcessors = [SizePreprocessor(yoloDetector.width, yoloDetector.height), FormatPreprocessor(color=cv2.COLOR_BGR2RGB)]
yoloDetector.preprocesors = preProcessors

listFiles = os.listdir(TEST_PATH)
numImages = int(len(listFiles)/2)

imagesList = [f for f in listFiles if f.split(".")[1].lower() in EXTENSIONS]

labelsList = [f for f in listFiles if f.split(".")[1].lower() == "txt"]

(detections, detectionTime) = processImagesFiles(yoloDetector, imagesList, TEST_PATH)

all_gt_boxes = []
labels = []
all_pred_boxes = []
labelsPredicted = []
all_scores = []

classes = yoloDetector.labels

dimensions = {
    "width": yoloDetector.width,
    "height": yoloDetector.height
}


widgets = ["Extrayendo datos del detector: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(detections), widgets=widgets).start()
for i, detection in enumerate(detections):
    labelName = labelsList[i]
    labelPath = os.path.join(TEST_PATH, labelName)
    labelsPredsImg = []
    scores_img = []
    pred_bboxes = []
    for label, confidence, bbox in detection:
        labelsPredsImg.append(label)
        scores_img.append(float(confidence)/100.0)
        (xmin, ymin, xmax, ymax) = darknet.bbox2points(bbox)
        pred_bboxes.append((xmin/yoloDetector.width, ymin/yoloDetector.height, xmax/yoloDetector.width, ymax/yoloDetector.height))
    
    (gt_boxes, gt_labels) = extractYoloLabel(labelPath)
    gt_labels_aux = []
    for gt_label in gt_labels:
        gt_labels_aux.append(classes[gt_label])
    all_gt_boxes.append(gt_boxes)
    labels.append(gt_labels_aux)
    all_pred_boxes.append(pred_bboxes)
    labelsPredicted.append(labelsPredsImg)
    all_scores.append(scores_img)
    pbar.update(i)
pbar.finish()

(dataResults, confusionMatrix) = extractResults(all_gt_boxes, labels, all_pred_boxes, labelsPredicted, all_scores, classes, dimensions)

total_gt_categories = []
# Muestra los resultados calculados
print("Imagenes evaluadas: " + str(numImages))
for txt in labelsList:
    txtPath = os.path.join(TEST_PATH, txt)
    (boxes, labels) = extractYoloLabel(txtPath)
    total_gt_categories.extend(labels)
print("Total Etiquetados:")
for category in yoloDetector.labels:
    print(category + ": " + str(len([True for label in total_gt_categories if yoloDetector.labels[label]==category])))
print()

print("Tiempo de detección por imagen: " + str(detectionTime) + " s")

FN_all = 0
for l in yoloDetector.labels:
    TP = len([i for i in dataResults[l]['predictions'] if i == "tp"])
    FP = len([i for i in dataResults[l]['predictions'] if i == "fp"])
    FN = len([i for i in dataResults[l]['predictions'] if i == "fn"])
    print(l + ": [TP: " + str(TP) + ", FP: " + str(FP) + "]")
    FN_all = FN_all + FN
print("FN: " + str(FN_all))

# Calcula la precision y recall para deteccion de objetos
(precisions_inter, precisions, recalls) = calculatePrecisionsRecallsOD(dataResults)

# Con la precision interpolada se obtiene la Average Precision y la Mean Average Precision
ap = calculateAveragePrecision(precisions_inter, recalls)
mAp = 0.0
for category in ap:
    mAp += ap[category]
mAp = mAp / len(ap)


for category in ap:
    print("AP@" + str(category) + " = " + str(ap[category]))
print("mAP = " + str(mAp))

# Se calcula la Location Recall Precision
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

for j, c in enumerate(yoloDetector.labels):
    dataResults_class = dataResults[c]
    for i, pred in enumerate(dataResults_class["pred_classes"]):
        if dataResults_class["scores"][i] > SCORE_THRESHOLD and (dataResults_class["ious"][i] is not None and dataResults_class["ious"][i] > IOU_THRESHOLD):
            confusionMatrix[j][yoloDetector.labels.index(pred)] += 1

saveConfusionMatrix(np.array(confusionMatrix), yoloDetector.labels)