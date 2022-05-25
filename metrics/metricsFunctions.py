from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import numpy as np


# Funcion de calculo de métricas vasicas y matriz de confusión
def evaluate(labels, predictions, names=None):
    report = classification_report(labels, predictions, target_names=names)
    print(report)
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal', values_format=None)

    plt.savefig("./confusionMatrix.jpg")

# Devuelve Precision y Recall a partir de los resultados proporcionados
def calcPrecisionRecall(tp, fp, fn):
    if tp == 0 and fp == 0:
        precision = 0.0
    else:
        precision = tp/(tp+fp)
    if tp == 0 and fn == 0:
        recall = 0.0
    else:
        recall = tp/(tp+fn)
    return (precision, recall)

# Calcula la Location Recall Precision a partir de los resultados proporcionados
def calcLRP(nTP, nFP, nFN, iousTP, iou_thresh):
    total = nTP + nFP + nFN
    nTPError = np.sum((1-np.array(iousTP))/(1-iou_thresh))
    lrp = (nTPError + nFP + nFN) / total
    return lrp

# Calcula el grado de coincidencia entre la localización de una objeto y posición real
def intersectionOverUnion(box_truth, box_pred):
    (xtl_truth, ytl_truth, xbr_truth, ybr_truth) = box_truth
    (xtl_pred, ytl_pred, xbr_pred, ybr_pred) = box_pred

    inter_top = max(ytl_truth, ytl_pred)
    inter_left = max(xtl_truth, xtl_pred)
    inter_right = min(xbr_truth, xbr_pred)
    inter_bottom = min(ybr_truth, ybr_pred)

    if inter_right > inter_left and inter_bottom > inter_top:
        I = (inter_right-inter_left)*(inter_bottom-inter_top)

        predArea = (ybr_pred-ytl_pred)*(xbr_pred-xtl_pred)
        truthArea = (ybr_truth-ytl_truth)*(xbr_truth-xtl_truth)
        
        U = predArea + truthArea - I

        IoU = I/U

        return IoU
    else:
        return 0.0

# Guarda una imagen de la matriz de confusion
def saveConfusionMatrix(cm, names=None):
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal', values_format=None)

    plt.savefig("./confusionMatrix.jpg")
