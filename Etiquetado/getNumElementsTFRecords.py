import tensorflow as tf
from TFRecord import TFRecord
from object_detection.utils import label_map_util
import os
import progressbar

TFRECORDS_PATH = "A:/00_ATRIA/11_Proyectos Financiados/02_Proyectos Presentados/01_Aceptados/07_2019_03_THD_Ecoembes_SIARA/04_PROYECTO/05_Documentos de trabajo/03_Imágenes/Records/3_classes/eval"
LABEL_MAP_PATH = "A:/00_ATRIA/11_Proyectos Financiados/02_Proyectos Presentados/01_Aceptados/07_2019_03_THD_Ecoembes_SIARA/04_PROYECTO/05_Documentos de trabajo/03_Imágenes/Records/3_classes/label_map.pbtxt"

category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)
category_elements = {}
for idx in category_index:
    category_elements[category_index[idx]['name']] = 0



if os.path.isdir(TFRECORDS_PATH):
    listDir = os.listdir(TFRECORDS_PATH)
    widgets = ["Contando elementos: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(listDir), widgets=widgets).start()
    for j, record in enumerate(listDir):
        recordPath = os.path.join(TFRECORDS_PATH, record)
        recordData = TFRecord(path=recordPath)
        elements = recordData.all_categories
        elementsArray = []
        for element in elements:
            elementsArray.extend(element)

        for category in category_elements:
            category_elements[category] += len([i for i, element in enumerate(elementsArray) if element.decode()==category])
        pbar.update(j)

pbar.finish()
for category in category_elements:
    print(category + ": " + str(category_elements[category]))
