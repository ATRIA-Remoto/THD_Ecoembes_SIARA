import os

PATH_YOLO = ""
PATH_DATA = ""
PATH_CFG = ""
PATH_WEIGHTS = ""

# TRAINING
command = PATH_YOLO + "/darknet detector train " + PATH_DATA + " " + PATH_CFG + " " + PATH_WEIGHTS + " -gpus 0 -clear 1"
os.system(command)
