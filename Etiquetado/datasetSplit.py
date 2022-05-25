import os
import random
from FileManagement_lib import copyFiles

DATA_PATH = "D:/Proyectos/THD_Ecoembes/Labels/TF_ALT/3classesV/Data"

TRAIN_PATH = "D:/Proyectos/THD_Ecoembes/Labels/TF_ALT/3classesV/Train"
TEST_PATH = "D:/Proyectos/THD_Ecoembes/Labels/TF_ALT/3classesV/Eval"

PERCENTAGE = 0.8

listFiles = os.listdir(DATA_PATH)
numFiles = len(listFiles)

numTrain = int(numFiles*PERCENTAGE+0.5)
random.shuffle(listFiles)

listTrain = listFiles[:numTrain]
listTest = listFiles[numTrain:]

copyFiles(listTrain, DATA_PATH, TRAIN_PATH)
copyFiles(listTest, DATA_PATH, TEST_PATH)
