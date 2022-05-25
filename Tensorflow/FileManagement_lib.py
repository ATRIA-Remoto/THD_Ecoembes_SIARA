# Funciones para gestion de ficheros
import os
from shutil import copyfile
import glob
import progressbar

# COmprueba una ruta especicada y si no existe la crea.
def checkCreateFilePath(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)

# Copia una lista de ficheros de una ruta a otra.
def copyFiles(fileList, SRC_FOLDER, DESTINY_FOLDER):
    checkCreateFilePath(DESTINY_FOLDER)

    widgets = ["Copying files: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(fileList), widgets=widgets).start()

    for (i, f) in enumerate(fileList):
        for file in glob.glob(os.path.join(SRC_FOLDER, f)):
            copyfile(file, os.path.join(DESTINY_FOLDER, os.path.basename(file)))
        pbar.update(i)

    pbar.finish()
    print("Ficheros Copiados")
    
#  mueve una lista de ficheros de una ruta a otra.
def moveFiles(fileList, SRC_FOLDER, DESTINY_FOLDER):
    checkCreateFilePath(DESTINY_FOLDER)

    widgets = ["Moving files: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(fileList), widgets=widgets).start()

    for (i, f) in enumerate(fileList):
        for file in glob.glob(os.path.join(SRC_FOLDER, f)):
            move(file, os.path.join(DESTINY_FOLDER, os.path.basename(file)))
        pbar.update(i)

    pbar.finish()
    print("Ficheros movidos")
