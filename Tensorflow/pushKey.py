# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 13:37:18 2018

@author: Jorge
"""

# Libreria para tratar diversos inputs con opencv
import cv2

# Valores de teclas muy usadas.
ESC = 27
SPACE = 32
INTRO = 13

# Se compreba si se ha pulsado [ESC]
def isEscPushed():
    k = cv2.waitKey(1) & 0xff
    if k == ESC:
        return True
    return False

	# Se compreba si se ha pulsado [SPACE]
def isSpacePushed():
    k = cv2.waitKey(1) & 0xff
    if k == SPACE:
        return True
    return False

# Se compreba si se ha pulsado [INTRO]
def isIntroPushed():
    k = cv2.waitKey(1) & 0xff
    if k == INTRO:
        return True
    return False

# Se compreba si se ha pulsado una tecla indicada por parametro.
def isPushed(key):
    k = cv2.waitKey(1) & 0xff
    if k == key:
        return True
    return False

# Espera a que se pulse [ESC]
def waitEsc():
    while True:
        k = cv2.waitKey(1) & 0xff
        if k == ESC:
            break
        
# Espera a que se pulse [SPACE]
def waitSpace():
    while True:
        k = cv2.waitKey(1) & 0xff
        if k == SPACE:
            break
        
# Espera a que se pulse [INTRO]
def waitIntro():
    while True:
        k = cv2.waitKey(1) & 0xff
        if k == INTRO:
            break

# Espera a que se pulse una tecla cualquiera
def waitAnyKey(time=0):
    k = cv2.waitKey(time) & 0xff
    return k

# Espera a que se pulse una tecla indicada por parametro.
def waitKey(key):
    while True:
        k = cv2.waitKey(1) & 0xff
        if k == key:
            break

# Espera a que se pulse una tecla y devuelve dicha tecla
def keyPushed(time=0):
    k = cv2.waitKey(time) & 0xff
    return k