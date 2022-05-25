from xml.dom import minidom
import time
import os

# XML_ORIGIN = 'D:\Proyectos\THD_Ecoembes\XML/20200828_1.xml'
# XML_NEW = 'D:\Proyectos\THD_Ecoembes\XML/20200828_1_BOX.xml'

XML_DIR = 'D:/Proyectos/THD_Ecoembes/Labels/XML/New'

def loadLabels(xml, name):
    elements = xml.getElementsByTagName('image')
    for element in elements:
        if element.attributes.get('name').nodeValue == name:
            for child in element.childNodes:
                if child.attributes is not None and child.tagName=="polygon":
                    return child.attributes['label'].nodeValue, child.attributes['polygon'].nodeValue
                elif child.attributes is not None and child.tagName=='points':
                    return child.attributes['label'].nodeValue, child.attributes['points'].nodeValue
                else:
                    return None

def getBoxPoints(points):
    xtl = 9999.99
    xbr = 0.0
    ytl = 9999.99
    ybr = 0.0
    for (x, y) in points:
        if x < xtl:
            xtl = x
        if x > xbr:
            xbr = x
        if y < ytl:
            ytl = y
        if y > ybr:
            ybr = y
    return xtl, ytl, xbr, ybr

# xmlOrigin = open(XML_ORIGIN, 'r')
# xmlNew = open(XML_NEW, 'w')

# Se actua sobre los ficheros XML del directorio
for xmlNameOrigin in os.listdir(XML_DIR):
    (name, extension) = xmlNameOrigin.split(".")
    if extension.lower() == "xml":
        # Se crea un ficheor con el mismo nombre indicando el nuevo tipo de anotacion
        xmlNameNew = name + "_BOX.xml"
        xmlOrigin = open(os.path.join(XML_DIR, xmlNameOrigin), 'r')
        xmlNew = open(os.path.join(XML_DIR, xmlNameNew), 'w')
        # Se leen las lineas del fichero xml
        for line in xmlOrigin.readlines():
            # Se obtienen las coordenadas etiquetadas en la imagen
            if "<polyline" in line or "<polygon" in line:
                (startLine, points) = line.split("points=")
                points = points.split(">")[0]
                points = points.split("z_order=")[0]
                points = points.replace("\"", "")
                points = points.split(";")
                listPoints = []
                for s in points:
                    s = s.split(",")
                    point = (float(s[0]), float(s[1]))
                    listPoints.append(point)
                # A partir d elas coordenadas se extraen los 4 puntos exteriores para formar una anotacion en forma de caja
                (xtl, ytl, xbr, ybr) = getBoxPoints(listPoints)
                # Se escriben los datos en el nuevo fichero
                if "<polyline" in startLine:
                    startLine = startLine.replace("<polyline", "<box")
                else:
                    startLine = startLine.replace("<polygon", "<box")
                line = startLine + "xtl=\"" + "{:.2f}".format(xtl) + "\" ytl=\"" + "{:.2f}".format(ytl) + "\" xbr=\"" + "{:.2f}".format(xbr) + "\" ybr=\"" + "{:.2f}".format(ybr) + "\">"
            # Si las lineas leidas no contienen datos de anotacion se escriben en el nuevo fichero en el formato correspondiente
            elif "</polyline>" in line:
                line = line.replace("</polyline>", "</box>")
                line = line.split("\n")[0]
            elif "</polygon>" in line:
                line = line.replace("</polygon>", "</box>")
                line = line.split("\n")[0]
            else:
                line = line.split("\n")[0]
            print(line, file=xmlNew)    # Se escriben en el nuevo fichero
        # Se cierra el fichero leido y el nuevo fichero recien escrito
        xmlOrigin.close()
        xmlNew.close()