from xml.dom import minidom

class LabelLoaderCVAT:

    def __init__(self, xmlPath):
        self.xml = minidom.parse(xmlPath)


    def load(self, name):

        elements = self.xml.getElementsByTagName('image')

        for element in elements:
            if element.attributes.get('name').nodeValue == name:
                for child in element.childNodes:
                    if child.attributes is not None:
                        return child.attributes['label'].nodeValue
                        