from ximea import xiapi, xidefs
import xml.etree.ElementTree as ET
from ctypes import *
from ctypes.wintypes import *

# Clase de control de camaras ximea
class XimeaCam:

    # Inicializacion de variables
    # Se inicia la camara con las caracteristicas deseadas
    def __init__(self, format_cam='XI_RGB24', exposure=10000, auto_wb=True, XML=None):
        try:
            self.format = None
            self.exposure = None
            self.img = None

            self.cam = xiapi.Camera()

            print('Opening first camera...')
            self.cam.open_device()

            if XML is None:
                # Formato de la imagen
                if format in xiapi.XI_IMG_FORMAT:
                    self.format = format_cam
                else:
                    self.format = 'XI_RGB24'

                self.cam.set_imgdataformat(self.format)
            
                # Se le asigna la exposicion
                self.exposure = exposure

                if auto_wb:
                    self.cam.enable_auto_wb()
                
                self.cam.set_exposure(self.exposure)
                #create instance of Image to store image data and metadata
                self.img = xiapi.Image()
                self.cam.start_acquisition()
            else:
                self.loadParams(XML)
                
            print('Exposure was set to %i us' %self.cam.get_exposure())

            
        except Exception as e:
            print("Exception: " + str(e))

    # Se captura una imagen con la camara.
    def takePhoto(self):
        self.cam.get_image(self.img)

        #create numpy array with data from camera. Dimensions of the array are 
        #determined by imgdataformat
        data = self.img.get_image_data_numpy()

        return data

    # Se cierra la comunicación con la camara
    def close(self):
        #stop data acquisition
        print('Stopping acquisition...')
        self.cam.stop_acquisition()

        #stop communication
        self.cam.close_device()
        
    # Carga parametros para la camara a partir de un fichero XML
    def loadParams(self, XML):
        try:
            tree = ET.parse(XML)
            root = tree.getroot()
            values = root[0]
            self.cam.stop_acquisition()

            for child in values:
                value = child.text
                param = child.tag
                if child.attrib['type'] == "int":
                    value = int(value)
                elif child.attrib['type'] == "bool":
                    value = bool(int(value))
                elif child.attrib['type'] == "float":
                    value = float(value)
                elif child.attrib['type'] == "string" and value is None:
                    value = ""

                prm = create_string_buffer(bytes(param, 'UTF-8')) #only python3.x
                
                if not param.split(':')[0] in xidefs.VAL_TYPE:
                    raise RuntimeError('invalid parameter')

                val_type = xidefs.VAL_TYPE[param.split(':')[0]]
                
                if val_type == 'xiTypeString':
                    val_len = DWORD(len(value))
                    val = create_string_buffer(bytes(value, 'UTF-8')) #only python3.x
                elif val_type == 'xiTypeInteger':
                    val_len = DWORD(4)
                    val = pointer(c_int(int(value)))
                elif val_type == 'xiTypeFloat':
                    val_len = DWORD(4)
                    val = pointer(FLOAT(value))
                elif val_type == 'xiTypeEnum':
                    val_len = DWORD(4)
                    val = pointer(c_int(int(value)))
                elif val_type == 'xiTypeBoolean':
                    val_len = DWORD(4)
                    val = pointer(c_int(int(value)))       
                elif val_type == 'xiTypeCommand':
                    val_len = DWORD(4)

                    val = pointer(c_int(int(value)))       
                    
                stat = self.cam.device.xiSetParam(
                    self.cam.handle,
                    prm,
                    val,
                    val_len,
                    xidefs.XI_PRM_TYPE[val_type]
                    )

                if not stat == 0:
                    print("[ERROR] Error modificando " + param)
            
            print('Exposure was set to %i us' %self.cam.get_exposure())
            
            #create instance of Image to store image data and metadata
            self.img = xiapi.Image()
            self.cam.start_acquisition()
        except Exception as e:
            print("[ERROR] Error al cargar parametros.")
            print(str(e))
