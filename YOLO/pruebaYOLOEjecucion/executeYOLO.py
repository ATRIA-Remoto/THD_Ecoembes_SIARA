import darknet
import cv2, os
from pushKey import keyPushed, ESC

CONFIG_FILE = "D:/Proyectos/THD_Ecoembes/codigo/THD/YOLO/data/yolov4-leaky-416.cfg"
DATA_FILE = "D:/Proyectos/THD_Ecoembes/codigo/THD/YOLO/data/THD.data"
WEIGHTS_FILE = "D:/Proyectos/THD_Ecoembes/codigo/THD/YOLO/backup_train/yolov4-leaky-416_best5_map-82-86.weights"

PATH_IMAGE_DIR = "D:/Proyectos/THD_Ecoembes/Labels/YOLO/NoBorders/3_classes/Test"

EXTENSIONS = ["jpg", "png", "bmp"]

network, class_names, class_colors = darknet.load_network(
        CONFIG_FILE,
        DATA_FILE,
        WEIGHTS_FILE,
        batch_size=1
    )


width = darknet.network_width(network)
height = darknet.network_height(network)
print("Height: " + str(height))
print("Width: " + str(width))
darknet_image = darknet.make_image(width, height, 3)

listImages = os.listdir(PATH_IMAGE_DIR)
listImages = [f for f in listImages if f.split(".")[1] in EXTENSIONS]

for imageName in listImages:
    print(imageName + ": ")
    imagePath = os.path.join(PATH_IMAGE_DIR, imageName)
    img = cv2.imread(imagePath)
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height),
                                    interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

    detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.5)
    darknet.print_detections(detections, True)

    image = darknet.draw_boxes(detections, frame_resized, class_colors)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Inference', image)

    key = None
    if len(detections) == 0:
        key = keyPushed(time=1)
    else:
        key = keyPushed(time=0)
    if key == ESC:
        break

cv2.destroyAllWindows()