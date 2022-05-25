import cv2
import numpy as np
import os

class ImageLoader:

    def __init__(self, preprocessors=None, labelLoader=None):
        self.preprocessors = preprocessors
        self.labelLoader = labelLoader

    def load(self, imagePath):
        imageList = os.listdir(imagePath)
        images = []
        labels = []
        for file in imageList:
            image = cv2.imread(imagePath + file)
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
                    
            if self.labelLoader is not None:
                label = self.labelLoader.load(file)
                labels.append(label)
            
            images.append(image)

        
        return (np.array(images), np.array(labels))
        