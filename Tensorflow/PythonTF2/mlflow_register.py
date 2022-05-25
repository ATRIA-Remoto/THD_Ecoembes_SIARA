import mlflow
from tensorflow.core.util import event_pb2
import tensorflow as tf
import os
from object_detection.utils import config_util, label_map_util

EXPERIMENTO = ""
CONFIG_FILE = "D:\Proyectos\THD_Ecoembes\codigo\THD\Tensorflow\PythonTF2\models\ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8\THD_mobilenet_Jorge.config"
LABEL_MAP = ""
TRAIN_PATH = ""
EVAL_PATH = ""

def readSaveTFEvents(path, prefixTag=""):
    if prefixTag != "":
        prefixTag = prefixTag + "/"
    if os.path.isdir(path):
        listTFEvents = os.listdir(path)
        for TFevent in listTFEvents:
            serialized_examples = tf.data.TFRecordDataset(os.path.join(path, TFevent))
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    t = tf.make_ndarray(value.tensor)
                    try:
                        t = float(t)
                        mlflow.log_metric(prefixTag+str(value.tag), t, step=int(event.step))
                    except Exception as e:
                        pass
    else:
        serialized_examples = tf.data.TFRecordDataset(path)
        for serialized_example in serialized_examples:
            event = event_pb2.Event.FromString(serialized_example.numpy())
            for value in event.summary.value:
                t = tf.make_ndarray(value.tensor)
                try:
                    t = float(t)
                    mlflow.log_metric(prefixTag+str(value.tag), t, step=int(event.step))
                except Exception as e:
                    pass


mlflow.set_experiment(EXPERIMENTO)


pipeline_config = config_util.get_configs_from_pipeline_file(CONFIG_FILE)
num_classes = config_util.get_number_of_classes(pipeline_config['model'])
img_size = config_util.get_spatial_image_size(config_util.get_image_resizer_config(pipeline_config['model']))
optimizer = config_util.get_optimizer_type(pipeline_config['train_config'])
opti2 = getattr(pipeline_config['train_config'].optimizer, optimizer)
learning_rate_type = config_util.get_learning_rate_type(getattr(pipeline_config['train_config'].optimizer, optimizer))
initial_learning_rate = getattr(opti2.learning_rate, learning_rate_type).initial_learning_rate
decay_steps = getattr(opti2.learning_rate, learning_rate_type).decay_steps

category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP, use_display_name=True)
classes_names = []
for idx in category_index:
    classes_names.append(category_index[idx]['name'])

mlflow.log_param("num_classes", num_classes)
mlflow.log_param("img_size", img_size)
mlflow.log_param("optimizer", optimizer)
mlflow.log_param("learning_rate_type", learning_rate_type)
mlflow.log_param("initial_learning_rate", initial_learning_rate)
mlflow.log_param("decay_steps", decay_steps)
mlflow.log_param("classes", classes_names)

readSaveTFEvents(TRAIN_PATH, prefixTag="Train")
readSaveTFEvents(EVAL_PATH, prefixTag="Eval")
mlflow.log_artifacts(CONFIG_FILE)