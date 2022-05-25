import subprocess
import configparser
import os, shutil, time
import watchdog.events 
import watchdog.observers
from pathlib import Path
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.core.util import event_pb2
import glob
from datetime import datetime
import mlflow
from object_detection.utils import config_util, label_map_util
import progressbar

parent_dir = os.path.abspath(os.path.dirname(__file__))
parser = configparser.ConfigParser()
parser.read(os.path.join(parent_dir, 'config.ini'))

python_path = parser.get('ObjectDetection', 'PATH_PYTHON')
train_function = parser.get('ObjectDetection', 'TRAIN_FUNC')
export_function = parser.get('ObjectDetection', 'EXPORT_FUNC')
pipeline_file = parser.get('ObjectDetection', 'PRETRAINED_MODEL_PATH')
num_steps = int(parser.get('ObjectDetection', 'STEPS'))
export_dir = parser.get('ObjectDetection', 'EXPORT_DIR')
checkpoint_dir = parser.get('ObjectDetection', 'CHECKPOINT_DIR')
checkpoint_steps = parser.get('ObjectDetection', 'CHECKPOINT_STEPS')
samples = parser.get('ObjectDetection', 'SAMPLE')
patience = int(parser.get('ObjectDetection', 'PATIENCE'))
keepFiles = bool(parser.get('ObjectDetection', 'KEEP_FILES'))
minEventSize = int(parser.get('ObjectDetection', 'MIN_EVENT_SIZE'))

mlflow_path = parser.get('ObjectDetection', 'PATH_MLFLOW')
experiment_name = parser.get('ObjectDetection', 'EXPERIMENT_NAME')
run_name = parser.get('ObjectDetection', 'RUN_NAME')

workingDirectory = os.getcwd()
best_step = None

def readUploadTFEvents(path, prefixTag="", wd="", bar=False):
    if prefixTag is not None and prefixTag != "":
        prefixTag = prefixTag + "/"
    serialized_examples_len = 0
    widgets = None
    pbar = None 
    mydir = os.getcwd()
    if os.path.isdir(path):
        listTFEvents = os.listdir(path)
        for i, TFevent in enumerate(listTFEvents):
            serialized_examples = tf.data.TFRecordDataset(os.path.join(path, TFevent))
            if bar:
                for serialized_example in serialized_examples:
                    serialized_examples_len = serialized_examples_len + 1
                widgets = ["Leyendo TFEvent (" + str(i+1) + "/" + str(len(listTFEvents)) + "): ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
                pbar = progressbar.ProgressBar(maxval=serialized_examples_len, widgets=widgets).start()
            for j, serialized_example in enumerate(serialized_examples):
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    t = tf.make_ndarray(value.tensor)
                    try:
                        t = float(t)
                        if wd is not None and wd != "":
                            os.chdir(wd)
                        mlflow.log_metric(prefixTag+str(value.tag), t, step=int(event.step))
                        if wd is not None and wd != "":
                            os.chdir(mydir)
                    except Exception as e:
                        if wd is not None and wd != "":
                            os.chdir(mydir)
                if bar:
                    pbar.update(j)
            if bar:
                pbar.finish()    
    else:
        serialized_examples = tf.data.TFRecordDataset(path)
        if bar:
            for serialized_example in serialized_examples:
                serialized_examples_len = serialized_examples_len + 1
            widgets = ["Leyendo TFEvent: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
            pbar = progressbar.ProgressBar(maxval=serialized_examples_len, widgets=widgets).start()
        for i, serialized_example in enumerate(serialized_examples):
            event = event_pb2.Event.FromString(serialized_example.numpy())
            for value in event.summary.value:
                t = tf.make_ndarray(value.tensor)
                try:
                    t = float(t)
                    if wd is not None and wd != "":
                        os.chdir(wd)
                    mlflow.log_metric(prefixTag+str(value.tag), t, step=int(event.step))
                    if wd is not None and wd != "":
                        os.chdir(mydir)
                except Exception as e:
                    if wd is not None and wd != "":
                        os.chdir(mydir)
            if bar:
                pbar.update(i)
        if bar:
            pbar.finish()

def reduce_events_size(directory):
    """From every training events file in train_dir create a new events file
       with the same content but without images 
    Args:
        directory (str): directory which contains training events files
    """ 
    print("[INFO] Reduciendo Events")
    directory = Path(directory)
    num_events_file = 0
    for events_file in directory.glob('events*'):
        num_events_file = num_events_file + 1
    #out_dir = Path(out_dir).mkdir()
    for i, events_file in enumerate(directory.glob('events*')):
        serialized_examples_len = 0
        for serialized_example in tf.data.TFRecordDataset(str(events_file)):
            serialized_examples_len = serialized_examples_len + 1
        widgets = ["Reduciendo Events (" + str(i+1) + "/" + str(num_events_file) + "): ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=serialized_examples_len, widgets=widgets).start()
        writer = tf.summary.create_file_writer(str(directory))
        for j, serialized_example in enumerate(tf.data.TFRecordDataset(str(events_file))):
            e = event_pb2.Event.FromString(serialized_example.numpy())
            for v in e.summary.value:
                if v.tag != 'train_input_images':
                    value = tf.make_ndarray(v.tensor)
                    with writer.as_default():
                        tf.summary.scalar(v.tag, value, step=e.step)
                        tf.summary.flush()
            pbar.update(j)
        pbar.finish()
        # pbar.update(i)
        events_file.unlink()
    # pbar.finish()

# COmprueba una ruta especicada y si no existe la crea.
def checkCreateFilePath(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)

try:
    os.makedirs(checkpoint_dir)
except Exception as e:
    print(str(e))

try:
    os.makedirs(os.path.join(checkpoint_dir, "eval"))
except Exception as e:
    print(str(e))

try:
    os.makedirs(os.path.join(checkpoint_dir, "train"))
except Exception as e:
    print(str(e))

try:
    os.makedirs(export_dir)
except Exception as e:
    print(str(e))

class Handler(watchdog.events.PatternMatchingEventHandler): 
    def __init__(self):
        self.loss_best = 100
        self.step_best = None
        self.patience_counter = 0
        self.counter = 0
        self.last_event_path = ""
        # Set the patterns for PatternMatchingEventHandler 
        watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=['events.out*'], 
                                                             ignore_directories=True, case_sensitive=True) 
        
    def on_created(self, event): 
        # Detect when a eval events file is created and process it
        print("***WARNING*** Watchdog received created event - % s" % event.src_path)
        print(type(event.src_path))
        self.counter = self.counter + 1
        print("counter = {}".format(self.counter))

        if self.counter >= 2:
            self.summary_values(self.last_event_path)
        self.last_event_path = event.src_path
            
    def summary_values(self, path):
        # Read the values of the events files and compares the Total Loss
        print(path)
        print("summary_values started")
             
        serialized_examples = tf.data.TFRecordDataset(path)
        for serialized_example in serialized_examples:
            events = event_pb2.Event.FromString(serialized_example.numpy())
            for value in events.summary.value:
                t = tf.make_ndarray(value.tensor)
                if value.tag == "Loss/total_loss":
                    print(value.tag, events.step, t, type(t))
                    if t < self.loss_best:
                        print("[INFO] Mejor modelo en step " + str(events.step) + " con loss " + str(t))
                        self.step_best = int(events.step)
                        self.loss_best = t
                        tmp_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
                        best_model_path = os.path.join(tmp_dir, 'checkpoints')
                        checkCreateFilePath(best_model_path)
                        ckpt = self.get_checkpoint_from_step(tmp_dir, events.step)
                        for ckpt_file in glob.glob(os.path.join(tmp_dir, 'evaluation', '{}*'.format(os.path.basename(ckpt)))):
                            shutil.copy2(ckpt_file, best_model_path)
                        with open(os.path.join(best_model_path, 'checkpoint'), 'w') as fp:
                            fp.write('model_checkpoint_path: "{}"'.format(os.path.basename(ckpt)))
                        self.patience_counter = 0
                    else: 
                        self.patience_counter = self.patience_counter + 1
                    print("Current patience is {}".format(self.patience_counter))

    def get_checkpoint_from_step(self, tmp_dir, step):
        all_ckpts = tf.train.get_checkpoint_state(os.path.join(tmp_dir, 'evaluation')).all_model_checkpoint_paths
        for ckpt_path in reversed(all_ckpts):
            if tf.train.load_variable(ckpt_path, 'step/.ATTRIBUTES/VARIABLE_VALUE') == step:
                return ckpt_path

class TensorBoard:

    def __init__(self, directory):
        self.dir = directory
        self.tb_proc = None
    
    def start(self):
        # os.system("tensorboard --logdir=" + str(self.dir) + " &")
        self.tb_proc = subprocess.Popen("tensorboard --logdir=" + str(self.dir))

    def stop(self):
        # os.system("taskkill /IM tensorboard.exe /F")
        self.tb_proc.kill()
        self.tb_proc.wait()


old_cuda_env = None
if "CUDA_VISIBLE_DEVICES" in os.environ:
    old_cuda_env = os.environ["CUDA_VISIBLE_DEVICES"]

tb = None
eval_proc = None
train_proc = None
observer = None
# observer_train = None
training_dir = None

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    eval_command = [python_path, "{}".format(train_function),
                    "--model_dir={}".format(checkpoint_dir),
                    "--pipeline_config_path={}".format(pipeline_file),
                    "--checkpoint_dir={}".format(checkpoint_dir),
                    "&"]

    num_evals = len(os.listdir(os.path.join(checkpoint_dir, "eval")))
    my_env_exc = os.environ.copy()
    eval_proc = subprocess.Popen(eval_command, env=my_env_exc)

    # Unhide GPU(s)
    if old_cuda_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_env
    else:
        del os.environ["CUDA_VISIBLE_DEVICES"]    

    train_command = [python_path, "{}".format(train_function), 
                    "--model_dir={}".format(checkpoint_dir), 
                    "--pipeline_config_path={}".format(pipeline_file),
                    "--logtostderr",
                    "--sample_1_of_n_eval_examples={}".format(samples),
                    "--checkpoint_every_n={}".format(checkpoint_steps)]

    ## SE PREPARA EL MLFLOW

    pipeline_config = config_util.get_configs_from_pipeline_file(pipeline_file)
    num_classes = config_util.get_number_of_classes(pipeline_config['model'])
    img_size = config_util.get_spatial_image_size(config_util.get_image_resizer_config(pipeline_config['model']))
    optimizer = config_util.get_optimizer_type(pipeline_config['train_config'])
    opti2 = getattr(pipeline_config['train_config'].optimizer, optimizer)
    learning_rate_type = config_util.get_learning_rate_type(getattr(pipeline_config['train_config'].optimizer, optimizer))
    initial_learning_rate = getattr(opti2.learning_rate, learning_rate_type).initial_learning_rate
    decay_steps = getattr(opti2.learning_rate, learning_rate_type).decay_steps

    label_map = pipeline_config['eval_input_config'].label_map_path
    category_index = label_map_util.create_category_index_from_labelmap(label_map, use_display_name=True)
    classes_names = []
    for idx in category_index:
        classes_names.append(category_index[idx]['name'])
    meta_architecture = pipeline_config["model"].WhichOneof("model")
    meta_architecture_config = getattr(pipeline_config["model"], meta_architecture)
    network_name = meta_architecture_config.feature_extractor.type
    mydir = os.getcwd()
    os.chdir(mlflow_path)
    mlflow.set_experiment(experiment_name)
    if run_name is not None and run_name != "":
        mlflow.set_tag("mlflow.runName", run_name)
    mlflow.log_param("network", network_name)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("img_size", img_size)
    mlflow.log_param("optimizer", optimizer)
    mlflow.log_param("learning_rate_type", learning_rate_type)
    mlflow.log_param("initial_learning_rate", initial_learning_rate)
    mlflow.log_param("decay_steps", decay_steps)
    mlflow.log_param("classes", classes_names)
    os.chdir(mydir)
    

    event_handler = Handler() 
    observer = watchdog.observers.Observer() 
    observer.schedule(event_handler, path=os.path.join(checkpoint_dir, "eval/"), recursive=True) 
    observer.start() 
    
    tb = TensorBoard(checkpoint_dir)
    tb.start()
                    
    my_env = os.environ.copy()    
    train_proc = subprocess.Popen(train_command, env=my_env)

    while train_proc.poll() is None:
        time.sleep(0.1)
        if event_handler.patience_counter == patience:
            print("*******STOPPING*********")
            train_proc.kill()
            # trained_dict= self.export_model(data, config)
            # return trained_dict
            training_dir = os.path.join(os.getcwd(), 'checkpoints')
            # eval_proc.kill()
            break
    train_proc.wait()

    time.sleep(1)

    last_event = event_handler.last_event_path
    while os.path.getsize(last_event) <= minEventSize:
        time.sleep(10)
        last_event = event_handler.last_event_path

    eval_proc.kill()
    eval_proc.wait()
    tb.stop()

    observer.stop()
    observer.join()
    event_handler.summary_values(event_handler.last_event_path)
    best_step = event_handler.step_best

    export_command = python_path + " {}".format(export_function) + " --input_type=image_tensor " + "--pipeline_config_path={} ".format(pipeline_file) + "--trained_checkpoint_dir={} ".format(os.path.join(os.getcwd(), "checkpoints")) + "--output_directory={} ".format(export_dir)
    os.system(export_command)
    shutil.make_archive(os.path.join(export_dir, 'exported_model'), 'gztar', export_dir)

    mydir = os.getcwd()
    os.chdir(mlflow_path)
    # Se registran los datos ya existentes
    readUploadTFEvents(os.path.join(checkpoint_dir, "train"), prefixTag="Train", wd=mlflow_path, bar=True)
    readUploadTFEvents(os.path.join(checkpoint_dir, "eval"), prefixTag="Eval", wd=mlflow_path, bar=True)
    mlflow.log_param("Best Step", best_step)
    mlflow.log_artifact(pipeline_file)
    mlflow.log_artifact(label_map)
    if os.path.isfile(os.path.join(export_dir,'exported_model')+".tar.gz"):
        mlflow.log_artifact(os.path.join(export_dir,'exported_model')+".tar.gz")
        os.remove(os.path.join(export_dir,'exported_model')+".tar.gz")
    os.chdir(mydir)

    if not keepFiles:
        shutil.rmtree(checkpoint_dir)
    else:
        reduce_events_size(os.path.join(checkpoint_dir, "train"))

except KeyboardInterrupt:
    os.chdir(workingDirectory)
    print("[INFO] Entrenamiento detenido Manualmente")
    if old_cuda_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_env
    print("[INFO] Reparado CUDA Env")
    if tb is not None:
        tb.stop()
    print("[INFO] Tensorboard parado")
    if eval_proc is not None:
        eval_proc.kill()
        eval_proc.wait()
    print("[INFO] Eval detenida")
    if train_proc is not None:
        train_proc.kill()
        train_proc.wait()
    print("[INFO] Training detenido")
    if observer is not None:
        observer.stop()
        observer.join()
    print("[INFO] Watchdog detenido")
    try:
        mydir = os.getcwd()
        os.chdir(mlflow_path)
        # Se registran los datos ya existentes
        readUploadTFEvents(os.path.join(checkpoint_dir, "train"), prefixTag="train", wd=mlflow_path, bar=True)
        readUploadTFEvents(os.path.join(checkpoint_dir, "eval"), prefixTag="eval", wd=mlflow_path, bar=True)
        if best_step is not None:
            mlflow.log_param("Best Step", best_step)
        mlflow.log_artifact(pipeline_file)
        mlflow.log_artifact(label_map)
        if os.path.isfile(os.path.join(export_dir,'exported_model')+".tar.gz"):
            mlflow.log_artifact(os.path.join(export_dir,'exported_model')+".tar.gz")
        os.chdir(mydir)
        print("[INFO] Datos subidos a mlflow")
        print("[INFO] Reduciendo Events")
        reduce_events_size(os.path.join(checkpoint_dir, "train"))
    except KeyboardInterrupt:
        print("[INFO] Proceso detenido, programa finalizado.")
except Exception as e:
    os.chdir(workingDirectory)
    print(str(e))
    fError = open("./error.log", "a")
    print(datetime.now().strftime('[%Y-%m-%d_%H-%M-%S]'), file=fError)
    print(str(e), file=fError)
    print("", file=fError)
    fError.close()
    if old_cuda_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_env
    if tb is not None:
        tb.stop()
    if eval_proc is not None:
        eval_proc.kill()
        eval_proc.wait()
    if train_proc is not None:
        train_proc.kill()
        train_proc.wait()
    if observer is not None:
        observer.stop()
        observer.join()
    mydir = os.getcwd()
    os.chdir(mlflow_path)
    readUploadTFEvents(os.path.join(checkpoint_dir, "train"), prefixTag="train", wd=mlflow_path, bar=True)
    readUploadTFEvents(os.path.join(checkpoint_dir, "eval"), prefixTag="eval", wd=mlflow_path, bar=True)
    if best_step is not None:
        mlflow.log_param("Best Step", best_step)
    mlflow.log_artifact(pipeline_file)
    mlflow.log_artifact(label_map)
    if os.path.isfile(os.path.join(export_dir,'exported_model')+".tar.gz"):
        mlflow.log_artifact(os.path.join(export_dir,'exported_model')+".tar.gz")
    os.chdir(mydir)

    print("[INFO] Reduciendo Events")
    reduce_events_size(os.path.join(checkpoint_dir, "train"))
os.chdir(mlflow_path)   # Se cambia el directorio para que mlflow finalice sin errores