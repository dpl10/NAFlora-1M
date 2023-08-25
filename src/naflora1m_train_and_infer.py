
#Updated 8/24/2023
#Originally wrote to run in cloud TPU in 2022.
#Updated 8/11/2022
# author: John Park

##
##      import libraries    ##
##
import tensorflow as tf
import mytflib as tfl
import tensorflow_addons as tfa
import os
from one_cycle_tf import OneCycle
import keras_efficientnet_v2
import numpy as np
import pandas as pd
from tensorflow.python.lib.io import file_io
import tqdm

## configuration

config_dict = dict()
config_dict["tfrec_structure"] = {"image":"str",
                                  "image_id":"str",
                                  "scientificName":"int",
                                 "family":"int" ,
                                  "genus":"int"
                                  }
config_dict["tfrec_shape"] =[480, 480]
config_dict["resize_resol"] =[380, 380]
config_dict["crop_ratio"] = 0.9
config_dict["n_epochs"] = 30
config_dict["aug_method"] = "standard"
config_dict["out_path"] = "/content/drive/MyDrive/naflora1m_results2/"
config_dict["max_LR"] = 7e-1
config_dict["init_LR"] = config_dict["max_LR"]/100
config_dict["wd"] = 1e-5
config_dict["N_tt_imgs"] = 210407
config_dict["class_balance_factor"] = 10

# get gsbucket address from the following link:
# https://www.kaggle.com/datasets/parkjohnychae/herbarium-2022-train-tfrec-480/code
# gsbucket address changes every week in Kaggle public dataset, make sure to update every week

trPATH = 'gs://kds-fc22994d3f9ed0cdc1ee1c5ac39051be9c2f43f8421a43be1c5f7688'
ttPATH = 'gs://kds-bc8f6ef16b3603b7ac31627f33088ce1f2319873e7ccaa2f6a6efc55'

# TPU detection / GPU detection

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # set tpu='local' for cloud TPU
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
print("Number of accelerators: ", strategy.num_replicas_in_sync)

# Functions # 
def get_model(num_classes, resize_resol, weight_name = 'imagenet21k'):
    
    base_model = keras_efficientnet_v2.EfficientNetV2S( 
        input_shape = (*resize_resol, 3),
        drop_connect_rate = 0.4,
        num_classes = 0,
        pretrained= weight_name)

    model=tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation="softmax")
 
    ])

    return model

def _load_tfrec_dataset(filenames, tfrec_format, tfrec_sizes, label_name, labeled=True, ordered=False):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 
    dataset = dataset.map(lambda Example: tfl.read_tfrecord(Example, 
                                                        TFREC_FORMAT = tfrec_format, 
                                                        TFREC_SIZES = tfrec_sizes,
                                                        LABEL_NAME = label_name))
    return dataset

def prepare_test_images(image, label, resize_factor):
    img = tf.image.central_crop(image, central_fraction = 0.9)
    img = tf.image.resize( img, size = resize_factor)
    return img, label

def get_test_ds_tfrec(LS_FILENAMES, TFREC_DICT, TFREC_SIZES, RESIZE_FACTOR, NUM_CLASSES, BATCH_SIZE, LABEL_NAME, MoreAugment = False):

    tfrec_format = tfl.tfrec_format_generator(TFREC_DICT)
    dataset = _load_tfrec_dataset(LS_FILENAMES, tfrec_format = tfrec_format, tfrec_sizes = TFREC_SIZES, 
                                  label_name = LABEL_NAME)
    dataset = dataset.map(tfl.normalize_RGB, num_parallel_calls=AUTO).prefetch(AUTO)
    dataset = dataset.map(lambda image, label: prepare_test_images(image, label, resize_factor = RESIZE_FACTOR), num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

# Load data 

tr_fns = tf.io.gfile.glob(trPATH +"/*train*.tfrec")
tt_fns = tf.io.gfile.glob(ttPATH +"/*test*.tfrec")
metadata_fns = tf.io.gfile.glob(trPATH +"/*metadata*")

# Get class weights
ls_train = tf.io.gfile.glob(trPATH+"/*.tsv")
with file_io.FileIO(ls_train[0], 'r') as f:
  df_train = pd.read_table(f)

print(ls_train)
print(df_train)
print(os.listdir())

map_name_to_cat_id = dict(zip(df_train.scientificName, df_train.category_id))
map_label_to_name = dict(zip(range(15501), sorted(set(df_train.scientificName))))
ls_Ids, mappings = tfl.ConvertLabelsToInt(df_train.scientificName)
dict_class_weights = tfl.GetDictCls(
    tfl.class_balanced_weight(ls_Ids, 
                              max_range = config_dict["class_balance_factor"] ))

config_dict["ls_train_files"] = tr_fns
config_dict["N_cls"] = len(set(df_train.scientificName))
config_dict["batch_size"] = 64*strategy.num_replicas_in_sync
STEPS_PER_EPOCH = 839772//config_dict["batch_size"]+1# int(N_tr_imgs/config_dict["batch_size"])
config_dict["steps_per_epoch"] = STEPS_PER_EPOCH


tr_ds = tfl.get_train_ds_tfrec_from_dict(config_dict, 
                                     label_name = "scientificName", 
                                     DataRepeat =True) 
batch_tr_ds = next(iter(tr_ds))

N_EPOCH = config_dict["n_epochs"]
cycle_size = N_EPOCH*STEPS_PER_EPOCH

ocLR = OneCycle(initial_learning_rate=config_dict["init_LR"],
             maximal_learning_rate=config_dict["max_LR"],
             cycle_size = cycle_size,
             shift_peak = 0.2, 
             final_lr_scale = 1e-2)

with strategy.scope():
    model = get_model(config_dict['N_cls'], config_dict['resize_resol'])
    model.compile(
        optimizer= tfa.optimizers.SGDW(learning_rate = ocLR, 
                                       weight_decay = config_dict["wd"],
                                       momentum = 0.9,
                                       nesterov = True),
        loss = tfl.SigmoidFocalCrossEntropy2(alpha = None,
                                             gamma = 0.5,
                                             reduction = tf.keras.losses.Reduction.AUTO),
         #tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1),
        metrics = tfa.metrics.F1Score(num_classes =config_dict['N_cls'])
    )

model_name = model.layers[0].get_config()['name']
print(model_name)
model.summary()

print(config_dict)

OutFileName = model_name+"_"+str(config_dict["resize_resol"][0])+"_OCEP"+str(config_dict["n_epochs"])+"_FC_CLSBW10_"+config_dict["aug_method"]

save_weights_freq = tf.keras.callbacks.ModelCheckpoint(
    filepath=OutFileName+"weights_h5",#epoch{epoch:02d}.h5", 
    verbose=1, 
    save_freq=5*config_dict["steps_per_epoch"],
    save_weights_only=True,
    save_best_only=True)

### Train ###

history = model.fit(
    tr_ds, 
    epochs= N_EPOCH, 
    steps_per_epoch = config_dict["steps_per_epoch"],
    class_weight = dict_class_weights,
    verbose=1,
  callbacks=[tfl.SaveModelHistory(config_dict,
                               OutFileName,
                               config_dict["out_path"]),
                               save_weights_freq,
              tf.keras.callbacks.TerminateOnNaN()])
config_dict["out_path"] = os.path.join(config_dict["out_path"], OutFileName+"_weights.h5")
model.save_weights(config_dict["out_path"]) 

### INFERENCE ###

tt_fns = tf.io.gfile.glob(ttPATH + '/*test*.tfrec')
print('Dataset: {} test images'.format(config_dict["N_tt_imgs"]))
test_dict = {"image":"str", "image_id":"int"}

AUTO = tf.data.experimental.AUTOTUNE

tt_ds = get_test_ds_tfrec( LS_FILENAMES = tt_fns,
                              TFREC_DICT = test_dict,
                              TFREC_SIZES =  config_dict["tfrec_shape"],
                              RESIZE_FACTOR = config_dict["resize_resol"],
                              NUM_CLASSES = config_dict["N_cls"],
                              BATCH_SIZE = config_dict["batch_size"],
                            LABEL_NAME = "image_id"
                          )

test_images_ds = tt_ds.map(lambda image, idnum: image)
test_Ids_ds = tt_ds.map(lambda image, idnum: idnum)
predictions = np.zeros(config_dict["N_tt_imgs"], dtype=np.int32)

for i, image in tqdm(enumerate(test_images_ds), total= (config_dict["N_tt_imgs"]//config_dict["batch_size"] + 1)):
    idx1 = i*config_dict["batch_size"]
    if (idx1 + config_dict["batch_size"]) > config_dict["N_tt_imgs"]:
        idx2 = config_dict["N_tt_imgs"]
    else:
        idx2 = idx1 + config_dict["batch_size"]
    predictions[idx1:idx2] = np.argmax(model.predict_on_batch(image), axis=-1)

predict_image_nums = np.zeros(config_dict["N_tt_imgs"], dtype=np.int32)

for i, image_nums in tqdm(enumerate(test_Ids_ds), total= (config_dict["N_tt_imgs"]//config_dict["batch_size"] + 1)):
    idx1 = i*config_dict["batch_size"]
    if (idx1 + config_dict["batch_size"]) > config_dict["N_tt_imgs"]:
        idx2 = config_dict["N_tt_imgs"]
    else:
        idx2 = idx1 + config_dict["batch_size"]
    predict_image_nums[idx1:idx2] = image_nums

prediction_cat_id = [map_name_to_cat_id[map_label_to_name[ele]] for ele in predictions]

pd.DataFrame({"Id":predict_image_nums,"Predicted":prediction_cat_id}).to_csv(os.path.join(config_dict["out_path"],OutFileName+"_submissions.csv"),index=False)