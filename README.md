# NAFlora-1M
NAFlora-1M: continental-scale high-resolution fine-grained plant classification dataset

## Updates
August 25th, 2023:
  * Overview
  * Training script
 
June 14th, 2023: 
  * Initialized repository

## Overview
In botany, a ‘flora’ is a complete account of the plants found in a geographic region. The dichotomous keys and detailed descriptions of diagnostic morphological features contained within a flora are used by botanists to determine which names to apply to plant specimens. This year's competition dataset aims to encapsulate the flora of North America so that we can test the capability of artificial intelligence to replicate this traditional tool —a crucial first step to harnessing AI’s potential botanical applications.

**NAFlora-1M** dataset comprises 1.05 M images of 15,501 vascular plants, which constitute more than 90% of the taxa documented in North America. Our dataset is constrained to include only vascular land plants (lycophytes, ferns, gymnosperms, and flowering plants).

Our dataset has a long-tail distribution. The number of images per taxon is as few as seven and as many as 100 images. Although more images are available, we capped the maximum number in an attempt to ensure sufficient but manageable training data size.

## Training 

```bash 
python3 src/naflora1m_train_and_infer.py

-------------------------------------------------------------------

/usr/local/lib/python3.10/dist-packages/keras/initializers/initializers.py:120: UserWarning: The initializer VarianceScaling is unseeded and being called multiple times, which will return identical values each time (even if the initializer is unseeded). Please update your code to provide a seed to the initializer, or avoid using the same initalizer instance more than once.
  warnings.warn(
Downloading data from https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-s-21k.h5
194646348/194646348 [==============================] - 1s 0us/step
>>>> Load pretrained from: /root/.keras/models/efficientnetv2/efficientnetv2-s-21k.h5
EfficientNetV2S
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 EfficientNetV2S (Functional  (None, 12, 12, 1280)     20331360  
 )                                                               
                                                                 
 global_average_pooling2d (G  (None, 1280)             0         
 lobalAveragePooling2D)                                          
                                                                 
 dropout (Dropout)           (None, 1280)              0         
                                                                 
 dense (Dense)               (None, 1024)              1311744   
                                                                 
 dropout_1 (Dropout)         (None, 1024)              0         
                                                                 
 dense_1 (Dense)             (None, 15501)             15888525  
                                                                 
=================================================================
Total params: 37,531,629
Trainable params: 37,377,757
Non-trainable params: 153,872
_________________________________________________________________

grab config info
done - saving config info to ./EfficientNetV2S_380_OCEP30_FC_CLSBW10_None_configs.json
model summary saved to EfficientNetV2S_380_OCEP30_FC_CLSBW10_None_model_summary.txt. initialization is done
{'name': 'SGDW', 'learning_rate': {'class_name': 'OneCycle', 'config': {'initial_learning_rate': 0.006999999999999999, 'maximal_learning_rate': 0.7, 'cycle_size': 49230, 'scale_mode': 'cycle', 'shift_peak': 0.2}}, 'decay': 0.0, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 1e-05, 'exclude_from_weight_decay': None}
Epoch 1/30
   6/1641 [..............................] - ETA: 20:16 - loss: 84.9471 - f1_score: 0.0000e+00WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0074s vs `on_train_batch_end` time: 28.2103s). Check your callbacks.
WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0074s vs `on_train_batch_end` time: 28.2103s). Check your callbacks.
1641/1641 [==============================] - 1491s 743ms/step - loss: 6.2415 - f1_score: 0.0019 - time: 1490.8735
Epoch 2/30
1641/1641 [==============================] - 1222s 745ms/step - loss: 3.1388 - f1_score: 0.1323 - time: 1221.8033
Epoch 3/30
1641/1641 [==============================] - 1224s 746ms/step - loss: 2.2029 - f1_score: 0.3254 - time: 1223.6055
Epoch 4/30
1641/1641 [==============================] - 1225s 746ms/step - loss: 1.8870 - f1_score: 0.4320 - time: 1224.5351
Epoch 5/30
```

## Details
There are a total of 15,501 vascular species in the dataset, with 800k training images, 200k test images. We show the top-10 families ordered in terms of species-level diversity.

| Family |	Number of Species	| Train Images |	Test Images |
|------|---------------|-------------|---------------|
Asteraceae|1,998|110,007| 27,605 |
Fabaceae|1,070|59,152| 14,803 |
Poaceae|964|53,547| 13,399 |
Cyperaceae|780|45,447| 11,410|
Boraginaceae|454|23,724| 5,948|
Brassicaceae|402|19,033| 4,752|
Plantaginaceae|380|21,054| 5,265|
Polygonaceae|359|18,899| 4,714|
Rosaceae|356|20,628| 5,165|
Laminaceae|309|16,854| 4,239|
|___|___|___|___|
Top-10 total|7,072|388,345|97,300|

 
## How to access the data 

* This section specifies details on about how to access the [data](https://www.kaggle.com/competitions/herbarium-2022-fgvc9/data).

### Links

* Training and test images - high resolution [163.17GB]
  * All images are resized so that the longest edge is 1000 px  
  * [Training images - 1000 px [119GB]](https://www.kaggle.com/competitions/herbarium-2022-fgvc9/data?select=train_images)
  * [Test images - 1000 px [44GB]](https://www.kaggle.com/competitions/herbarium-2022-fgvc9/data?select=test_images)
* Training and test images - adjusted resolution [88.61GB]
  * Images are resized to 480x480, and formated in [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord).
  * [Training images - 480px [70GB]](https://www.kaggle.com/datasets/parkjohnychae/herbarium-2022-train-tfrec-480)
  * [Test images - 480px [18GB]](https://www.kaggle.com/datasets/parkjohnychae/herbarium-2022-test-tfrec-480)
* [Train metadata [667MB]](https://www.kaggle.com/competitions/herbarium-2022-fgvc9/data?select=train_metadata.json)
* [Test metadata [23MB]](https://www.kaggle.com/competitions/herbarium-2022-fgvc9/data?select=test_metadata.json)
  

## Kaggle competition
NAFlora-1M was benchmarked in the Herbarium 2022: _The flora of North America_ [Kaggle competition](https://www.kaggle.com/competitions/herbarium-2022-fgvc9).

### Annotation Format
We follow the annotation format of the [COCO dataset](http://mscoco.org/dataset/#download) and add additional fields. The annotations are stored in the [JSON format](http://www.json.org/) and are organized as follows:
```
{ 
  "annotations" : [annotation],
  "categories" : [category],
  "genera" : [genus]
  "images" : [image],
  "distances" : [distance],
  "licenses" : [license],
  "institutions" : [institution]
}


annotation {
  "image_id" : int,
  "category_id" : int,
  "genus_id" : int,
  "institution_id" : int   
}

image {
  "image_id" : int,
  "file_name" : str,
  "license" : int
}

category {
  "category_id" : int, 
  "scientificName" : str,
  # We also provide a super-category for each species.
  "authors" : str, # correspond to 'authors' field in the wcvp
  "family" : str, # correspond to 'family' field in the wcvp
  "genus" : str, # correspond to 'genus' field in the wcvp
  "species" : str, # correspond to 'species' field in the wcvp
}

genera {
  "genus_id" : int,
  "genus" : str
}

distance {
  # We provide the pairwise evolutionary distance between categories (genus_id0 < genus_id1). 
  "genus_id_x" : int,    
  "genus_id_y" : int,    
  "distance" : float
}

institution {
  "institution_id" : int
  "collectionCode" : str
}

license {
  "id" : int,
  "name" : str,
  "url" : str
}
```

### Evaluation through late submission

It is possible to get performance metric for our test data through the [submssions page](https://www.kaggle.com/competitions/herbarium-2022-fgvc9/submissions)

The submission format for the Kaggle competition is a csv file with the following format:
```
Id,predicted
12345,0 
67890,83 
```
The `Id` column corresponds to the test image id. The `predicted` column corresponds to 1 category id, for scientificName (species).

## Terms of Use

* CC BY-NC-ND-4.0: Commerical use of the data and pre-trained model is restricted.

## Pretrained Models

* Pretrained models and sample code will soon be released.

