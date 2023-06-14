# NAFlora-1M
NAFlora-1M: continental-scale high-resolution fine-grained plant classification dataset

## Updates
June 14th, 2023: 
  * Initialized repository

## Kaggle competition
NAFlora-1M was benchmarked in the Herbarium 2022: The flora of North America [Kaggle competition](https://www.kaggle.com/competitions/herbarium-2022-fgvc9).

## Details
There are a total of 15,501 vascular species in the dataset, with 800k training images, 200k test images. We show the top-10 and the bottom-10 families ordered in terms of number of species-level diversity

| Family |	Number of Species	| Train Images |	Test Images |
|------|---------------|-------------|---------------|
1|2,917|118,800|8,751|
2|2,031|87,192|6,093|
3|1,258|143,950|3,774|
4|369|7,835|1,107|
5|321|6,864|963|
6|284|22,754|852|
7|262|8,007|786|
8|234|20,104|702|
9|178|5,966|534|
10|144|11,156|432|
|------|---------------|-------------|---------------|
Total|15.5k|800k|200k|

* This is a placeholder for species-level distribution 

## How to access the data 

* This section specifies details on about how to access the [data](https://www.kaggle.com/competitions/herbarium-2022-fgvc9/data).

### Links

* Training and test images - high resolution [163.17GB]
  * All images are resized so that the longest edge is 1000 px  
  * [Training images - 1000 px](https://www.kaggle.com/competitions/herbarium-2022-fgvc9/data?select=train_images)
  * [Test images - 1000 px](https://www.kaggle.com/competitions/herbarium-2022-fgvc9/data?select=test_images)
* Training and test images - adjusted resolution [88.61GB]
  * Images are resized to 480x480, prepared in [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord) format.
  * [Training images - 480px [70GB]](https://www.kaggle.com/datasets/parkjohnychae/herbarium-2022-train-tfrec-480)
  * [Test images - 480px [18GB]](https://www.kaggle.com/datasets/parkjohnychae/herbarium-2022-test-tfrec-480)
* Train metadata [link](https://www.kaggle.com/competitions/herbarium-2022-fgvc9/data?select=train_metadata.json)
* Test metadata [link](https://www.kaggle.com/competitions/herbarium-2022-fgvc9/data?select=test_metadata.json)
  

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

*placeholder for terms of usage

## Pretrained Models

*placeholder for pretrained models

