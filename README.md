# NAFlora-1M
NAFlora-1M: continental-scale high-resolution fine-grained plant classification dataset

## Updates
August 25th, 2023:
  * Dataset Construction
June 14th, 2023: 
  * Initialized repository

## Dataset Construction
In field expeditions for plant diversity research, plants are collected, pressed, and dried in the field. Once they arrive at research facilities they are identified by researchers—usually specialists in a particular flora or group of plants. After species identification, each specimen is mounted on blank archival herbarium paper. As part of the worldwide effort of to bring biological collection online for biodiversity research, the botanical community has already digitized 24 million plant specimens representing 93,000 species and made them publicly available. The plant specimen images obtained from each source institution are created following strict, standardized protocols~\citep{thiers2016digitization}. 

Data collection for NAFlora1M involved downloading these "digitized" specimen records and images. A complete updated list of North American (Canada, Greenland, and United States of America) vascular plants was obtained from the [Checklist of the Vascular Plants of the Americas](https://www.nature.com/articles/s41597-021-00997-6) (CVPA). Records for the 17,041 vascular plant species were then reterived from the two largest public biodiversity aggregators — [GBIF](https://www.gbif.org/)(Global Biodiversity Information) and [iDigBio](https://www.idigbio.org/) — with names standardizing via [World Checklist of Vascular Plants](https://www.nature.com/articles/s41597-021-00997-6)(WCVPv5). In total, we found 8,776,687 images belonging the 17,041 vascular plant species. The choice of plant species was based on the number of images available ($10 \leq n$). From the 8.7M available images, we selected records to maximize the number of species with at least 10 images while at the same time minimizing the number of download servers (to restrict the amount of custom download code required). We selected 15 download servers which provide images for 54 herbaria. The number of images per species was capped at 100. For species with more than 100 images available, we randomly selected 100 images to download (random replacements were selected for non-functional URLs). The dataset was randomly partitioned into an 80/20% split for training/testing. We made sure that at least two images for each class were included in the test partition. 

Finally, as a post-processing step, we removed text information from all images to ensure models could not use it to identify the plants and to nullify any privacy concerns. The [Character Region Awareness for Text](https://arxiv.org/pdf/1904.01941.pdf) (CRAFT) model was employed to draw bounding boxes around areas containing text. Uniform noise was added within each bounding box and the pixels were then passed through a gaussian blur filter. This process ensured that any text in the image that may reveal metadata was obfuscated from the model—forcing the model to focus on plant specimen differences rather than learning to read. Images were then resized to 1,000 pixels in the largest dimension. In the end, **NAFlora-1M** had 1,050,179 images from 54 institutions representing 90.06% of the species known from North America (15,501 species). A 80/20 split was applied resulting in 839,772 training images and 210,407 testing images. We constructed a mini dataset, NAFlora-mini, for internal testing purposes: it consists of 1,368 species, 45,234 training images, and 13,231 testing images—with a similar imbalanced structure as NAFlora-1M. More specifics about dataset construction are detailed in the supplementary.

## Kaggle competition
NAFlora-1M was benchmarked in the Herbarium 2022: _The flora of North America_ [Kaggle competition](https://www.kaggle.com/competitions/herbarium-2022-fgvc9).

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

