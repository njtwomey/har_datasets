# Introduction

This repository aims to provide a unified interface to wearable-based Human Activity Recognition (HAR) datasets. The philosophy is to acquire many datasets from a wide variety of recording conditions and to translate these into a consistent data format in order to more easily address open questions on feature extraction/representation learning, meta/transfer learning, active learning amongst other tasks. Ultimately, I am to create a home for the easier understanding of the stability, strengths and weaknesses of the state-of-the-art in HAR. 

# Setup

## Virtual environment

It is good practise to use virtual environments when using this. I have recently been using [miniconda](https://docs.conda.io/en/latest/miniconda.html) as my python management system. It works exactly like anaconda. The following commands create a new environment, activates it and installs the requirements to that environment.

```bash 
pipenv install --python 3.8 --skip-lock --dev
pipenv shell
pre-commit install
```

## dotenv 

Several global variables are required for this library to work. I set these up with the [dotenv](https://pypi.org/project/python-dotenv/) library. This searches for a file called `.env` that should be found in the project root. It then loads environment variables called `PROJECT_ROOT`, `ZIP_ROOT` and `BUILD_ROOT`. In my system, these are set up roughly as follows. 

```bash 
export PROJECT_ROOT = "/users/username/workspace/har_datasets"
export ZIP_ROOT = "/users/username/workspace/har_datasets/data/zip"
export BUILD_ROOT = "/users/username/workspace/har_datasets/data/build"
```

# Data Format

The data from all datasets listed in this project are converted into one consistent format that consistes of four key elements: 

1. the train/validation/test fold definition file; 
2. the label file; 
3. the data file; and
4. an index file. 

Note, the serialisation format used in this repository is that data are stored on a per-sample basis. This means that each of the files listed above will have the same number of rows. 

## Index File

The following columns are required for the index file:

```
subject, trail, time
```

`subject` defines a subject identifier, `trial` allows for different trials to be specified (eg it can distinguish data from subjects who perform a task several times), and `time` defines the time (absolute or relative). Subject and trial should be integers, but need not be contiguous. Although time can be considered unnecessary in many applications (especially if the recording was done in a controlled environment or following a script) it is added here to allow for the detection of missing data (missing time stamps) and time-of-day features (if `time` represents epoch time, for example).

This file must have three columns only. 

## Task Files

The following structure is required for the task files

```
label_vals
```

This file must have at least one column. In general, it is expected that the column will be a list of strings (where the string corresponds to the target). This is not a requirement, however, and the label values may be vector-valued. It is important that the correct model and evaluation criteria are associated with the task. 

## Data File

The data format is quite simple:

```
x, y, z
```

where `x`, `y` and `z` correspond to the axes of the wearable. By default different files are created for each modality (ie accelerometer, gyroscope and magnetomoter) and for each location (eg wrist, waist). For example, if one accelerometer is on the wrist a file called `accel-wrist` will be created for it. There is no restriction on the number of colums in this file, but we expect that more often than not 3 columns will be present for each axis of the device. 

This file must have at least one column. 

## Fold Definitions

Train and test folds are defined by the columns of this file: 

```python
fold_1
-1
-1
-1
0
0
0
1
1
1
```

The behaviour of these folds is based on scikit-learn's [PredefinedSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html) module. Additional folds can (if necessary) be defined by adding supplementary columns to this file. For example if doing 10 times 10-fold cross validation, 10 fold identifiers would be contained in each of the 10 columns. 

This file must have at least one column. 

Several special fold definitions are also supported. `LOSO` performs leave one subject out cross validation, and `deployable` learns models on all of the data with the expectation that this model is to be deployed outside of the scope of the pipeline that created it. 

# Contributing

I hope to receive pull requests for new datasets, processing methods, features, and models to this repository. Requests are likely to be accepted once the exact data format, feature extraction, modelling and evaluation interfaces are relatively stable. 

## Contributing Datasets

1. Create a new [yaml](https://en.m.wikipedia.org/wiki/YAML) file in the `metadata/datasets` directory and fill out the information as accurately as possible. Follow the styles and detail given in the entries named `anguita2013`, `pamap2` and `uschad`. The entry of accurate metadata will be heavily strictly moderated before a submission is accepted. Note:
    - The name of the file and the `name` filed in the yaml file dataset name must be lower case.
    - List all sensor modalities in the dataset in the `modalities` field. The modality names should be consistent with the values found in `metadata/modality.yaml`.
    - List all sensor placements in the dataset in the `placements` field  The placement names should be consistent with the values found in `metadata/placement.yaml`.
    - List all outputs in the dataset in the `sources` field. For example, if a data source arrives from an accelerometer placed on the wrist, a dict entry like `{"placement": "wrist", "modality": "accel"}`. This can be tedious, but there is great value in doing this. 
    - If the dataset introduces a new task, add a new file to the `metadata/tasks/<task-name>.yaml` file. List all new target names in this file (see `metadata/tasks/har.yaml` for example). 
    - If the dataset introduces a new target to an existing task, add it to the end of `tasks/<task-name>.yaml`. 
    - If the sensor has been placed on a new location add it to the end of `metadata/placement.yaml`. 
    - If the sensor is of a new modality, add it to the end of `metadata/modality.yaml`. 
2. Run `make table`. This will update the dataset table in the `tables` directory. Ensure this command executes successully and verify that the entered information is accurate.
3. Run `make data`. This will download the archive automatically based on the URLs provided in the `download_urls` field from step 1 above. 
4. Copy the file `src/datasets/__new__.py` to `src/datasets/<dataset-name>.py` (`<dataset-name>` is defined by #1 above). The prupose of this file is to translate the data to the expected format described in the sections above. In particular, separate files with the wearable data, annotated labels, pre-defined folds, and index files are required. Use the existing examples of the aforementioned datasets (`anguita2013`, `pamap2` and `uschad`) that can be found in `src/datasets` as examples of how this has been achieved. 

## Contributing Pipelines

(Under construction. See `examples/basic_har.py` for basic examples.)

## Contributing Models

(Under construction. See `src/models/sklearn/basic.py` for basic examples.)


# Datasets

The following table enumerates the datasets that are under consideration for inclusion in this repository.

| First Author | Dataset Name | Paper (URL) | Data Description (URL) | Data Download (URL) | Year | fs | Accel | Gyro | Mag | #Subjects | #Activities | Notes | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Banos | banos2012 | [A benchmark dataset to evaluate sensor displacement in activity recognition](http://www.orestibanos.com/paper_files/banos_ubicomp_2012.pdf) | [Description](http://archive.ics.uci.edu/ml/datasets/REALDISP+Activity+Recognition+Dataset) | [Download](http://archive.ics.uci.edu/ml/machine-learning-databases/00305/realistic_sensor_displacement.zip) | 2012 | 50 | yes | yes | yes | 17 | 33 |  |
| Banos | banos2015 | [mHealthDroid: a novel framework for agile development of mobile health applications](https://link.springer.com/chapter/10.1007/978-3-319-13105-4_14) | [Description](http://archive.ics.uci.edu/ml/datasets/mhealth+dataset) | [Download](http://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip) | 2015 | 50 | yes | yes | yes | 10 | 12 |  |
| Barshan | barshan2014 | [Recognizing daily and sports activities in two open source machine learning environments using body-worn sensor units](https://ieeexplore.ieee.org/abstract/document/8130901/) | [Description](https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities) | [Download](https://archive.ics.uci.edu/ml/machine-learning-databases/00256/data.zip) | 2014 | 25 | yes | yes | yes | 8 | 19 |  |
| Bruno | bruno2013 | [Analysis of Human Behavior Recognition Algorithms based on Acceleration Data](https://www.researchgate.net/profile/Barbara_Bruno2/publication/261415865_Analysis_of_human_behavior_recognition_algorithms_based_on_acceleration_data/links/53d001320cf25dc05cfca025.pdf) | [Description](DescriptionURL) | [Download](DownloadURL) | 2013 | 32 | yes |  |  | 16 | 14 | Notes |
| Casale | casale2015 | [Personalization and user verification in wearable systems using biometric walking patterns](https://dl.acm.org/citation.cfm?id=2339117) |  |  | 2012 | 52 | yes |  |  | 7 | 15 |  |
| Chen | utdmhad | [UTD-MHAD: A Multimodal Dataset for Human Action Recognition Utilizing a Depth Camera and a Wearable Inertial Sensor](https://ieeexplore.ieee.org/abstract/document/7350781) | [Description](https://www.utdallas.edu/~kehtar/UTD-MHAD.html) | [Download](http://www.utdallas.edu/~kehtar/UTD-MAD/Inertial.zip) | 2015 | 50 | yes | yes | | 9 | 21 |  |
| Chavarriaga | opportunity | [The Opportunity challenge: A benchmark database for on-body sensor-based activity recognition](https://www.sciencedirect.com/science/article/pii/S0167865512004205) | [Description](https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition) | [Download](https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip) | 2012 | 30 | yes | yes | yes | 12 | 7 | Several annotation tracks. |
| Chereshnev | hugadb | [HuGaDB: Human Gait Database for Activity Recognition from Wearable Inertial Sensor Networks](https://link.springer.com/chapter/10.1007/978-3-319-73013-4_12) | [Description](https://github.com/romanchereshnev/HuGaDB) | [Download](https://www.dropbox.com/s/7nb9g650i5m9k6c/HuGaDB.zip?dl=0) | 2017 | ~56 | yes | yes |  | 18 | 12 |  |
| Kwapisz | wisdm | [Activity Recognition using Cell Phone Accelerometers](http://www.cis.fordham.edu/wisdm/includes/files/sensorKDD-2010.pdf) | [Description](http://www.cis.fordham.edu/wisdm/dataset.php) | [Download](http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz) | 2012 | 20 | yes |  |  | 29 | 6 |  |
| Micucci | micucci2017 | [UniMiB SHAR: A Dataset for Human Activity Recognition Using Acceleration Data from Smartphones](https://www.mdpi.com/2076-3417/7/10/1101/html) | [Description](http://www.sal.disco.unimib.it/technologies/unimib-shar/) | [Download](https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip?dl=0) | 2017 | 50 | yes |  |  | 30 | 8 | Notes |
| Ortiz | ortiz2015 | [Human Activity Recognition on Smartphones with Awareness of Basic Activities and Postural Transitions](https://link.springer.com/chapter/10.1007/978-3-319-11179-7_23) | [Description](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based%20Recognition%20of%20Human%20Activities%20and%20Postural%20Transitions) | [Download](http://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip) | 2015 | 50 | yes | yes |  | ? | 7 | With postural transitions |
| Shoaib | shoaib2014 | [Fusion of Smartphone Motion Sensors for Physical Activity Recognition](https://www.mdpi.com/1424-8220/14/6/10146) | [Description](https://www.researchgate.net/publication/266384007_Sensors_Activity_Recognition_DataSet) | [Download](https://www.researchgate.net/profile/Muhammad_Shoaib20/publication/266384007_Sensors_Activity_Recognition_DataSet/data/542e9d260cf277d58e8ec40c/Sensors-Activity-Recognition-DataSet-Shoaib.rar) | 2014 | 50 | yes | yes | yes | 7 | 7 |  |
| Siirtola | siirtola2012 | [Recognizing human activities user-independently on smartphones based on accelerometer data](https://dialnet.unirioja.es/servlet/articulo?codigo=3954593) | [Description](http://www.oulu.fi/bisg/node/40364) | [Download](http://www.ee.oulu.fi/research/neurogroup/opendata/OpenHAR.zip) | 2012 | 40 | yes | | | 7 | 5 |  |
| Stisen | stisen2015 | [Smart Devices are Different: Assessing and MitigatingMobile Sensing Heterogeneities for Activity Recognition](https://dl.acm.org/citation.cfm?id=2809718) | [Description](https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition) | [Download](https://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip) | 2015 | 50-200 | yes |  |  | 9 | 6 |  |
| Sztyler | sztyler2016 | [On-body localization of wearable devices: An investigation of position-aware activity recognition](https://ieeexplore.ieee.org/document/7456521) | [Description](http://sensor.informatik.uni-mannheim.de/index.html#dataset_realworld) | [Download](http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip) | 2016 | 50 | yes | yes | yes | 15 | 8 | Many other sensors also (video, light, sound, etc) |
| Twomey | spherechallenge | [The SPHERE Challenge: Activity Recognition with Multimodal Sensor Data](https://arxiv.org/abs/1603.00797) | [Description](https://data.bris.ac.uk/data/dataset/8gccwpx47rav19vk8x4xapcog) | [Download](https://data.bris.ac.uk/datasets/8gccwpx47rav19vk8x4xapcog/8gccwpx47rav19vk8x4xapcog.zip) | 2016 | 20 | yes | | | 20 | 20 |  |
| Ugulino | ugulino2012 | [Wearable Computing: Accelerometers’ Data Classification of Body Postures and Movements](https://link.springer.com/chapter/10.1007/978-3-642-34459-6_6) | [Description](http://groupware.les.inf.puc-rio.br/har) | [Download](http://groupware.les.inf.puc-rio.br/static/har/SystematicReview-RIS-Format.zip) | 2012 | 50 | yes |  |  | 4 | 5 |  |
| Vavoulas | mobiact | [The MobiAct Dataset: Recognition of Activities of Daily Living using Smartphones](http://www.scitepress.org/Papers/2016/57924/57924.pdf) | [Description](https://bmi.teicrete.gr/en/the-mobifall-and-mobiact-datasets-2/) | [Download](https://drive.google.com/file/d/0B5VcW5yHhWhielo5NTk1Q3ZiWDQ/) | 2016 | 100 | yes |  |  | 57 | 9 |  |






# Project Structure

This project follows the [DataScience CookieCutter](https://drivendata.github.io/cookiecutter-data-science/) template with the aim of facilitating reproducible models and results. the majority of commands are executed with the `make` command, and we also provide a high-level data loading interface.

