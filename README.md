# HAR Datasets

This repository aims to provide a unified interface to datasets for the task of accelerometer-based Human Activity Recognition (HAR). The philosophy is to catalogue as many datasets as possible from a wide variety of recording conditions.

 terms of data format, feature extraction, label space, sampling frequency, device location/orientation, etc. for the purpose of understanding the efficacy of transfer learning, online learning, lifelong learning,  data representation, feature extraction across a large collection of datasets.


# Project Structure

This project follows the [DataScience CookieCutter](https://drivendata.github.io/cookiecutter-data-science/) template with the aim of facilitating reproducible models and results. the majority of commands are executed with the `make` command, and we also provide a high-level data loading interface.

# Proposed Format

All data will be translated to the following a simple CSV format with the following columns:

```
time, subject_id, sequence_id, activity_labels, fold_id, x, y, z
```

where `time` is in seconds and of the type double, `subject_id` is an integer identifier of the subjects, `sequence_id` identifiers of contiguous activities (one subject may therefore perform a task several times), `x`, `y`, `z` are the x, y, and z axis data (whether acceleration, magnetometer, or gyroscope), and `activity_labels` are the labels of the dataset. Finally, `fold_id` is an identifier that is used to specify the fold in which the data should appear (negative values will only be used in training, consistent with scikit-learn's [PredefinedSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html) module).

We have made the decision to keep our data formal relatively simple since we hope it will provide a language-agnostic interface to the data so that users of, for example, R, MATLAB, Python, C++, etc can use the data once it has been built. Datasets with several views into movement (eg with volunteers wearing several devices, or with IMUs providing not only acceleration data, but also gyroscope and magnetometer data) we have made the decision that each 'view' be contained in a separate file since in some cases the data are sampled at different rates. However, the data may be merged together using the `subject_id`, `time`, and `sequence_id` fields of the file above.

# Current Datasets

The following table enumerates the datasets accounted for in this repository, sorted by the surname of the first author of the paper.

| First Author | Dataset Name | Paper (URL) | Data Description (URL) | Data Download (URL) | Year | fs | Accel | Gyro | Mag | #Subjects | #Activities | Notes | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Anguita | anguita2013 | [A Public Domain Dataset for Human Activity Recognition Using Smartphones](https://pdfs.semanticscholar.org/83de/43bc849ad3d9579ccf540e6fe566ef90a58e.pdf) | [Description](http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) | [Download]() | 2013 | 50 | yes | yes | | 30 | 6 |  |
| Banos | mhealth | [mHealthDroid: A Novel Framework for Agile Development of Mobile Health Applications](https://link.springer.com/chapter/10.1007/978-3-319-13105-4_14) | [Description](http://archive.ics.uci.edu/ml/datasets/MHEALTH+Dataset) | [Download](http://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip) | 2014 | 50 | yes | yes | yes | 10 | 12 |
| Casale | casale2015 | [Personalization and user verification in wearable systems using biometric walking patterns](https://dl.acm.org/citation.cfm?id=2339117) |  |  | 2012 | 52 | yes |  |  | 7 | 15 |  |
| Chen | utdmhad | [UTD-MHAD: A Multimodal Dataset for Human Action Recognition Utilizing a Depth Camera and a Wearable Inertial Sensor](https://ieeexplore.ieee.org/abstract/document/7350781) | [Description](https://www.utdallas.edu/~kehtar/UTD-MHAD.html) | [Download](http://www.utdallas.edu/~kehtar/UTD-MAD/Inertial.zip) | 2015 | 50 | yes | yes | | 9 | 21 |  |
| Chereshnev | hugadb | [HuGaDB: Human Gait Database for Activity Recognition from Wearable Inertial Sensor Networks](https://link.springer.com/chapter/10.1007/978-3-319-73013-4_12) | [Description](https://github.com/romanchereshnev/HuGaDB) | [Download](https://www.dropbox.com/s/7nb9g650i5m9k6c/HuGaDB.zip?dl=0) | 2017 | ~56 | yes | yes |  | 18 | 12 |  |
| Koskimaki | koskimaki2014 | [Recognizing Gym Exercises Using Acceleration Data from Wearable Sensors](https://www.researchgate.net/profile/Heli_Koskimaeki/publication/287208424_Recognizing_gym_exercises_using_acceleration_data_from_wearable_sensors/links/56bdb5b508ae3b4ebe8b0a64.pdf) | [Description](http://www.oulu.fi/cse/node/23065) | [Download](http://www.ee.oulu.fi/research/neurogroup/opendata/MyoGymData.zip) | PublicationYear | SamplingFrequency | HasAccelerometer | HasGyroscope | HasMagnetometer | NumSubjects | NumActivities | Gym activities |
| Kwapisz | wisdm | [Activity Recognition using Cell Phone Accelerometers](http://www.cis.fordham.edu/wisdm/includes/files/sensorKDD-2010.pdf) | [Description](http://www.cis.fordham.edu/wisdm/dataset.php) | [Download](http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz) | 2012 | 20 | yes |  |  | 29 | 6 |  |
| Micucci | micucci2017 | [UniMiB SHAR: A Dataset for Human Activity Recognition Using Acceleration Data from Smartphones](https://www.mdpi.com/2076-3417/7/10/1101/html) | [Description](http://www.sal.disco.unimib.it/technologies/unimib-shar/) | [Download](https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip?dl=0) | 2017 | 50 | yes |  |  | 30 | 8 | Notes |
| Ortiz | ortiz2015 | [Human Activity Recognition on Smartphones with Awareness of Basic Activities and Postural Transitions](https://link.springer.com/chapter/10.1007/978-3-319-11179-7_23) | [Description](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based%20Recognition%20of%20Human%20Activities%20and%20Postural%20Transitions) | [Download](http://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip) | 2015 | 50 | yes | yes |  | ? | 7 | With postural transitions |
| Reiss | pamap2 | [Introducing a new benchmarked dataset for activity monitoring](https://ieeexplore.ieee.org/abstract/document/6246152) | [Description](http://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring) | [Download](http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip) | 2012 | 100 | yes | yes | yes | 10 | 12 |  |
| Shoaib | shoaib2014 | [Fusion of Smartphone Motion Sensors for Physical Activity Recognition](https://www.mdpi.com/1424-8220/14/6/10146) | [Description](https://www.researchgate.net/publication/266384007_Sensors_Activity_Recognition_DataSet) | [Download](https://www.researchgate.net/profile/Muhammad_Shoaib20/publication/266384007_Sensors_Activity_Recognition_DataSet/data/542e9d260cf277d58e8ec40c/Sensors-Activity-Recognition-DataSet-Shoaib.rar) | 2014 | 50 | yes | yes | yes | 7 | 7 |  |
| Siirtola | siirtola2012 | [Recognizing human activities user-independently on smartphones based on accelerometer data](https://dialnet.unirioja.es/servlet/articulo?codigo=3954593) | [Description](http://www.oulu.fi/bisg/node/40364) | [Download](http://www.ee.oulu.fi/research/neurogroup/opendata/OpenHAR.zip) | 2012 | 40 | yes | | | 7 | 5 |  |
| Stisen | stisen2015 | [Smart Devices are Different: Assessing and MitigatingMobile Sensing Heterogeneities for Activity Recognition](https://dl.acm.org/citation.cfm?id=2809718) | [Description](https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition) | [Download](https://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip) | 2015 | 50-200 | yes |  |  | 9 | 6 |  |
| Sztyler | sztyler2016 | [On-body localization of wearable devices: An investigation of position-aware activity recognition](https://ieeexplore.ieee.org/document/7456521) | [Description](http://sensor.informatik.uni-mannheim.de/index.html#dataset_realworld) | [Download](http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip) | 2016 | 50 | yes | yes | yes | 15 | 8 | Many other sensors also (video, light, sound, etc) |
| Twomey | spherechallenge | [The SPHERE Challenge: Activity Recognition with Multimodal Sensor Data](https://arxiv.org/abs/1603.00797) | [Description](https://data.bris.ac.uk/data/dataset/8gccwpx47rav19vk8x4xapcog) | [Download](https://data.bris.ac.uk/datasets/8gccwpx47rav19vk8x4xapcog/8gccwpx47rav19vk8x4xapcog.zip) | 2016 | 20 | yes | | | 20 | 20 |  |
| Vavoulas | mobiact | [The MobiAct Dataset: Recognition of Activities of Daily Living using Smartphones](http://www.scitepress.org/Papers/2016/57924/57924.pdf) | [Description](https://bmi.teicrete.gr/en/the-mobifall-and-mobiact-datasets-2/) | [Download](https://drive.google.com/file/d/0B5VcW5yHhWhielo5NTk1Q3ZiWDQ/) | 2016 | 100 | yes |  |  | 57 | 9 |  |
| Zhang | uschad | [USC-HAD: A Daily Activity Dataset for Ubiquitous Activity Recognition Using Wearable Sensors](http://sipi.usc.edu/had/mi_ubicomp_sagaware12.pdf) | [Description](http://sipi.usc.edu/had/) | [Download](http://sipi.usc.edu/had/USC-HAD.zip) | 2012 | 100 | yes | yes | | 15 | 12 |  |



# Contributing

We will gladly accept contributions to this repository in any form, but particularly we welcome additional datasets, new feature extraction processes, view representations, and bug fixes.


## Adding new Datasets

New datasets can be added by contacting me via email or by submitting a new issue (preferred). Simply provide me with information required to populate a new row in the table above. If you have a transformer that will convert the data to the preferred format please attach this too. If not I will then attempt to write a converter for the data but this may take some time.

## Update via Pull Request

Two steps must be performed for a Pull Request to be accepted: 1. update the table above; and 2. add the transformer to the repository. These steps are outlined in more detail below:

### Update the Table

The table above can be updated by adding a row with the following information:

``` MarkDown
| AuthorName | DatasetName | [PaperName](PaperURL) | [Description](DescriptionURL) | [Download](DownloadURL) | PublicationYear | SamplingFrequency | HasAccelerometer | HasGyroscope | HasMagnetometer | NumSubjects | NumActivities | Notes |
```

Please insert the new row alphabetically based on the first author's name and then by publication date if there is a tie. Note, the name of the dataset will be immutable and only in exceptional circumstances will the name be changed.

### Add Transformer

A new data transformer should be placed in `src/converters/<DatasetName>.py` where `<DatasetName>` matches the second element of the newly inserted row. This file must provide a function called `<DatasetName>` wich accepts as an argument the Contained within this file should be a function called `<DatasetName>` which returns pandas dataframes. Using the `spherechallenge` dataset as an example, a file in `src/data/spherechallenge.py` will contain the followign:

``` Python
def spherechallenge(input_path):
    data = load_sphere_challenge_data(input_path)
    return data
```

It is important that there is consistency between the name of the dataset in the table above, the name of the file in the `src` directory and the name of the function since the module importer reads the data information from this table and dynamically loads the transformation functions dynamically. In other words, the function must be importable as follows

``` Python
from spherechallenge import spherechallenge
```

## Adding New Feature Representations

We have implemented several feature extraction processes in the `src/features` directory and interfaces to map these features to the above datasets also. These should be relatively straightforward to add since they will typically operate on a matrix of acceleration data and will return a vector. As a simple example one may extract the `mean`, `standard deviation`, `range`, `min` and `max` values as follows:

``` Python
import numpy as np

stat_funcs = [np.mean, np.std, np.ptp, np.min, np.max]

def extract_stat_features(data):
	return np.concatenate([func(data, axis=0) for func in stat_funcs])
```

## Adding Transformers

Several pre-processing techniques are often applied to accelerometer data. For example, it is common to separate the 'body' and 'gravity' components from each other, compute the magnitude of the data etc.


# Project Organization

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── features       <- The representation of the processed data.
│   ├── processed      <- The intermediate data, transformed to the desired format.
│   └── raw            <- The original, immutable data dump. All datasets have unique
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
```

