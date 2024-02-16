# Btrack 
This is the open source code and data of the paper **"BLE Location Tracking Attacks by Exploiting
Frequency Synthesizer Imperfection"**, which is accepted by IEEE INFOCOM 2024

The models we have trained and the dataset is uploaded to the [Google drive](https://drive.google.com/drive/folders/1Ld7K3ad2meLJg_RM7i3zlg27SIVEaZTj?usp=drive_link).
The models we trained ended with a number corresponding to the trainning step.
We also store the trainning output in a markdown file named as `<device name>.md`, so you can locate the model with the step. 

The code file and its usage is shown as follows:

File | Models | Description 
----|--------|------------|
feature_extractor.py | - | The code to extract the transient phase from the raw data.
basic_feature_extractor.py | - | The helper code for data preprocessing
manu_feature_ana.py| - | Manuaully extract TDFs and show the changes of temperature changes (Fig.9)
cnn_unique.py | models/each_fix | The code for idetentify each device (Fig.13).
cnn_dis.py   | models/goods    | The code for distance and site robustness (Fig.16)
cnn_temp_main.py | models/temp_good | The code for temperature robustness (Fig.15)
cnn_track.py  | models/track   | The code for the case study (Fig.17)
cfo_based_classifier.py | -| The code for comparing with the SOTA CFO-based method