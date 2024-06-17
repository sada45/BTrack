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

------------
P.S.

We collect 256/512 packets of each node in different scenarios, but the number of signal traces in the dataset may a bit less than the number. The reason is the interference in 2.4GHz can cause:
(1) failed fine-grained preamble detection (`scripts/fext/feature_extractor.py/get_ramp_seg_cfo()`).
(2) transient delay detection error and filtered out by our model filter.

For distance robustness, the office sceinaro we used for distance evaluation has lots of interference, so we see "sudden drops" of the target identification accuary, especially between 3m-7m.
We just repeat data collection to alleviate the impact and the result still show accuracy drop between 3m-7m in the paper.
As for the case study (tracking), accuracy gradually decreases with distance.
When we fist collect the dataset, it also has the same "sudden drop" issue since the building is fully covered with WiFi.
The dataset we used now is collected at the same place but at a time when most of the people went out for a conference and only two people are left in this floor.

17th Jun 2024 Update:
The dataset for the nRF52840 in different environment has been updated.