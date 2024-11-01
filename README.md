#  Solution of Team lyy1 for FLARE2024 Challenge

## This repository is the official implementation of [Crop and Segment: Efficient Whole body
pan-cancer segmentation](https://drive.google.com/file/d/1BZoABEBkfdd0j8t2B7nrgdMDKNs99rt4/view?usp=drive_link) of Team lyy1 on FLARE24 challenge.

## Introduction

### Overview of our work.

![image](./IMG/roi.png)

![image](./IMG/fill.png)



## Environments and Requirements

The basic language for our work is [python](https://www.python.org/), and the baseline
is [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1). So, you can install the nnunet frame with
the [GitHub Repository](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1), or use the following comments:

```
pip install torch torchvision torchaudio
pip install -e .
```

## Prepocessing

You can use the nnunet to train the organ model and get the organ pseudo labels.Then fill the tumour label.

### convert CT images to npy

we modify the normalization function with ___preprocessing.py___,
and you could use the following comments to processing the CT images:

```
python nnunet/experiment_planning/nnUNet_convert_decathlon_task.py -i [FLARE24_imageTr_path]

python nnunet/experiment_planning/nnUNet_plan_and_preprocess -t [TASK_ID] --verify_dataset_integrity
```

It must be noted that the method is based on the __nnU-Net__, so I recommend you to convert the dataset within nnU-Net's
data preprocessing.

The usage and note concerning for ___nnUNet_convert_decathlon_task.py___ is recorded
on [website](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/dataset_conversion.md).

After preprocessing, we will obtain several folders:

```
- nnU-Net_base_folder
    - nnUNet_prepocessing
    - nnUNetFrame
    - nnUNet_raw
    - nnUNet_trained_models
```

### Resample the data to our voxel spacing

```
python data_convert.py -nnunet_preprocessing_folder -imagesTr_floder -labelTr_floder
```
where the **nnunet_preprocessing_folder** is the folder path of the dataset planed by nnunet. like 'nnU-Net_base_folder/nnUNet_preprocessed/Task0101_FLARE2024T1/nnUNetData_plans_v2.1_stage1'

## Training

```
python nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2 DATA_ID FOLD --npz --disable_saving
```

## Inference
```
cd tumour_inference
python inference.py [INPUT_FOLDER] [OUTPUT_FOLDER]
```
Before the Inference, you should move the best nnunet checkpoints to replace the three files in folder __'checkpoints'__.

## Evaluation

## Results


## Acknowledgement

MACCAI FLARE2024 https://www.codabench.org/competitions/2319/


## 




