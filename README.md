# Data4PedIntPred

This repository contains :
- Code and annotations to create a dataset of individual pedestrian bounding boxes (without the frames) with their 2d keypoints created by OpenPifPaf, and their intention (crossing / not crossing) from the JAAD dataset. The created database is used to train a modified version of the MotionBERT model to predict pedestrian intention in the context of Autonomous Vehicules.
- Code to process any inference video by using OpenPifPaf to generate 2D keypoints for all pedestrians, and to recreate the inference video with the model's output prediction.

## Installation
The videos need to be imported. This is done by executing the following line in the *JAAD_DS* folder
```
./download_clips.sh
```
Packages required :
```
conda create -n data4depintpred python=3.7 anaconda
conda activate data4depintpred
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```
This code was developped and tested with Python 3.7.

## Training Dataset Creation

To create the dataset, open a terminal in the repository folder and run the following command :
```
python3 dataset.py --data_path=<folder_path> --compute_kps --regen
```
- `--data_path` helps to specify the folder path if different from the current one
- `--compute_kps` flag to compute keypoints with bounding boxes. Only the boundingbox will be included in the output pickle file if the flag is omitted.
- `--regen` flag to regenerate the database

## Inference Data Creation

To create the inference data, open a terminal in the repository folder and run the following command :
```
python3 dataset.py --data_path=<folder_path> --compute_kps --regen
```
- `--data_path` helps to specify the folder path if different from the current one
- `--compute_kps` flag to compute keypoints with bounding boxes. Only the boundingbox will be included in the output pickle file if the flag is omitted.
- `--regen` flag to regenerate the database

## Output

The processed directory tree should look like this:
```
.
├── README.md
├── dataset.py
└── JAAD_DS
    ├── download_clips.sh
    ├── jaad_dataset.pkl
    ├── annotations
    │   └── video_X.xml
    ├── JAAD_clips
    │   └── video_X.mp4
    └── LICENSE
```

The output of the code is a pickle file *jaad_database.pkl* containing a dictionary with the following structure :
```
'annotations': {
    'vid_id'(str): {
        'num_frames':   int
        'width':        int
        'height':       int
        'ped_annotations'(str): {
            'ped_id'(str): list (dict) {
                'old_id':       str
                'frames':       list (int)
                'occlusion':    list (int)
                'bbox':         list ([x1 (float), y1 (float), x2 (float), y2 (float)])
                '2dkp':         list (array(array))
                'cross':        list (int)}}}}
'split': {
    'train_ID': list (str)
    'test_ID':  list (str)}
'ckpt': str
'seq_per_vid': list (int)
```