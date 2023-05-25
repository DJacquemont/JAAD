# Jaad4MotionBERT

This repository contains code and annotations to create a dataset of individual pedestrian bounding boxes (without the frames) with their 2d keypoints created by OpenPifPaf, and their intention (crossing / not crossing) from the JAAD dataset. The created database is used to train a modified version of the MotionBERT model to predict pedestrian intention in the context of Autonomous Vehicules.

# Requirements
The videos need to be imported. This is done by executing the following line in the *JAAD_DS* folder
```
./download_clips.sh
```
Packages required :
```
pip install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install --upgrade openpifpaf==0.10.1
```

This code was developped and tested with Python 3.10.6.
## Dataset Creation

To create the dataset, open a terminal in the *JAAD* folder and run the following command :
```
python3 dataset.py
```

## Output

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