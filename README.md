# Jaad4MotionBERT

This repository contains code and annotations to create a dataset of individual pedestrian bounding boxes (without the frames), and their intention (crossing / not crossing) from the JAAD dataset. The created database is used to train a modified version of the MotionBERT model to predict pedestrian intention in the context of Autonomous Vehicules.

## Dataset Creation

To create the dataset, open a terminal in the *JAAD* folder and run the following command :
```
python3 dataset.py
```
This code was tested with Python 3.10.6.

## Output

The output of the code is a pickle file *jaad_database.pkl* containing a dictionary with the following structure :
```
Jaad Database :
'annotations': {
    'vid_id'(str): {
        'num_frames':   int
        'width':        int
        'height':       int
        'ped_annotations'(str): {
            'ped_id'(str): list (dict) {
                'old_id':       str
                'frames:        list (int)
                'occlusion':    list (int)
                'bbox':         list ([x1 (float), y1 (float), x2 (float), y2 (float)])
                'cross':        list (int)}}}}
'split': {
    'train_ID': list (str)
    'test_ID':  list (str)}
```
