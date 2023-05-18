import pickle
import numpy as np
import xml.etree.ElementTree as ET
from os.path import join, abspath, exists
from os import listdir, makedirs

class JAAD(object):
    def __init__(self, data_path='', regen_pkl=False):
        """
        Constructor of the jaad class
        :param data_path: Path to the folder of the dataset
        :param regen_pkl: Whether to regenerate the database
        """
        self._year = '2016'
        self._name = 'JAAD'
        self._regen_pkl = regen_pkl
        self._image_ext = '.png'

        # Paths
        self._jaad_path = data_path if data_path else self._get_default_path()
        assert exists(self._jaad_path), \
            'Jaad path does not exist: {}'.format(self._jaad_path)
        self._annotation_path = join(self._jaad_path, 'annotations')

    # Path generators
    @property
    def cache_path(self):
        """
        Generate a path to save cache files
        :return: Cache file folder path
        """
        cache_path = abspath(self._jaad_path)
        if not exists(cache_path):
            makedirs(cache_path)
        return cache_path

    def _get_default_path(self):
        """
        Return the default path where jaad_raw files are expected to be placed.
        :return: the default path to the dataset folder
        """
        return 'dataset/jaad'

    def _get_video_ids(self):
        """
        Returns a list of all video ids
        :return: The list of video ids
        """
        return [vid.split('.')[0] for vid in listdir(self._annotation_path)]


    # Annotation processing helpers
    def _map_text_to_scalar(self, label_type, value):
        """
        Maps a text label in XML file to scalars
        :param label_type: The label type
        :param value: The text to be mapped
        :return: The scalar value
        """
        map_dic = {'occlusion': {'none': 0, 'part': 1, 'full': 2},
                   'action': {'standing': 0, 'walking': 1},
                   'nod': {'__undefined__': 0, 'nodding': 1},
                   'look': {'not-looking': 0, 'looking': 1},
                   'hand_gesture': {'__undefined__': 0, 'greet': 1, 'yield': 2,
                                    'rightofway': 3, 'other': 4},
                   'reaction': {'__undefined__': 0, 'clear_path': 1, 'speed_up': 2,
                                'slow_down': 3},
                   'cross': {'not-crossing': 0, 'crossing': 1, 'irrelevant': -1},
                   'age': {'child': 0, 'young': 1, 'adult': 2, 'senior': 3},
                   'designated': {'ND': 0, 'D': 1},
                   'gender': {'n/a': 0, 'female': 1, 'male': 2},
                   'intersection': {'no': 0, 'yes': 1},
                   'motion_direction': {'n/a': 0, 'LAT': 1, 'LONG': 2},
                   'traffic_direction': {'OW': 0, 'TW': 1},
                   'signalized': {'n/a': 0, 'NS': 1, 'S': 2},
                   'vehicle': {'stopped': 0, 'moving_slow': 1, 'moving_fast': 2,
                               'decelerating': 3, 'accelerating': 4},
                   'road_type': {'street': 0, 'parking_lot': 1, 'garage': 2},
                   'traffic_light': {'n/a': 0, 'red': 1, 'green': 2}}

        return map_dic[label_type][value]

    def _get_annotations(self, vid):
        """
        Generates a dictinary of annotations by parsing the video XML file
        :param vid: The id of video to parse
        :return: A dictionary of annotations
        """

        # vid feed -> 30 FPS
        forecast_time = 2 #s
        forecast_frames = 30 * forecast_time
        label_frames = 15


        path_to_file = join(self._annotation_path, vid + '.xml')
        tree = ET.parse(path_to_file)
        ped_annt = 'ped_annotations'

        annotations = {}
        annotations['num_frames'] = int(tree.find("./meta/task/size").text)
        annotations['width'] = int(tree.find("./meta/task/original_size/width").text)
        annotations['height'] = int(tree.find("./meta/task/original_size/height").text)
        annotations[ped_annt] = {}

        nbr_seq = 0
        ped_tracks = tree.findall("./track")

        for t in ped_tracks:    

            boxes = t.findall('./box')

            new_id = boxes[0].find('./attribute[@name=\"id\"]').text
                  
            if 'b' in new_id:

                old_id = boxes[0].find('./attribute[@name=\"old_id\"]').text

                tmp_bbox = []
                tmp_occ = []
                tmp_cross = []
                tmp_frames = []

                if 'pedestrian' in old_id:
                    for b in boxes:
                        tmp_bbox.append(
                            [float(b.get('xtl')), float(b.get('ytl')), float(b.get('xbr')), float(b.get('ybr'))])
                        tmp_occ.append(self._map_text_to_scalar('occlusion',
                                                                b.find('./attribute[@name=\"occlusion\"]').text))
                        tmp_frames.append(int(b.get('frame')))
                        tmp_cross.append(self._map_text_to_scalar('cross', b.find('./attribute[@name=\"cross\"]').text))

                    if (len(tmp_bbox)- 30 - 1)/30 >= 1 :
                        annotations[ped_annt][new_id] = []
                        nbr_seq += int(((len(tmp_bbox) - 30 - 1)/30))
                        for i in range(0, int(((len(tmp_bbox) - 30 - 1)/30))):
                            print("-----------------------------")
                            print("pedestrian id                : " + str(new_id))
                            print("Current sequence             : " + str(i))
                            
                            annotation_bbox = np.array(tmp_bbox[i*30:(i+2)*30])
                            annotation_occ = tmp_occ[i*30:(i+2)*30]
                            annotation_frames = tmp_frames[i*30:(i+2)*30]

                            end_idx_cross = min(len(tmp_cross[(i+2)*30:]), label_frames)

                            print("Min between                  : " + str(len(tmp_cross[(i+2)*30:])) + " and " + str(label_frames) + " is " + str(end_idx_cross))

                            annotation_cross = np.amax(np.array(tmp_cross[(i+2)*30:(i+2)*30+end_idx_cross]))

                            annotations_dict = {'old_id': old_id, 'frames': annotation_frames,
                                                'bbox': annotation_bbox, 'occlusion': annotation_occ, 'cross': annotation_cross}

                            annotations[ped_annt][new_id].append(annotations_dict)

                            print("Current frames annotations   : " + str(i*30) + " - " + str((i+2)*30))
                            print("Current frames label         : " + str((i+2)*30) + " - " + str((i+2)*30 + end_idx_cross))
                            print("Length bbox vector           : " + str(len(annotations[ped_annt][new_id][i]['bbox'])))
                            print("Length cross vector          : " + str(len(np.array(tmp_cross[(i+2)*30:(i+2)*30+end_idx_cross]))))
                            #print("Cross vector                 : " + str(np.array(tmp_cross[(i+2)*30:(i+2)*30+end_idx_cross])))
                            #print("End index cross              : " + str((i+2)*30 + end_idx_cross))
                            print("Cross value                  : " + str(annotations[ped_annt][new_id][i]['cross']))
                
        return annotations, nbr_seq

    
    def generate_database(self):
        """
        Generate a database of jaad dataset by integrating all annotations
        Dictionary structure:
        'annotations': {
            'vid_id'(str): {
                'num_frames':   int
                'width':        int
                'height':       int
                'ped_annotations'(str): {
                    'ped_id'(str): list(dict) {
                        'old_id':       str
                        'frames:        list(int)
                        'occlusion':    list(int)
                        'bbox':         list([x1, y1, x2, y2])
                        'cross':        list(int)}
        'split': {
            'train_ID': list(int)
            'test_ID':  list(int)}

        :return: A database dictionary
        """
        print("Generating database for jaad\n\n")

        # Generates a list of behavioral xml file names for  videos
        cache_file = join(self.cache_path, 'jaad_database.pkl')
        if exists(cache_file) and not self._regen_pkl:
            with open(cache_file, 'rb') as fid:
                try:
                    database = pickle.load(fid)
                except:
                    database = pickle.load(fid, encoding='bytes')
            print('jaad database loaded from {}'.format(cache_file))
            return database

        video_ids = sorted(self._get_video_ids())
        database = {'split': {'train_ID':[], 'test_ID': []}, 'annotations': {}}
        database_vid_ID = []
        nbr_seq_vid_ID = []
        for vid in video_ids:
            print('---------------------------------------------------------')
            print('----------- Getting annotations for %s ----------' % vid)
            print('---------------------------------------------------------')
            
            vid_annotations, nbr_seq = self._get_annotations(vid)
            if (nbr_seq != 0):
                database['annotations'][vid] = vid_annotations
                database_vid_ID.append(vid)
                nbr_seq_vid_ID.append(nbr_seq)
                print('\nAnnotations found for %s\n' % vid)
            else:
                print('\nNo pedestrian annotations found for %s\n' % vid)   

        # Creating testset/trainset
        cumsum = np.cumsum(nbr_seq_vid_ID)/sum(nbr_seq_vid_ID)
        res = next(x for x, val in enumerate(cumsum) if val > 0.2)
        database['split']['train_ID'] = database_vid_ID[res:]
        database['split']['test_ID'] = database_vid_ID[:res]

        with open(cache_file, 'wb') as fid:
            pickle.dump(database, fid, pickle.HIGHEST_PROTOCOL)
        print('The database is written to {}'.format(cache_file))

        return database

if __name__ == "__main__":
    DS = JAAD(data_path='/home/dimitri/DLAV_ws/JAAD/JAAD', regen_pkl=True)
    DS.generate_database()