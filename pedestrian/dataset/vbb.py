# -*- coding: utf-8 -*-
"""
https://github.com/CasiaFan/Dataset_to_VOC_converter/blob/master/vbb2voc.py
"""


import os, glob
import cv2
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
from lxml import etree, objectify


class SeqVbb(object):

    """Parser for vbb/seq format in Caltech Pedestrian Dataset"""

    def __init__(self):
        pass


    def readvbb(self, vbb_file, cam_id):
        filename = os.path.splitext(os.path.basename(vbb_file))[0]
        annos = defaultdict(dict)
        vbb = loadmat(vbb_file)
        # object info in each frame: id, pos, occlusion, lock, posv
        objLists = vbb['A'][0][0][1][0]
        objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
        # person index
        person_index_list = np.where(np.array(objLbl) == "person")[0]
        for frame_id, obj in enumerate(objLists):
            if len(obj) > 0:
                frame_name = str(cam_id) + "_" + str(filename) + "_" + str(frame_id+1) + ".jpg"
                annos[frame_name] = defaultdict(list)
                annos[frame_name]["id"] = frame_name
                annos[frame_name]["label"] = "person"
                for id, pos, occl in zip(obj['id'][0], obj['pos'][0], obj['occl'][0]):
                    id = int(id[0][0]) - 1  # for matlab start from 1 not 0
                    if not id in person_index_list:  # only use bbox whose label is person
                        continue
                    pos = pos[0].tolist()
                    occl = int(occl[0][0])
                    annos[frame_name]["occlusion"].append(occl)
                    annos[frame_name]["bbox"].append(pos)
                if not annos[frame_name]["bbox"]:
                    del annos[frame_name]
        return annos


    def readseq(self, seq_file, cam_id, anno_dict=None):
        cap = cv2.VideoCapture(seq_file)
        v_id = os.path.splitext(os.path.basename(seq_file))[0]

        # captured frame list if annd_dict given
        if anno_dict is not None:
            cap_frames_index = np.sort([int(os.path.splitext(id)[0].split("_")[2]) for id in anno_dict.keys()])
        else:
            cap_frames_index = np.array([])

        index = 1
        out_imgs = {}
        while True:
            ret, frame = cap.read()
            if ret:
                if cap_frames_index.shape[0] > 0  and not index in cap_frames_index:
                    index += 1
                    continue
                frameid = str(cam_id)+"_"+v_id+"_"+str(index)+".jpg"
                out_imgs[frameid] = frame
            else:
                break
            index += 1
        return out_imgs


    def getbbox(self, anno):
        bboxes = np.array(anno['bbox']).astype(int)
        bboxes[:,2] += bboxes[:, 0]
        bboxes[:,3] += bboxes[:, 1]
        return bboxes


    def show(self, imgs, annos, index=None):

        if index is not None:
            assert index >= 0 and index < len(imgs.keys())
            key = sorted(imgs.keys())[index]
            img = imgs[key]
            if key in annos.keys():
                bboxes = self.getbbox(annos[key])
                for bbox in bboxes:
                    img = cv2.rectangle(imgs[key], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.imshow('img', img)
            ch = cv2.waitKey(0) & 0xff
            return

        keyinds = sorted(['{:0>6}'.format(x.split('.')[0].split('_')[-1]) for x in imgs.keys()])
        
        def fetch_head():
            keyname = list(imgs.keys())[0] 
            keyid = keyname.split('.')[0].split('_')[-1]
            key_loc = keyname.find(keyid)
            return keyname[:key_loc]

        key_head = fetch_head() 

        for ind in range(len(imgs.keys())):
            key = key_head + str(int(keyinds[ind])) + '.jpg'
            img = imgs[key]
            if key in annos.keys():
                bboxes = self.getbbox(annos[key])
                for bbox in bboxes:
                    img = cv2.rectangle(imgs[key], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            cv2.putText(img, '{}/{}'.format(ind, len(imgs.keys())), 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),2)
            cv2.imshow('img', img)

            ch = cv2.waitKey(5) & 0xff
            if ch == 27: #ord('q')
                break
        print('frame sum: %d' % len(imgs.keys()))
    

if __name__ == "__main__":
    #root = os.path.expanduser('~/tmp/Caltech')
    #cameraID = 'set00'
    #vbb = os.path.join(root, 'V000.vbb')
    #seq = os.path.join(root, 'V000.seq')

    root = os.path.expanduser('~/data/pedestrian/caltech')
    cameraID = 'set06'
    videoID = 'V000'
    seq = os.path.join(root, cameraID, videoID+'.seq')
    vbb = os.path.join(root, 'annotations', cameraID, videoID+'.vbb')

    parser = SeqVbb()
    print('parsing vbb...')
    annos = parser.readvbb(vbb, cameraID)
    print('parsing seq...')
    #imgs = parser.readseq(seq, cameraID, annos) # only show frame with target
    imgs = parser.readseq(seq, cameraID) # show all frames

    parser.show(imgs, annos)
