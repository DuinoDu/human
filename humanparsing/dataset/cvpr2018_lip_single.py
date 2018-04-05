#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from PIL import Image
import tqdm
import zipfile
try:
    import commands
except Exception as e:
    import subprocess as commands

default_root = os.path.expanduser('~/data')
if not os.path.exists(default_root):
    os.makedirs(default_root)


def extract(inputfile, savedir):
    print('Unzip ' + inputfile + ' ...')
    zip_ref = zipfile.ZipFile(inputfile, 'r')
    zip_ref.extractall(savedir)
    zip_ref.close()


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def ln(src, dst):
    cmd = 'ln -s {} {}'.format(src, dst)
    (status, output) = commands.getstatusoutput(cmd)
    output = output.split('\n')


class LIPsingle(object):

    def __init__(self, name, ziproot):
        self.name = name
        self.ziproot = ziproot
        self.save_path = os.path.join(default_root, name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.voc_name = self.name + '_voc'


    def unzip(self):
        trainval_img = os.path.join(self.ziproot, 'TrainVal_images.zip')
        test_img =  os.path.join(self.ziproot, 'Testing_images.zip')

        extract(trainval_img, self.save_path) 
        extract(os.path.join(self.save_path, 'TrainVal_images.zip'), self.save_path) 
        extract(test_img, self.save_path) 
        extract(os.path.join(self.save_path, 'Testing_images.zip'), self.save_path) 

        trainval_anno =  os.path.join(self.ziproot, 'TrainVal_parsing_annotations.zip')
        extract(trainval_anno, self.save_path) 
        trainval_anno =  os.path.join(self.save_path, 
                                      'TrainVal_parsing_annotations',
                                      'TrainVal_parsing_annotations.zip')
        extract(trainval_anno, self.save_path) 


    def convert2voc(self):
        voc_root = os.path.join(self.save_path, self.voc_name)
        if not os.path.exists(voc_root):
            os.makedirs(voc_root)
        else:
            print(voc_root + ' already exists')
            return
        
        jpegdir = os.path.join(voc_root, 'JPEGImages') 
        annodir = os.path.join(voc_root, 'SegmentationClass')
        imgsetdir1 = os.path.join(voc_root, 'ImageSets/Main')
        imgsetdir2 = os.path.join(voc_root, 'ImageSets/Segmentation')
        mkdir(jpegdir)
        mkdir(annodir)
        mkdir(imgsetdir1)
        mkdir(imgsetdir2)
        
        train_imgdir = os.path.join(self.save_path, 'train_images')
        train_segdir = os.path.join(self.save_path, 'train_segmentations')
        val_imgdir = os.path.join(self.save_path, 'val_images')
        val_segdir = os.path.join(self.save_path, 'val_segmentations')
        trainid = os.path.join(self.save_path, 'train_id.txt')
        validid = os.path.join(self.save_path, 'val_id.txt')

        trainimg = sorted([os.path.join(train_imgdir, x) for x in sorted(os.listdir(train_imgdir)) if x.endswith('.jpg')])
        validimg = sorted([os.path.join(val_imgdir, x) for x in sorted(os.listdir(val_imgdir)) if x.endswith('.jpg')])
        trainseg = sorted([os.path.join(train_segdir, x) for x in sorted(os.listdir(train_segdir)) if x.endswith('.png')])
        validseg = sorted([os.path.join(val_segdir, x) for x in sorted(os.listdir(val_segdir)) if x.endswith('.png')])
        

        print('creating images ...')
        t = tqdm.tqdm()
        t.total = len(trainimg + validimg)
        for img in trainimg + validimg:
            t.update()
            dstname = os.path.join(jpegdir, os.path.basename(img))
            ln(img, dstname)

        print('creating segmentations ...')
        t = tqdm.tqdm()
        t.total = len(trainseg + validseg)
        for seg in trainseg + validseg:
            t.update()
            dstname = os.path.join(annodir, os.path.basename(seg))
            ln(seg, dstname)

        ln(trainid, os.path.join(imgsetdir1, 'train_seg.txt'))
        ln(validid, os.path.join(imgsetdir1, 'val_seg.txt'))

        ln(trainid, os.path.join(imgsetdir2, 'train.txt'))
        ln(validid, os.path.join(imgsetdir2, 'val.txt'))

        # create test id list
        testid = os.path.join(self.save_path, 'test_id.txt')
        cmd = "ls {}/testing_images | cut -d'.' -f1 > {}".format(self.save_path, testid)
        (status, output) = commands.getstatusoutput(cmd)
        ln(testid, os.path.join(imgsetdir2, 'test.txt'))



if __name__ == "__main__":
    ziproot = os.path.expanduser('~/mnt/dataset/cvpr2018/LIP')
    ds = LIPsingle('LIPsingle', ziproot)
    ds.unzip()
    ds.convert2voc()
