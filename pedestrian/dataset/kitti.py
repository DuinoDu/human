# -*- coding: utf-8 -*-
from base import Pedestrian, default_root
import os
import os.path as osp
from scipy.io import loadmat
import glob
import numpy as np
import tqdm
from lxml import etree
try:
    import commands
except Exception as e:
    import subprocess as commands


class Kitti(Pedestrian):

    """Kitti Dataset
    http://www.cvlibs.net/datasets/kitti/eval_object.php
    """

    def __init__(self, root=osp.join(default_root, 'kitti')):
        super(Kitti, self).__init__('kitti', root)
        self.classes = ('car', 'pedestrian', 'cyclist')
        self.train_val_ratio = 0.8

    def download(self):
        print("Refer scripts/shell/fetch/pedestrian/kitti.sh to download dataset.")


    def unzip(self):
        print("Do it by yourself.")


    def convert2voc(self):
        if os.path.exists(osp.join(self.root, self.voc_name)):
            print('kitti_voc already exists')
            return False
        voc_root, jpgdir, annodir, splitdir = self.create_voc()

        if not os.path.exists(os.path.join(self.root, 'training/image_2')):
            raise IOError('Image not found, please download and unzip')

        train_list = []
        val_list = []
        test_list = []

        train_img = osp.join(self.root, 'training/image_2/{:0>6}.png')
        train_anno = osp.join(self.root, 'training/label_2/{:0>6}.txt')
        test_img = osp.join(self.root, 'testing/image_2/{:0>6}.png')

        sum_train = len(os.listdir(osp.join(self.root, 'training/image_2')))
        sum_test = len(os.listdir(osp.join(self.root, 'testing/image_2')))

        t = tqdm.tqdm()
        t.total = sum_train + sum_test

        for ind in range(sum_train + sum_test):
            t.update()
            imgfile = train_img.format(ind) if ind < sum_train else \
                      test_img.format(ind-sum_train)
            # Annotations
            if ind < sum_train:
                annofile = train_anno.format(ind)
                anno_lines = [x.strip() for x in open(annofile, 'r').readlines()]

                anno_dict = {}
                anno_dict['id'] = '{:0>6}'.format(ind+1)
                anno_dict['bboxes'] = []

                for anno in anno_lines:
                    anno = anno.split(' ')
                    clsname = anno[0].lower()
                    if clsname not in self.classes:
                        continue
                    
                    bbox = [int(float(x)) for x in anno[4:8]] 
                    bbox_checked = self.check_anno(np.array([bbox]), imgfile, imagemagick=True)
                    if bbox_checked.shape[0] < 1:
                        continue
                    bbox_checked = bbox_checked[0]
                    anno_dict['bboxes'].append({'name':clsname, 'xyxy': bbox_checked.tolist()})
                if len(anno_dict['bboxes']) == 0:
                    continue
                anno_tree = self.anno2xml(imgfile, anno_dict)
                etree.ElementTree(anno_tree).write(
                    osp.join(annodir, '{:0>6}.xml'.format(ind+1)),
                    pretty_print=True)
            # JPEGImages
            cmd = 'ln -s {} {}'.format(
                    imgfile,
                    osp.join(jpgdir, '{:0>6}.jpg'.format(ind+1)))
            (status, output) = commands.getstatusoutput(cmd)
            # ImageSets/Main
            if ind < int(sum_train * self.train_val_ratio):
                train_list.append('{:0>6}'.format(ind+1))
            elif ind < sum_train:
                val_list.append('{:0>6}'.format(ind+1))
            else:
                test_list.append('{:0>6}'.format(ind+1))
        
        trainval_list = train_list + val_list
        self.create_split(splitdir, None, trainval_list, test_list, train_list, val_list)

        print("===> Successfully.")
        print("Kitti in voc-format is saved in {}".format(voc_root))
        return True


    def eval():
        print('Refer to http://kitti.is.tue.mpg.de/kitti/devkit_object.zip')


if __name__ == "__main__":
    ds = Kitti()
    ret = ds.convert2voc()
    if ret:
        ds.create_fake_test_anno()
