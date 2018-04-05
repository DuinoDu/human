# -*- coding: utf-8 -*-

import os
import os.path as osp
from PIL import Image
import cv2
import numpy as np
from lxml import objectify, etree
import tqdm

try:
    import commands
except Exception as e:
    import subprocess as commands


default_root = osp.expanduser('~/data/pedestrian')


class Pedestrian(object):

    """Base class for pedestrian detection dataset"""

    def __init__(self, name, root):
        self._name = name
        self._root = root
        self._voc_name = self._name + '_voc'
        if not osp.exists(root):
            os.makedirs(root)


    @property
    def root(self):
        return self._root
    def set_root(self, root):
        self._root = root

    @property
    def name(self):
        return self._name

    @property
    def voc_name(self):
        return self._voc_name


    def download(self):
        raise NotImplementedError
    

    def unzip(self):
        raise NotImplementedError


    def convert2voc(self):
        raise NotImplementedError


    def eval(self):
        raise NotImplementedError


    def create_voc(self):
        voc_root = osp.join(self.root, self.name+'_voc')
        jpgdir = osp.join(voc_root, 'JPEGImages')
        annodir = osp.join(voc_root, 'Annotations') 
        splitdir = osp.join(voc_root, 'ImageSets/Main')

        def mkdir(f):
            if not osp.exists(f):
                os.makedirs(f)
        
        mkdir(voc_root)
        mkdir(jpgdir)
        mkdir(annodir)
        mkdir(splitdir)
        return voc_root, jpgdir, annodir, splitdir


    def create_split(self, splitdir, clsname, trainval_list, test_list, train_list=None, val_list=None):
        if not os.path.exists(splitdir):
            raise ValueError("{} not exist.".format(splitdir))

        trainvalfile = 'trainval.txt'
        testfile = 'test.txt'

        if clsname is not None:
            trainvalfile = '{}_trainval.txt'.format(clsname)
            testfile = '{}_test.txt'.format(clsname)

        trainvalfile = osp.join(splitdir, trainvalfile)
        testfile = osp.join(splitdir, testfile) 
        
        for file, lines in zip([trainvalfile, testfile], [trainval_list, test_list]):
            with open(file, 'w') as fid:
                for line in lines:
                    fid.write(line + '\n')
        
        if train_list is not None and val_list is not None:
            trainfile = osp.join(splitdir, 'train.txt')
            valfile = osp.join(splitdir, 'val.txt') 
            for file, lines in zip([trainfile, valfile], [train_list, val_list]):
                with open(file, 'w') as fid:
                    for line in lines:
                        fid.write(line + '\n')

    
    def check_anno(self, bboxes, imgfile, imagemagick=False):
        h, w = self.get_image_wh(imgfile, imagemagick)

        # 1<= x <= w, 1 <= y <= h
        bboxes[:, 0] = np.maximum(bboxes[:, 0], 1)
        bboxes[:, 1] = np.maximum(bboxes[:, 1], 1)
        bboxes[:, 2] = np.minimum(bboxes[:, 2], w)
        bboxes[:, 3] = np.minimum(bboxes[:, 3], h)
        
        # delete x1 == x2, y1 == y2
        # np.delete is not used, because bboxes is also used in for-loop
        new_bboxes = np.array([[0, 0, 0, 0]]).astype(bboxes.dtype)
        for i in range(bboxes.shape[0]):
            if bboxes[i, 0] != bboxes[i, 2] and bboxes[i, 1] != bboxes[i, 3]:
                new_bboxes = np.append(new_bboxes, 
                                       np.expand_dims(bboxes[i], 0), 
                                       0)
        bboxes = new_bboxes[1:]

        # x1 < x2, y1 < y2
        new_bboxes = np.zeros_like(bboxes)
        new_bboxes[:, 0] = np.minimum(bboxes[:, 0], bboxes[:, 2])
        new_bboxes[:, 2] = np.maximum(bboxes[:, 0], bboxes[:, 2])
        new_bboxes[:, 1] = np.minimum(bboxes[:, 1], bboxes[:, 3])
        new_bboxes[:, 3] = np.maximum(bboxes[:, 1], bboxes[:, 3])
        bboxes = new_bboxes
        return bboxes

    
    def get_image_wh(self, imgfile, imagemagick=False):
        if imagemagick:
            cmd = 'identify {} | cut -d \' \' -f 3'.format(imgfile)
            (status, output) = commands.getstatusoutput(cmd)
            w = int(output.split('x')[0])
            h = int(output.split('x')[1])
        else:
            h, w = cv2.imread(imgfile).shape[:2]
        return (h, w)


    def anno2xml(self, imgfile, anno_dict):
        """Convert anno dict to xml tree

        Args:
            imgfile (str): image path 
            anno_dict (TODO): annotation dict
                {'id': filename, 'bboxes':[{'name':'person', 'xyxy':[]}, ...]}

        Returns: 
            element tree
        """

        height, width = self.get_image_wh(imgfile, imagemagick=True)

        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.annotation(
            E.folder(self.voc_name),
            E.filename(anno_dict['id']),
            E.source(
                E.database(self.name),
                E.annotation(self.name),
                E.image(self.name),
                E.url('None')
            ),
            E.size(
                E.width(width),
                E.height(height),
                E.depth(3)
            ),
            E.segmented(0),
        )
        for bbox in anno_dict['bboxes']:
            name = bbox['name']
            bbox = [int(x) for x in bbox['xyxy']]
            xmin, ymin ,xmax, ymax = bbox

            E = objectify.ElementMaker(annotate=False)
            anno_tree.append(
                E.object(
                    E.name(name),
                    E.bndbox(
                        E.xmin(xmin),
                        E.ymin(ymin),
                        E.xmax(xmax),
                        E.ymax(ymax)),
                E.difficult(0),
                E.occlusion(0))
            )
        return anno_tree


    def create_fake_test_anno(self):
        """ Create fake annotations xml file for testset.
        """
        voc_root, jpgdir, annodir, splitdir = self.create_voc()
        lines = [x.strip() for x in open(splitdir+'/test.txt', 'r').readlines()]

        annofiles_exist = [x for x in os.listdir(annodir) if x.endswith('.xml')]
        if lines[0] in annofiles_exist:
            print("Test annotation already exists, create_fake_test_anno failed.")
            return

        t = tqdm.tqdm()
        t.total = len(lines) 

        print("Creating fake test annotations ...")
        for line in lines:
            t.update()
            imgfile = osp.join(jpgdir, '{}.jpg'.format(line))
            anno_tree = self.anno2xml(imgfile, {"id":line, "bboxes":[]})
            etree.ElementTree(anno_tree).write(
                osp.join(annodir, '{:0>6}.xml'.format(line)),
                pretty_print=True)
        print("create_fake_test_anno successfully.")
