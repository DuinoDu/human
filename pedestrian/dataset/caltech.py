# -*- coding: utf-8 -*-

import time
import os
import os.path as osp
import commands
from base import Pedestrian, default_root
import tqdm
from vbb import SeqVbb
import cv2
import numpy as np
from lxml import etree


class Caltech(Pedestrian):

    """Caltech Pedestrian Dataset
    http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians
    """

    def __init__(self, root=osp.join(default_root, 'caltech')):
        super(Caltech, self).__init__('caltech', root)
        self.img_filename = 'set{:0>2}.tar'
        self.anno_filename = 'annotations.zip'
        self.test_type = 'voc' # 'caltech'

    
    def set_test_type(self, test_type):
        self.test_type = test_type
        if test_type == 'caltech':
            print("Please make sure caltech/setxx/frame are deleted.")
            

    def download(self):
        urlroot = 'http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/'
        imgfile = osp.join(urlroot, self.img_filename)
        annofile = osp.join(urlroot, self.anno_filename)

        print("===> Downloading Caltech ...")

        # download imagefiles
        for i in range(11):
            cmd = 'wget {} -P {}'.format(imgfile.format(i), self.root)
            (status, output) = commands.getstatusoutput(cmd)
            output = output.split('\n')

        # download annotation
        cmd = 'wget {} -P {}'.format(annofile, self.root)
        (status, output) = commands.getstatusoutput(cmd)
        output = output.split('\n')

        print("===> Successfully.")

    
    def unzip(self):
        imgfile = osp.join(self.root, self.img_filename)
        annofile = osp.join(self.root, self.anno_filename)

        print("===> Unzip Caltech ...")

        # download imagefiles
        for i in range(11):
            cmd = 'tar xvf {} -C {}'.format(imgfile.format(i), self.root)
            (status, output) = commands.getstatusoutput(cmd)
            output = output.split('\n')

        # download annotation
        cmd = 'unzip {} -d {}'.format(annofile, self.root)
        (status, output) = commands.getstatusoutput(cmd)
        output = output.split('\n')

        print("===> Successfully.")


    def create_xml(self, bboxes, imgfile, cnt, annodir):
        bboxes = self.check_anno(bboxes, imgfile)

        if isinstance(bboxes, np.ndarray):
            bboxes = bboxes.tolist()
        if len(bboxes) < 1:
            return False
        anno_dict = {}
        anno_dict['id'] = "{:0>6}".format(cnt)
        anno_dict['bboxes'] = []
        for bbox in bboxes:
            anno_dict['bboxes'].append({'name':'person', 'xyxy':bbox})
        anno_tree = self.anno2xml(imgfile, anno_dict)
        etree.ElementTree(anno_tree).write(
            osp.join(annodir, '{:0>6}.xml'.format(cnt)),
            pretty_print=True)
        return True


    def convert2voc(self):
        if os.path.exists(osp.join(self.root, self.voc_name)):
            print('caltech_voc already exists')
            return False
        voc_root, jpgdir, annodir, splitdir = self.create_voc()

        print('===>')
        print('This may take some time...')

        parser = SeqVbb()

        train_list = []
        val_list = []
        test_list = []
        cnt = 1

        for i in range(11):
            setid = 'set{:0>2}'.format(i)
            seqdir = osp.join(self.root, setid)
            vbbdir = osp.join(self.root, "annotations", setid)
            framedir = osp.join(seqdir, 'frame') 

            print('parsing '+setid)
            vbbs = sorted([osp.join(vbbdir, x) for x in sorted(os.listdir(vbbdir)) if x.endswith('.vbb')])
            # parse seq to jpg, saved in setxx/frame
            if not osp.exists(framedir):
                os.makedirs(framedir)
                seqs = sorted([osp.join(seqdir, x) for x in sorted(os.listdir(seqdir)) if x.endswith('.seq')])
                for seq, vbb in zip(seqs, vbbs):
                    anno = parser.readvbb(vbb, setid)
                    if self.test_type == 'caltech' and i > 5:
                        imgs = parser.readseq(seq, setid) # parse all images
                    else:
                        imgs = parser.readseq(seq, setid, anno) # only parse image with label

                    # fetch_head
                    keyname = list(imgs.keys())[0] 
                    keyid = keyname.split('.')[0].split('_')[-1]
                    key_loc = keyname.find(keyid)
                    key_head = keyname[:key_loc]
                    keyinds = sorted(['{:0>6}'.format(x.split('.')[0].split('_')[-1]) for x in imgs.keys()])
                    # each seq contains many images
                    for ind in range(len(imgs.keys())):
                        if self.test_type == 'voc' or \
                          (self.test_type == 'caltech' and i > 5 and (ind+1) % 30 == 0): 
                            key = key_head + str(int(keyinds[ind])) + '.jpg'
                            cv2.imwrite(osp.join(framedir, key), imgs[key])

            # parse annotations
            annos = {}
            for vbb in vbbs:
                anno = parser.readvbb(vbb, setid)
                annos.update(anno)

            # save to voc
            imgfiles = sorted([os.path.join(framedir, x) for x in sorted(os.listdir(framedir)) if x.endswith('.jpg')])

            t = tqdm.tqdm()
            t.total = len(imgfiles)

            for imgfile in imgfiles:
                t.update()
                key = osp.basename(imgfile)
                # Annotations
                if self.test_type == 'voc' or \
                  (self.test_type == 'caltech' and i <= 5):
                    bboxes = parser.getbbox(annos[key])
                    ret = self.create_xml(bboxes, imgfile, cnt, annodir)
                    if not ret:
                        continue
                # JPEGImages
                cmd = 'ln -s {} {}'.format(
                    imgfile,
                    osp.join(jpgdir, '{:0>6}.jpg'.format(cnt)))
                (status, output) = commands.getstatusoutput(cmd)
                # ImageSets/Main
                if i < 5:
                    train_list.append("{:0>6}".format(cnt))
                elif i == 5:
                    val_list.append("{:0>6}".format(cnt))
                else:
                    test_list.append("{:0>6}".format(cnt))
                cnt += 1

        trainval_list = train_list + val_list
        self.create_split(splitdir, None, 
                trainval_list, 
                test_list,
                train_list,
                val_list)

        print("===> Successfully.")
        print("Caltech in voc-format is saved in {}".format(voc_root))
        return True

    
    def eval(self):
        print('===>') 
        print('Use matlab to eval detection performance.')
        print('Download below two files for evaluation')
        print('    1. Piotrs Matlab Toolbox: https://pdollar.github.io/toolbox/archive/piotr_toolbox.zip')
        print('    2. Caltech eval code: http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/code/code3.2.1.zip')


if __name__ == "__main__":
    ds = Caltech()
    ds.set_test_type('caltech')
    ret = ds.convert2voc()
    if ret:
        ds.create_fake_test_anno()
