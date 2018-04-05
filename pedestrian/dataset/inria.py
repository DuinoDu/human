# -*- coding: utf-8 -*-

import os
import os.path as osp
from base import Pedestrian, default_root
import re
from lxml import etree
import tqdm
try:
    import commands
except Exception as e:
    import subprocess as commands


class INRIAPerson(Pedestrian):

    """INRIA Person Dataset
    http://pascal.inrialpes.fr/data/human/
    """

    def __init__(self, root=osp.join(default_root, 'INRIAPerson')):
        super(INRIAPerson, self).__init__('INRIAPerson', root)
        self.filename = 'INRIAPerson.tar'


    def download(self):
        urlroot = 'ftp://ftp.inrialpes.fr/pub/lear/douze/data'
        remote_file = osp.join(urlroot, self.filename)
        zip_file = osp.join(self.root, self.filename)
        if not osp.exists(zip_file):
            print("===> Downloading INRIAPerson ...")
            cmd = 'wget {} -P {}'.format(remote_file, self.root)
            (status, output) = commands.getstatusoutput(cmd)
            output = output.split('\n')
            print("===> Successfully.")

    
    def unzip(self):
        zip_file = osp.join(self.root, self.filename)
        unzip_file = osp.join(self.root, self.name)
        if not osp.exists(unzip_file):
            print("===> Unzip INRIAPerson ...")
            cmd = 'tar xvf {} -C {}'.format(zip_file, self.root)
            (status, output) = commands.getstatusoutput(cmd)
            output = output.split('\n')
            print("===> Successfully.")
        self.set_root(unzip_file)


    def _parse_lst(self, lstfile):
        output = []
        with open(lstfile, 'r', encoding='latin-1') as fid:
            for line in fid.readlines():
                if line.startswith('Bounding box for object'):
                    (role, bbox_str) = line.strip().split(':', 1)
                    m = re.findall(r'(\w*[0-9]+)\w*', bbox_str)
                    xmin = int(m[0])
                    ymin = int(m[1])
                    xmax = int(m[2])
                    ymax = int(m[3])
                    output.append([xmin, ymin ,xmax, ymax])
        return output


    def convert2voc(self):
        if os.path.exists(osp.join(self.root, self.voc_name)):
            return
        voc_root, jpgdir, annodir, splitdir = self.create_voc()

        trainval_list = []
        test_list = []

        train_pos_img_list = osp.join(self.root, 'Train/pos.lst')
        train_anno_list = osp.join(self.root, 'Train/annotations.lst')


        jpgfiles = sorted([osp.join(self.root, x.strip()) for x in open(train_pos_img_list, 'r').readlines()])
        annofiles = sorted([osp.join(self.root, x.strip()) for x in open(train_anno_list, 'r').readlines()])

        sum_train = len(jpgfiles)

        test_pos_img_list = osp.join(self.root, 'Test/pos.lst')
        test_anno_list = osp.join(self.root, 'Test/annotations.lst')
        
        jpgfiles += sorted([osp.join(self.root, x.strip()) for x in open(test_pos_img_list, 'r').readlines()])
        annofiles += sorted([osp.join(self.root, x.strip()) for x in open(test_anno_list, 'r').readlines()])

        if len(jpgfiles) != len(annofiles):
            raise (ValueError, "len of image({}) != len of anno({})".format(len(jpgfiles), len(annofiles)))

        t = tqdm.tqdm()
        t.total = len(jpgfiles)

        for ind, (jpg, xml) in enumerate(zip(jpgfiles, annofiles)):
            ind += 1
            cmd = 'ln -s {} {}'.format(
                    jpg,
                    osp.join(jpgdir, '{:0>6}.jpg'.format(ind)))
            (status, output) = commands.getstatusoutput(cmd)

            bboxes = self._parse_lst(xml)
            anno_dict = {}
            anno_dict['id'] = '{:0>6}'.format(ind)
            anno_dict['bboxes'] = []
            for box in bboxes:
                anno_dict['bboxes'].append({'name':'person', 'xyxy': box})
            anno_tree = self.anno2xml(jpg, anno_dict)
            etree.ElementTree(anno_tree).write(
                osp.join(annodir, '{:0>6}.xml'.format(ind)),
                pretty_print=True)

            if ind > sum_train:
                test_list.append('{:0>6}'.format(ind))
            else:
                trainval_list.append('{:0>6}'.format(ind))
            t.update()

        self.create_split(splitdir, None, trainval_list, test_list)

        print("===> Successfully.")
        print("Caltech in voc-format is saved in {}".format(voc_root))


    def eval(self):
        print('===>') 
        print('Not found.')


if __name__ == "__main__":

    import sys
    if sys.version_info.major == 2:
        print("Only python3 support")
        sys.exit()
        
    #ds = INRIAPerson(root="H:\\Data\\INRIAPerson")
    ds = INRIAPerson()
    #ds.download()
    #ds.unzip()
    ds.convert2voc()
