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


class CityPersons(Pedestrian):

    """CityPersons Dataset
    https://bitbucket.org/shanshanzhang/citypersons
    CityPersons: A Diverse Dataset for Pedestrian Detection.
    """

    def __init__(self, root=osp.join(default_root, 'citypersons')):
        super(CityPersons, self).__init__('citypersons', root)
        self.classes = ('ignore', 'pedestrian', 'rider', 'sitting', 'otherperson', 'peoplegroup')
        self.index_to_class = dict(zip(range(len(self.classes)), self.classes))


    def download(self):
        print("Refer scripts/shell/fetch/pedestrian/cityperson.sh to download dataset.")


    def unzip(self):
        print("Do it by yourself.")


    def convert2voc(self):
        if os.path.exists(osp.join(self.root, self.voc_name)):
            return False
        voc_root, jpgdir, annodir, splitdir = self.create_voc()

        if not os.path.exists(os.path.join(self.root, 'leftImg8bit')):
            raise IOError('Image not found, please download and unzip')

        train_list = []
        val_list = []
        test_list = []

        train_imgdir = osp.join(self.root, 'leftImg8bit/train')
        val_imgdir = osp.join(self.root, 'leftImg8bit/val')
        test_imgdir = osp.join(self.root, 'leftImg8bit/test')

        train_annos = loadmat(osp.join(self.root, 'shanshanzhang-citypersons/annotations/anno_train.mat'))
        val_annos = loadmat(osp.join(self.root, 'shanshanzhang-citypersons/annotations/anno_val.mat'))

        t = tqdm.tqdm()
        t.total = train_annos['anno_train_aligned'].shape[1] + \
                  val_annos['anno_val_aligned'].shape[1]

        ind = 1
        for phase in ('train', 'val'):
            annos_mat = train_annos if phase is 'train' else val_annos 
            imgdir = train_imgdir if phase is 'train' else val_imgdir

            for anno in annos_mat['anno_{}_aligned'.format(phase)][0]:
                t.update()

                cityname = str(anno[0][0][0][0])
                imgname = str(anno[0][0][1][0])
                imgfile = osp.join(imgdir, cityname, imgname)

                anno_dict = {}
                anno_dict['id'] = '{:0>6}'.format(ind)
                anno_dict['bboxes'] = []

                for bbox_anno in anno[0][0][2]:
                    clsname = self.index_to_class[bbox_anno[0]]
                    bbox = bbox_anno[1:5] # using (x1, y1, w, h) rather than (x1_vis, ...)
                    bbox[2:] += bbox[:2] # (x,y,w,h) -> (x1,y1,x2,y2)

                    bbox_checked = self.check_anno(np.array([bbox]), imgfile, imagemagick=True)
                    if bbox_checked.shape[0] < 1:
                        continue
                    bbox_checked = bbox_checked[0]

                    anno_dict['bboxes'].append({'name':clsname, 'xyxy': bbox_checked.tolist()})

                if len(anno_dict['bboxes']) == 0:
                    continue

                anno_tree = self.anno2xml(imgfile, anno_dict)
                etree.ElementTree(anno_tree).write(
                    osp.join(annodir, '{:0>6}.xml'.format(ind)),
                    pretty_print=True)

                cmd = 'ln -s {} {}'.format(
                        imgfile,
                        osp.join(jpgdir, '{:0>6}.jpg'.format(ind)))
                (status, output) = commands.getstatusoutput(cmd)

                if phase is 'train':
                    train_list.append('{:0>6}'.format(ind))
                else:
                    val_list.append('{:0>6}'.format(ind))
                ind += 1

        # craete ImageSets JPEGImage for testset
        phase = 'test'
        test_imgs = glob.glob(test_imgdir+'/*/*.png')
        for imgfile in test_imgs:
            cmd = 'ln -s {} {}'.format(
                    imgfile,
                    osp.join(jpgdir, '{:0>6}.jpg'.format(ind)))
            (status, output) = commands.getstatusoutput(cmd)
            test_list.append('{:0>6}'.format(ind))
            ind += 1

        trainval_list = train_list + val_list
        self.create_split(splitdir, None, 
                trainval_list, test_list,
                train_list, val_list)

        print("===> Successfully.")
        print("Caltech in voc-format is saved in {}".format(voc_root))
        return True


    def eval():
        print('Refer to https://bitbucket.org/shanshanzhang/citypersons/src/f44d4e585d51d0c3fd7992c8fb913515b26d4b5a/evaluation/eval_script/')


if __name__ == "__main__":
    ds = CityPersons()
    ret = ds.convert2voc()
    if ret:
        ds.create_fake_test_anno()
