#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../dataset"))
from vbb import SeqVbb
import os.path as osp
import numpy as np
import cv2


def fetch_head(imgs):
    keyname = list(imgs.keys())[0] 
    keyid = keyname.split('.')[0].split('_')[-1]
    key_loc = keyname.find(keyid)
    return keyname[:key_loc]


def draw_detections(img, detections, ind, score_th=0.5):
    for key, detection in detections.items():
        cur_dets = detection[np.where(detection[:, 0] == ind)[0]]
        for det in cur_dets:
            if det[-1] < score_th:
                continue
            x1 = int(det[1])
            y1 = int(det[2])
            x2 = int(det[3]) + x1
            y2 = int(det[4]) + y1
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return img


def display(args):
    if not os.path.exists(args.caltech_root):
        print("Caltech dataset not found. Please download.")
        return

    res_path = osp.join(args.caltech_root, 'res')
    if not os.path.exists(res_path):
        print("detection results not found, please downloand or create your own.")
        return

    setid = 'set{:0>2}'.format(args.setid)
    videoid = "V{:0>3}".format(args.videoid)
    print('parsing '+ setid + "/" + videoid)

    # get images and gt
    seqdir = osp.join(args.caltech_root, setid)
    vbbdir = osp.join(args.caltech_root, "annotations", setid)
    vbbs = sorted([osp.join(vbbdir, x) for x in sorted(os.listdir(vbbdir)) if x.endswith('.vbb')])
    seqs = sorted([osp.join(seqdir, x) for x in sorted(os.listdir(seqdir)) if x.endswith('.seq')])
    assert args.videoid >= 0 and args.videoid < len(seqs), "Unknown video id: {}".format(args.videoid)
    seq = seqs[args.videoid]
    vbb = vbbs[args.videoid]
    parser = SeqVbb() 
    annos = parser.readvbb(vbb, setid)
    imgs = parser.readseq(seq, setid)

    # get detections
    methods = [x for x in os.listdir(res_path) if not x.endswith('.zip')]
    #methods = ['SDS-RCNN']
    methods = ['FasterRCNN']
    print(methods)
    detections = {}
    for method in methods:
        det_file = os.path.join(res_path, method, setid, videoid+'.txt')
        lines = [x.strip() for x in open(det_file, 'r').readlines()]
        det = np.zeros((len(lines), 6)).astype(np.float32)
        for line_ind, line in enumerate(lines):
            line = [float(x) for x in line.split(',')]
            det[line_ind] = line
        detections[method] = det

    # loop imgs
    keyinds = sorted(['{:0>6}'.format(x.split('.')[0].split('_')[-1]) for x in imgs.keys()])
    key_head = fetch_head(imgs) 
    for ind in range(len(imgs.keys())):
        if args.x30 and (ind+1) % 30 != 0:
            continue

        key = key_head + str(int(keyinds[ind])) + '.jpg'
        img = imgs[key]

        # draw gt
        if key in annos.keys():
            bboxes = parser.getbbox(annos[key])
            for bbox in bboxes:
                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        # draw detections
        if args.x30 and (ind+1) % 30 == 0:
            img = draw_detections(img, detections, ind+1, score_th=0.7)

        cv2.putText(img, '{}/{}'.format(ind+1, len(imgs.keys())), 
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),2)
        cv2.imshow('img', img)
        ch = cv2.waitKey(1000 if args.x30 else 5) & 0xff
        if ch == 27: #ord('q')
            break


def main(args):
    display(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display detection results in caltech format.')
    parser.add_argument('--caltech_root', default=os.environ['HOME']+'/data/pedestrian/caltech', type=str, help='detection results path')
    parser.add_argument('--setid', default=6, type=int, help='video set index, [6,7,8,9,10]')
    parser.add_argument('--videoid', default=0, type=int, help='video index, start from 0')
    parser.add_argument('--x30', default=1, type=int, help='only show 30-th frame')
    args = parser.parse_args()
    main(args)
