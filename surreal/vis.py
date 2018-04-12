#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import glob
import scipy.io as sio
import cv2
import numpy as np

default_root = os.path.expanduser('~/data/cmu')


def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def parse_mp4(f):
    cap = cv2.VideoCapture(f)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc = 0
    ret = True
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()
    return buf


def vis_mask(img, mask, color, index=-1, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""

    img = img.astype(np.float32)
    if index == -1:
        idx = np.nonzero(mask)
    else:
        idx = np.where(mask == index)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * color

    if show_border:
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (255, 255, 255), border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)


def draw_mask_on_img(img, mask):
    assert img.shape[0] == mask.shape[0], 'Not same shape of img and mask'
    for i in np.unique(mask):
        if i == 0:
            continue
        color = colormap(True)[i]
        img = vis_mask(img, mask, color, index=i)
    return img


def vis(args):
    videos = sorted(glob.glob(os.path.join(args.surreal_root, args.split, '*/*/*.mp4')))
    depths = sorted(glob.glob(os.path.join(args.surreal_root, args.split, '*/*/*_depth.mat')))
    infos = sorted(glob.glob(os.path.join(args.surreal_root, args.split, '*/*/*_info.mat')))
    segms = sorted(glob.glob(os.path.join(args.surreal_root, args.split, '*/*/*_segm.mat')))

    cnt = 0
    
    for cnt in range(len(videos)):
        video = videos[cnt]
        segm = segms[cnt]

    #for video, segm in zip(videos, segms):
        imgs = parse_mp4(video)

        seg_dict = sio.loadmat(segm)
        # rewrite key
        for i, key in enumerate([ x for x in seg_dict.keys() if 'segm_' in x]):
            new_key = 'segm_{:0>2}'.format(key.split('_')[-1])
            seg_dict[new_key] = seg_dict.pop(key)
        seg_keys = sorted([ x for x in seg_dict.keys() if 'segm_' in x])

        for ind, img in enumerate(imgs):
            seg = seg_dict[seg_keys[ind+1]]
            img = draw_mask_on_img(img, seg)

            cv2.imshow('img', img)
            ch = cv2.waitKey(20)

        cnt += 100
        if cnt > 1000:
            break


def main(args):
    vis(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize human data in SURREAL')
    parser.add_argument('--surreal_root', default=default_root, type=str, help='surreal dataset root')
    parser.add_argument('--split', default='train', type=str, help='image split, train | val | test')
    args = parser.parse_args()
    main(args)
