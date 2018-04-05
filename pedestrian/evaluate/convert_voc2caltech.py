#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import tqdm
try:
    import commands
except Exception as e:
    import subprocess as commands


def map_id(voc_id, imgdir):
    imgfile = os.path.join(imgdir, "{}.jpg".format(voc_id))
    
    cmd = "ls -lh {}".format(imgfile)
    (status, output) = commands.getstatusoutput(cmd)
    filename = output.split('\n')[0]\
                     .split(' ')[-1]\
                     .split('/')[-1]\
                     .split('.')[0]
    return filename.split('_')


def fetch_txtname(outdir, setID_videoID):
    setID, videoID= setID_videoID.split('_')
    txtdir = os.path.join(outdir, setID)
    if not os.path.exists(txtdir):
        os.makedirs(txtdir)
    return os.path.join(txtdir, videoID+'.txt')


def convert(args):
    results = [x.strip() for x in open(args.det_txt, 'r').readlines()]


    import pickle
    if os.path.exists('tmp.pkl'):
        with open('tmp.pkl', 'r') as fid:
            transform = pickle.load(fid)
    else:

        t = tqdm.tqdm()
        t.total = len(results) 

        print('parsing...')
        transform = {} 
        for res in results:
            t.update()
            res = res.split(' ')
            voc_id = res[0]
            cal_id = map_id(voc_id, args.imgdir)
            key = cal_id[0]+'_'+cal_id[1]

            if key not in transform.keys():
                transform[key] = {}
            if cal_id[2] not in transform[key].keys():
                transform[key][cal_id[2]] = []
            transform[key][cal_id[2]].append(res[1:])

        with open('tmp.pkl', 'w') as fid:
            pickle.dump(transform, fid)


    # saving to txt 
    for k, v in transform.items():
        filename = fetch_txtname(args.output, k)
        f = open(filename, 'w')
        frameids = sorted(map(int, v.keys())) 
        for frameid in frameids:
            for box in v[str(frameid)]:
                f.write(str(frameid) + ',')
                f.write(box[1]) 
                f.write(',')
                f.write(box[2])
                f.write(',')
                f.write(str(float(box[3]) - float(box[1])))
                f.write(',')
                f.write(str(float(box[4]) - float(box[2])))
                f.write(',')
                f.write(box[0] + '\n')
        f.close()


def main(args):
    convert(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert detections to caltech format')
    parser.add_argument('--det_txt', default='', type=str, help='detection result txt')
    parser.add_argument('--imgdir', default='', type=str, help='test image path')
    parser.add_argument('--output', default='output', type=str, help='output path')
    
    args = parser.parse_args()
    main(args)
