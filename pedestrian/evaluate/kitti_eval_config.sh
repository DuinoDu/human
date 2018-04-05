#!/bin/bash

if [ ! -d kitti ];then
    mkdir kitti
fi
cd kitti
wget http://kitti.is.tue.mpg.de/kitti/devkit_object.zip
unzip devkit_object.zip
