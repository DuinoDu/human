#!/bin/bash

function clr() {
    if [ -d __MACOSX ];then
        rm __MACOSX -r
    fi
}

if [ ! -d caltech ];then
    mkdir caltech
fi
cd caltech

curl https://pdollar.github.io/toolbox/archive/piotr_toolbox.zip -o piotr_toolbox.zip
wget http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/code/code3.2.1.zip 

unzip piotr_toolbox.zip && clr
mkdir code3.2.1
cd code3.2.1
unzip ../code3.2.1.zip && clr

mkdir data-USA
cd data-USA
caltech_root="$HOME/data/pedestrian/caltech"
if [ ! -d $caltech_root ];then
    echo "Please download Caltech dataset first"
    exit
fi
ln -s $caltech_root/annotations
if [ ! -d $caltech_root/res ];then
    mkdir $caltech_root/res
fi
ln -s $caltech_root/res
cd res

# download method result
res_root="http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res"
methods=(\
    "ACF" \
    "ACF++" \
    "Checkerboards" \
    "Checkerboards+" \
    "ConvNet" \
    "DeepCascade" \
    "DeepCascade+" \
    "HOG" \
    "MS-CNN" \
    "RPN+BF" \
    "SA-FastRCNN" \
    "SDS-RCNN" \
    "VJ")
for m in ${methods[*]} ; do
    echo "fetching ${m}.zip"
    wget $res_root"/"$m".zip"
    unzip ${m}".zip" && clr
done

cd ../../
echo "addpath(genpath('../toolbox'));" > runeval.m 
echo "addpath(genpath('./'));" >> runeval.m
echo "savepath;" >> runeval.m
echo "dbEval;" >> runeval.m

echo ""
echo "Caltech evaluation env are saved in `pwd`/caltech"
echo "Add your algorithm and runm caltech/code3.2.1/runeval.m"
echo "Install matlab-linux first."
