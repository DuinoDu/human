#!/bin/bash

if [ ! -d citypersons ];then
    mkdir citypersons
fi
cd citypersons

if [ ! -d ~/data/pedestrian/citypersons/shanshanzhang-citypersons ];then
    if [ ! -d ~/data/pedestrian/citypersons ];then
        echo "Download citypersons dataset first"
        exit
    fi
    _pwd=`pwd`
    cd ~/data/pedestrian/citypersons
    wget https://bitbucket.org/shanshanzhang/citypersons/get/c13bbdfa9862.zip
    unzip c13bbdfa9862.zip
    mv shanshanzhang-citypersons* shanshanzhang-citypersons
    cd $_pwd
fi

ln -s ~/data/pedestrian/citypersons/shanshanzhang-citypersons/evaluation
