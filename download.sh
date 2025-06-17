#!/bin/bash

cd /home/guest/work/Datasets;

curl -L -O https://github.com/DIAGNijmegen/picai_labels/archive/refs/heads/main.zip;
unzip main.zip -d picai_labels_all;



dir_count_and_space() {
    ls -l "$1" | grep -c ^d
    du -sh "$1"
}

read_metadata(){
    head -n 26 $1
}

# jupyter notebook --no-browser --port=8888 --ip=127.0.0.1 # on remote
# ssh -N -L 8888:localhost:8888 guest@77.26.202.187 # on local
