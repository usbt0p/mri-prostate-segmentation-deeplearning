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
