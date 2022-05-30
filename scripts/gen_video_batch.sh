#!/bin/bash

ORIGIN_DIR=$1
TARGET_DIR=$2
ORIGIN_LEN=${#ORIGIN_DIR}

for i in $(ls -d $ORIGIN_DIR/*)
do 
    # echo ${i%%/}
    # echo $ORIGIN_LEN
    echo ${i:ORIGIN_LEN+1}
    python -m runners.a1.a1-demo -f ${i%%/} -r $2/${i:ORIGIN_LEN+1}.mp4
done