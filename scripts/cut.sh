#!/bin/bash

ORIGIN_DIR=$1
TARGET_DIR=$2

for i in $(ls -d $ORIGIN_DIR/*)
do 
    echo ${i%%/}
    python -m tools.cutter -f ${i%%/} -o $2 -s $3
done