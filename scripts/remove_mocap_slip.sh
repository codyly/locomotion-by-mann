#!/bin/bash

ORIGIN_DIR=$1
TARGET_DIR=$2

for i in $(ls -d $ORIGIN_DIR/*)
do 
    echo ${i%%/}
    python -m tools.slip_eliminator -f ${i%%/} -o $2
done