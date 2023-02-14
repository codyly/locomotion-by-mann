#!/bin/bash

ORIGIN_DIR=$1
TARGET_DIR=$2

for i in $(ls -d $ORIGIN_DIR/*)
do
    echo ${i%%/}
    python3 -m tools.slip_eliminator -f ${i%%/} -o $2 -t 0.01
done
