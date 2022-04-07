#!/bin/bash

echo "Generate Forwarding Motion Cluster with Different Speeds"

for runs in {0..100}
do
	velocity=$(echo "$runs/50.0" | bc -l)
    cmd="python -m runners.a1.a1-forward -v $velocity"
    echo $cmd 
    python -m runners.a1.a1-forward -v $velocity
    echo "finished "$runs"/100"
done

echo "Done."