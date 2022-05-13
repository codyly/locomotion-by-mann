#!/bin/bash

outdir=$1
startup=$2
minspeed=$3
maxspeed=$4
numberofruns=$5

echo "Generate Forwarding Motion Cluster with Different Speeds"


# for runs in {0..$iters}
for (( run=1;run<=$numberofruns;run++ ))
do
	velocity=$(echo "$minspeed + $run*($maxspeed-$minspeed)/$numberofruns" | bc -l)
    echo $velocity
    cmd="python -m runners.a1.a1-forward -v $velocity"
    echo $cmd 
    python -m runners.a1.a1-forward -v $velocity -o $outdir -s $startup
    echo "finished $run/$numberofruns"
done

echo "Done."