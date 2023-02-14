#!/bin/bash

outdir=$1
startup=$2
mincoeff=$3
maxcoeff=$4
direction=$5
numberofruns=$6

echo "Generate Turning Motion Cluster with Different Angles"


# for runs in {0..$iters}
for (( run=0;run<=$numberofruns;run++ ))
do
	coeff=$(echo "$mincoeff + $run*($maxcoeff-$mincoeff)/$numberofruns" | bc -l)
    echo $coeff
    cmd="python3 -m runners.a1.a1-turning -c $coeff -d $direction -s $startup -o $outdir"
    echo $cmd
    python3 -m runners.a1.a1-turning -c $coeff -d $direction -s $startup -o $outdir
    echo "finished $run/$numberofruns"
done

echo "Done."
