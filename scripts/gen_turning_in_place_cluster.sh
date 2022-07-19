#!/bin/bash

outdir=$1
startup=$2
mincoeff=$3
maxcoeff=$4
numberofruns=$5

echo "Generate Turning Motion Cluster with Different Angles"

for direction in left right
do
    for method in move no_move
    do 
        for (( run=0;run<=$numberofruns;run++ ))
        do
            coeff=$(echo "$mincoeff + $run*($maxcoeff-$mincoeff)/$numberofruns" | bc -l)
            echo $coeff
            cmd="python -m runners.a1.a1-turning-in-place -m $method -c $coeff -d $direction -s $startup -o $outdir"
            echo $cmd 
            python -m runners.a1.a1-turning-in-place -m $method -c $coeff -d $direction -s $startup -o $outdir
            echo "finished $run/$numberofruns"
        done
    done 
done 

echo "Done."