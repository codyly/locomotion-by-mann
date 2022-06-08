#!/bin/bash

outdir=$1
numberofruns=$2

echo "Generate Randomly Mixed Profiles"

# for runs in {0..$iters}
for (( run=1;run<=$numberofruns;run++ ))
do
    cmd="python -m runners.a1.a1-random-mix -o $outdir"
    echo $cmd 
    python -m runners.a1.a1-random-mix -o $outdir
    echo "finished $run/$numberofruns"
done

echo "Done."