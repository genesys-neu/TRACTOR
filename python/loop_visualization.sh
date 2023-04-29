#!/bin/bash
for t in `ls -d ../colosseum/raw/*04_10*` # ../colosseum/raw/*
do
  for s in 4 8 16 32 #64
  do
    echo "Processing $t slicelen $s"
    python visualize_inout.py --path $t --mode inference --slicelen $s --model ../model/model.$s.cnn.pt --chZeros=True
  done
done
