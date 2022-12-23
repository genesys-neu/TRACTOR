#!/bin/bash
for d in /home/mauro/ray_results/traffic_class__slice_analysis/*
do
  if [[ -d "$d" ]]
  then
    for s in $d/*
    do
      if [[ -d "$s" ]]
      then
        python confusion_matrix.py --logdir $s
      fi
    done
  fi
done

