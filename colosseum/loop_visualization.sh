#!/bin/bash
for t in raw/*
do
  echo "Processing $t"
  python visualize_inout.py --path $t
done
