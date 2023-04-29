#!/bin/bash
for l in 8 16 32 64 #4 #64 128
do
  for m in emuc #emu co
  do
    #python ORAN_dataset.py --trials Trial1 Trial2 Trial3 Trial4 Trial5 Trial6 --filemarker wCQI --slicelen $l --mode $m
    python torch_train_ORAN.py --ds_file dataset__${m}__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__slice${l}_wCQI.pkl --isNorm --num-workers 16
  done
done

