#!/bin/bash

#FILEMARKER=noTimestamp
FILEMARKER=noTs_newcollect
TRAINLOGDIR=train_log3

for l in 3 4 8 16 32 64
do
  for m in emuc #emu co
  do
    #python ORAN_dataset.py --trials Trial7 --trials_multi Trial4 Trial5 --mode $m --data_type singleUE_raw multiUE --slicelen $l --filemarker ${FILEMARKER} --drop_colnames Timestamp
    python torch_train_ORAN.py --ds_file SingleUE/dataset__emuc__Trial7__slice${l}_singleUE_raw_${FILEMARKER}__globalnorm.pkl Multi-UE/dataset__emuc__Trial4_Trial5__slice${l}_multiUE_${FILEMARKER}__globalnorm.pkl --isNorm --ds_path ../logs --cp_path ./${TRAINLOGDIR}/ --norm_param_path ../logs/global__cols_maxmin__${FILEMARKER}_slice${l}.pkl --exp_name slice${l}__TransV1__nopos__relabel__${FILEMARKER} --relabel_train --transformer v1
    python confusion_matrix.py --logdir ./${TRAINLOGDIR}/slice${l}__TransV1__nopos__relabel__${FILEMARKER}/
  done
done
