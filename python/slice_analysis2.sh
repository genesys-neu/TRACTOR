#!/bin/bash

FILEMARKER=noTs_oldcollect
TRAINLOGDIR=train_log4

#FILEMARKER=noTs_newcollect
#TRAINLOGDIR=train_log3

for l in 16 #4 8 16 32 64
do
  for m in emuc #emu co
  do
    # OLD DATASET (only single UE)
    #python ORAN_dataset.py --trials Trial1 Trial2 Trial3 Trial4 Trial5 Trial6 --mode $m --slicelen $l --data_type singleUE_raw --filemarker ${FILEMARKER} --drop_colnames Timestamp
    python torch_train_ORAN.py --ds_file SingleUE/dataset__${m}__Trial1_Trial2_Trial3_Trial4_Trial5_Trial6__slice${l}_singleUE_raw_${FILEMARKER}.pkl --isNorm --ds_path ../logs --cp_path ./${TRAINLOGDIR}/ --norm_param_path ../logs/global__cols_maxmin__${FILEMARKER}_slice${l}.pkl --exp_name ICNC_ctrlcheck__slice${l}__${FILEMARKER} --relabel_train --transformer v1
    python confusion_matrix.py --logdir ./${TRAINLOGDIR}/ICNC_ctrlcheck__slice${l}__${FILEMARKER}/

    # NEW DATASET (re-collected by Josh with different configuration, both Single and Multi UE data
    #python ORAN_dataset.py --trials Trial7 --trials_multi Trial4 Trial5 --mode $m --data_type singleUE_raw multiUE --slicelen $l --filemarker ${FILEMARKER} --drop_colnames Timestamp
    #python torch_train_ORAN.py --ds_file SingleUE/dataset__${m}__Trial7__slice${l}_singleUE_raw_${FILEMARKER}__globalnorm.pkl Multi-UE/dataset__emuc__Trial4_Trial5__slice${l}_multiUE_${FILEMARKER}__globalnorm.pkl --isNorm --ds_path ../logs --cp_path ./${TRAINLOGDIR}/ --norm_param_path ../logs/global__cols_maxmin__${FILEMARKER}_slice${l}.pkl --exp_name slice${l}__TransV1__nopos__relabel__${FILEMARKER} --relabel_train --transformer v1
    #python confusion_matrix.py --logdir ./${TRAINLOGDIR}/slice${l}__TransV1__nopos__relabel__${FILEMARKER}/


  done
done


# NOTES
# (rllib_py39) mauro@mauro-Alienware-Area-51-R7:~/Research/ORAN/repo/TRACTOR/python$ md5sum train_logs/no_timestamp__newCTRLCheck/model.16.trans_v1.pt
# badf648bf2ab8ce7b77fe133b3448ff9  train_logs/no_timestamp__newCTRLCheck/model.16.trans_v1.pt
# (rllib_py39) mauro@mauro-Alienware-Area-51-R7:~/Research/ORAN/repo/TRACTOR/python$ md5sum train_logs/last_working_model/model.16.trans_v1.pt
# badf648bf2ab8ce7b77fe133b3448ff9  train_logs/last_working_model/model.16.trans_v1.pt

# (rllib_py39) mauro@mauro-Alienware-Area-51-R7:~/Research/ORAN/repo/TRACTOR/python$ md5sum train_logs/last_working_model/global__cols_maxmin__noTimestamp.pkl
# ed4e03e96d256807aee611f36b1a5701  train_logs/last_working_model/global__cols_maxmin__noTimestamp.pkl
# (rllib_py39) mauro@mauro-Alienware-Area-51-R7:~/Research/ORAN/repo/TRACTOR/python$ md5sum ../logs/global__cols_maxmin__noTimestamp.pkl
# ed4e03e96d256807aee611f36b1a5701  ../logs/global__cols_maxmin__noTimestamp.pkl

