#!/bin/bash
export VO_CMS_SW_DIR="/cvmfs/cms.cern.ch"
export COIN_FULL_INDIRECT_RENDERING=1
export SCRAM_ARCH="slc6_amd64_gcc481"

start=$(($1*100))
end=$(($1*100+100))

echo $start $end

cd /het/p4/dshih/jet_images-deep_learning/ML4GW/flow_posterior/

# flowsamples=testpoints_fLDeltaf0.01_5p_withRN_forsteve_newRN_proj.npy_flowsamples.npy
# testpoints=testpoints_fLDeltaf0.01_5p_withRN_forsteve_newRN_proj.npy

# flowsamples=testpoints_fLDeltaf0.01_5p_withRN_forsteve_newRN_LSTM_flowsamples.npy
# flowsamples=testpoints_fLDeltaf0.01_5p_withRN_forsteve_newRN_SimpleDNN_flowsamples.npy
# python -u calculate_likelihood_flowsamples.py --start=${start} --end=${end} --samples=${flowsamples} --npulsar=5 --withRN

# flowsamples=testpoints_fLDeltaf0.01_10p_withRN_forsteve_newRN_LSTM_flowsamples.npy
# flowsamples=testpoints_fLDeltaf0.01_10p_withRN_forsteve_newRN_SinglePulsarStackedLSTM_v2_flowsamples.npy
flowsamples=testpoints_fLDeltaf0.01_10p_withRN_forsteve_newRN_UniformPrior_flowsamples.npy
python -u calculate_likelihood_flowsamples.py --start=${start} --end=${end} --samples=${flowsamples} --npulsar=10 --withRN

# trystr=1sttry
# tpfile=testpoints_fLDeltaf0.01_5p_withRN_forsteve_proj_${trystr}.npy
# python -u calculate_likelihood_flowsamples.py --start=${start} --end=${end} --testpoint=testpoints/${tpfile} --samples=testpoints/${tpfile}_flowsamples.npy --npulsar=5 --withRN

# trystr=2ndtry
# tpfile=testpoints_fLDeltaf0.01_5p_withRN_forsteve_proj_${trystr}.npy
# python -u calculate_likelihood_flowsamples.py --start=${start} --end=${end} --testpoint=testpoints/${tpfile} --samples=testpoints/${tpfile}_flowsamples.npy --npulsar=5 --withRN

# trystr=3rdtry
# tpfile=testpoints_fLDeltaf0.01_5p_withRN_forsteve_proj_${trystr}.npy
# python -u calculate_likelihood_flowsamples.py --start=${start} --end=${end} --testpoint=testpoints/${tpfile} --samples=testpoints/${tpfile}_flowsamples.npy --npulsar=5 --withRN

# trystr=4thtry
# tpfile=testpoints_fLDeltaf0.01_5p_withRN_forsteve_proj_${trystr}.npy
# python -u calculate_likelihood_flowsamples.py --start=${start} --end=${end} --testpoint=testpoints/${tpfile} --samples=testpoints/${tpfile}_flowsamples.npy --npulsar=5 --withRN

# trystr=5thtry
# tpfile=testpoints_fLDeltaf0.01_5p_withRN_forsteve_proj_${trystr}.npy
# python -u calculate_likelihood_flowsamples.py --start=${start} --end=${end} --testpoint=testpoints/${tpfile} --samples=testpoints/${tpfile}_flowsamples.npy --npulsar=5 --withRN

