#!/bin/bash

for alg in 'CSVI' 'CSVI_RSD' 'SVI' 'SVI_Ind' 'SVI_OPT' 'SVI_SMAP'
do
    python3 main.py --alg $alg  get_init &
done

wait
echo -e 'initialization done'


for alg in 'CSVI' 'CSVI_RSD' 
do
    python3 main.py  --alg $alg run_vi --vi_stepsched 5 &
done

for alg in 'SVI' 'SVI_Ind' 'SVI_OPT' 'SVI_SMAP'
do 
    python3 main.py  --alg $alg  run_vi --vi_stepsched 15 &
done

wait
echo -e 'VI inference done'


for alg in 'CSVI' 'CSVI_RSD'
do 
    for alpha in 10 20 50 100 200 500 
    do
        python3 main.py --alg $alg --alpha $alpha --init_folder "results/sensitivity/" --init_title "alpha_" get_init &
    done
done 


wait
echo -e 'Initialization across alpha done'


for alg in 'CSVI' 'CSVI_RSD'
do
    python3 main.py --alg $alg  --init_folder "results/sensitivity/" --init_title "alpha_" --vi_folder "results/sensitivity/" --vi_title "alpha_elbo_"  run_vi --vi_stepsched 5 &
done


wait
echo -e 'alpha sensitivity done'


for alg in 'CSVI' 'CSVI_RSD' 'SVI' 'SVI_Ind' 'SVI_OPT' 'SVI_SMAP'
do
    for step in 5 10 15 20 25 30 
    do 
        python3 main.py --alg $alg  --vi_folder "results/sensitivity/" --vi_title "step_"  run_vi --vi_stepsched $step&
    done
done

wait
echo -e 'vi_stepsched sensitivity done'

