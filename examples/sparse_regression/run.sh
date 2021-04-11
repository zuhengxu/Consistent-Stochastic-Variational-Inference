#!/bin/bash

for dataset in "SYN" 
do
    for mu in "Prior" "SMAP_adam"
    do
        for L in  "Ind" "Random"
        do
            python3 main.py --dataset $dataset --mu_scheme $mu --L_scheme $L get_init &
        done
    done
done

wait 
echo -e 'SYN Initialization done'


for dataset in "SYN"
do  
    for ID in 1 2 3 4 5 
    do 
        python3 main.py --dataset $dataset --mu_scheme "Prior" --L_scheme "Ind" --trial $ID run_vi --vi_alg "SVI_adam" &
        python3 main.py --dataset $dataset --mu_scheme "Prior" --L_scheme "Random" --trial $ID run_vi --vi_alg "SVI_adam" &
        python3 main.py --dataset $dataset --mu_scheme "SMAP_adam" --L_scheme "Ind" --trial $ID run_vi --vi_alg "CSVI_adam" &
        python3 main.py --dataset $dataset --mu_scheme "SMAP_adam" --L_scheme "Random" --trial $ID run_vi --vi_alg "CSVI_adam" &
    done
done

wait 
echo -e 'SYN VI done'



for dataset in "REAL" 
do
    for mu in  "SMAP_adam"
    do
        for L in  "Ind" "Random"
        do
            for ID in 1 2 3 4 5 
            do
                python3 main.py --dataset $dataset --trial $ID --mu_scheme $mu --L_scheme $L  --alpha 1 get_init --map_stepsched 'lambda itr: 0.002'&
            done
        done
    done
done

for L in "Ind" "Random"
do
    python3 main.py --dataset "REAL" --mu_scheme "Prior" --L_scheme $L  get_init --map_stepsched 'lambda itr: 0.002'&
done

wait 
echo -e 'REAL Initialization done'


for dataset in "REAL"
do  
    for ID in 1 2 3 4 5  
    do 
        python3 main.py --dataset $dataset --mu_scheme "Prior" --L_scheme "Ind" --trial $ID run_vi --vi_alg "SVI_adam" --vi_stepsched 'lambda iter: 0.0002'&
        python3 main.py --dataset $dataset --mu_scheme "Prior" --L_scheme "Random" --trial $ID run_vi --vi_alg "SVI_adam" --vi_stepsched 'lambda iter: 0.0002'&
        python3 main.py --dataset $dataset --mu_scheme "SMAP_adam" --L_scheme "Ind" --trial $ID run_vi --vi_alg "CSVI_adam" --vi_stepsched 'lambda iter: 0.0002'&
        python3 main.py --dataset $dataset --mu_scheme "SMAP_adam" --L_scheme "Random" --trial $ID run_vi --vi_alg "CSVI_adam" --vi_stepsched 'lambda iter: 0.0002'&
    done
done

wait
echo -e 'REAL VI done'
