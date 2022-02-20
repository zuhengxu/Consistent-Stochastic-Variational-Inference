#!/bin/bash

##############
## synthetic example
##############
for dataset in "SYN" 
do
	for ID in 1 2 3 4 5
	do
		for mu in "Prior" "SMAP_adam"
		do
			for L in  "Ind" "Random"
			do
				python3 main.py --dataset $dataset --mu_scheme $mu --L_scheme $L get_init &
			done
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
        python3 main.py --dataset $dataset --mu_scheme "SMAP_adam" --L_scheme "Ind" --trial $ID run_vi --vi_alg "SVI_adam" &
        python3 main.py --dataset $dataset --mu_scheme "Prior" --L_scheme "Random" --trial $ID run_vi --vi_alg "SVI_adam" &
        python3 main.py --dataset $dataset --mu_scheme "SMAP_adam" --L_scheme "Ind" --trial $ID run_vi --vi_alg "CSVI_adam" &
        python3 main.py --dataset $dataset --mu_scheme "SMAP_adam" --L_scheme "Random" --trial $ID run_vi --vi_alg "CSVI_adam" &
        python3 main.py --dataset $dataset --mu_scheme "SMAP_adam" --L_scheme "Random" --trial $ID run_vi --vi_alg "CSL" --vi_stepsched 'lambda iter: 0.01/(1+iter)' &
        python3 main.py --dataset $dataset --mu_scheme "Prior" --L_scheme "Random" --trial $ID run_vi --vi_alg "CSL" --vi_stepsched 'lambda iter: 0.01/(1+iter)' &
    done
done

### regularized SVI
# for dataset in "SYN"
# do 
#     for ID in 1 2 3 4 5
#     do
#         for lmd in {0.1,0.5,1.0,2.0,5.0}
#         do 
#             python3 main.py --dataset $dataset --mu_scheme "Prior" --L_scheme "Random" --trial $ID run_rsvi --vi_alg "RSVI" --regularizer $lmd &
#         done 
#     done 
# done 

# wait 
# echo -e 'SYN RSVI done'


# for dataset in "syn"
# do 
#     for id in 1 2 3 4 5 
#     do
#         for lmd in {0.1,0.5,1.0,2.0,5.0}
#         do 
#             python3 main.py --dataset $dataset --mu_scheme "SMAP_adam" --L_scheme "Ind" --trial $ID run_rsvi --vi_alg "RSVI" --regularizer $lmd &
#         done 
#     done 
# done 
# wait 
# echo -e 'SYN VI done'

for dataset in "SYN"
do 
    for ID in 1 2 3 4 5
    do
        for lmd in {0.1,0.3} 
        do 
            python3 main.py --dataset $dataset --mu_scheme "Prior" --L_scheme "Random" --trial $ID run_rsvi --vi_alg "RSVI" --regularizer $lmd &
        done 
    done 
done 


for dataset in "SYN"
do 
    for id in 1 2 3 4 5 
    do
        for lmd in {0.1,0.3} 
        do 
            python3 main.py --dataset $dataset --mu_scheme "SMAP_adam" --L_scheme "Ind" --trial $ID run_rsvi --vi_alg "RSVI" --regularizer $lmd &
        done 
    done 
done 
wait 
echo -e 'SYN VI done'

##############
## real-data example
##############
for dataset in "REAL" 
do
    for mu in  "SMAP_adam"
    do
        for L in  "Ind" "Random"
        do
            for ID in {1..10}
            do
                python3 main.py --dataset $dataset --trial $ID --mu_scheme $mu --L_scheme $L  --alpha 0.03 get_init --map_stepsched 'lambda itr: 0.0005'&
            done
        done
    done
done

for L in "Ind" "Random"
do
    python3 main.py --dataset "REAL" --mu_scheme "Prior" --L_scheme $L  get_init --map_stepsched 'lambda itr: 0.0005'&
done

wait 
echo -e 'REAL Initialization done'


for dataset in "REAL" 
do  
    for ID in {1..20}  
    do 
        python3 main.py --dataset $dataset --mu_scheme "Prior" --L_scheme "Ind" --trial $ID run_vi --vi_alg "SVI_adam" --vi_stepsched 'lambda iter: 0.0002'&
        python3 main.py --dataset $dataset --mu_scheme "Prior" --L_scheme "Random" --trial $ID run_vi --vi_alg "SVI_adam" --vi_stepsched 'lambda iter: 0.0002'&
        python3 main.py --dataset $dataset --mu_scheme "SMAP_adam" --L_scheme "Ind" --trial $ID run_vi --vi_alg "CSVI_adam" --vi_stepsched 'lambda iter: 0.0002'&
        python3 main.py --dataset $dataset --mu_scheme "SMAP_adam" --L_scheme "Random" --trial $ID run_vi --vi_alg "CSVI_adam" --vi_stepsched 'lambda iter: 0.0002'&
        python3 main.py --dataset $dataset --mu_scheme "SMAP_adam" --L_scheme "Ind" --trial $ID run_vi --vi_alg "CSL" --vi_stepsched 'lambda iter: 0.01' &
        python3 main.py --dataset $dataset --mu_scheme "Prior" --L_scheme "Random" --trial $ID run_vi --vi_alg "CSL" --vi_stepsched 'lambda iter: 0.01' &
    done
done

wait
echo -e 'REAL VI done'
