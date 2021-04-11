for alg in 'CSVI' 'CSVI_RSD'
do 
    for alpha in 10000 100000
    do
        python3 main.py --alg $alg --alpha $alpha --init_folder "results/sensitivity/" --init_title "Add_" get_init &
    done
done 


wait
echo -e 'Initialization across alpha done'


for alg in 'CSVI' 'CSVI_RSD'
do
    python3 main.py --alg $alg  --init_folder "results/sensitivity/" --init_title "Add_" --vi_folder "results/sensitivity/" --vi_title "alpha_Add_"  run_vi --vi_stepsched 5 &
done


wait
echo -e 'alpha sensitivity done'


