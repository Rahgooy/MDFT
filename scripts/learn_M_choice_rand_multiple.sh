#!/usr/bin/env bash
cd ..

for n in 20 30 50 100 150;
do
#    o=3
#    s=2
#    for i in `seq 1 100`;
#    do
#        input="data/random/set_${s}/set_hotaling_n${n}_l100_o${o}_${i}.pickle"
#        output="results/learn_m/random/set_${s}/n_${n}_l100_o${o}_${i}.txt"
#        python main.py --niter 1000 --ntest 10000 --i ${input} --o ${output} --m
#    done

    o=5
    s=5
    for i in `seq 1 10`;
    do
        input="data/random/set_o_${o}/set_hotaling_n${n}_l100_o${o}_${i}.pickle"
        output="results/learn_m/random/set_o_${o}/n_${n}_l100_o${o}_${i}.txt"
        python main.py --niter 1000 --ntest 10000 --i ${input} --o ${output} --m
    done

#    o=7
#    s=4
#    for i in `seq 1 100`;
#    do
#        input="data/random/set_${s}/set_hotaling_n${n}_l100_o${o}_${i}.pickle"
#        output="results/learn_m/random/set_${s}/n_${n}_l100_o${o}_${i}.txt"
#        python main.py --niter 1000 --ntest 10000 --i ${input} --o ${output} --m
#    done
#
#    o=9
#    s=5
#    for i in `seq 1 100`;
#    do
#        input="data/random/set_${s}/set_hotaling_n${n}_l100_o${o}_${i}.pickle"
#        output="results/learn_m/random/set_${s}/n_${n}_l100_o${o}_${i}.txt"
#        python main.py --niter 1000 --ntest 10000 --i ${input} --o ${output} --m
#    done
#
#    o=11
#    s=6
#    for i in `seq 1 100`;
#    do
#        input="data/random/set_${s}/set_hotaling_n${n}_l100_o${o}_${i}.pickle"
#        output="results/learn_m/random/set_${s}/n_${n}_l100_o${o}_${i}.txt"
#        python main.py --niter 1000 --ntest 10000 --i ${input} --o ${output} --m
#    done

done

