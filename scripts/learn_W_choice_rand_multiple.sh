#!/usr/bin/env bash
cd ..
for o in 5 7 9;
do
    for n in 20 30 50 100 150;
    do
        for i in `seq 1 100`;
        do
            input="data/random/set_o_${o}/set_hotaling_n${n}_l100_o${o}_${i}.pickle"
            output="results/learn_w/random/set_o_${o}/n_${n}_l100_o${o}_${i}.txt"
            python main.py --niter 1000 --ntest 10000 --i ${input} --o ${output} --w
        done
    done
done

