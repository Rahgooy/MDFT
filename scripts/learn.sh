names=(M w Mw)
learnM=(True False True)
learnw=(False True True)

for nopts in 3 5 7 10; do
  combs=(1)
  na=$nopts
  #if [ $nopts == 7 ]; then
  #  combs=(1 3 5 10)
  #fi
  for ncomb in ${combs[@]}; do # Number of combinations of options that sampled
    if (($ncomb > 1)); then
      na=3
    fi

    for s in 0 1 2; do
      for type in pref_based time_based; do
        python learn.py \
          --m ${learnM[$s]} \
          --w ${learnw[$s]} \
          --i "data/${type}/set_nopts${nopts}_ncomb${ncomb}_nproblem50_no${na}.mat" \
          --o "results/NN/${type}/${names[$s]}/set_nopts${nopts}_ncomb${ncomb}_nproblem50_no${na}"
      done
    done
  done
done

for nsamples in 20 30 50 150; do
  for type in time_based; do
    python learn.py \
      --m True \
      --w False \
      --ntrain $nsamples \
      --i "data/${type}/set_nopts5_ncomb1_nproblem50_no5.mat" \
      --o "results/NN/${type}/M/set_nopts5_ncomb1_nproblem50_no5_nsamples${nsamples}"
  done
done
