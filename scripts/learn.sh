names=(M w Mw)
learnM=(True False True)
learnw=(False True True)
for nopts in 3 5 7 10; do
  combs=(1)
  na=$nopts
  if [ $nopts == 10 ]; then
    combs=(1 5 10 20)
  fi
  for ncomb in ${combs[@]}; do # Number of combinations of options that sampled
    if (($ncomb > 1)); then
      na=3
    fi

    for s in 0 1 2; do
      python learn.py \
        --m ${learnM[$s]} \
        --w ${learnw[$s]} \
        --i "data/set_nopts${nopts}_ncomb${ncomb}_nproblem50_no${na}.mat" \
        --o "results/NN/${names[$s]}/set_nopts${nopts}_ncomb${ncomb}_nproblem50_no${na}"
    done
  done
done
