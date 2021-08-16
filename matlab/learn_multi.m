clear
clc

Ns = 5000;  % no. of simulations per condition
nproblem = 50;

for nopts=[3, 5, 7, 10]% Number of options in each dataset
    combs = [1];
    na = nopts;
    if nopts == 7
        combs = [1, 3, 5, 10];
    end
    for ncomb=combs % Number of combinations of options that sampled
        if ncomb > 1
            na = 3;
        end
        for s = [1, 2, 3] % 1: estimate M, 2 estimate w, 3 both
            rng(100); % Set the seed
            learn(nopts, ncomb, nproblem, na, s, Ns, 1);
            rng(100); % Set the seed
            learn(nopts, ncomb, nproblem, na, s, Ns, 0);
        end
    end
end



