clear
clc

Ns = 5000;  % no. of simulations per condition
nproblem = 50;

for nopts=[3, 5, 7, 10]% Number of options in each dataset
    combs = [1];
    na = nopts;
    if nopts == 10
        combs = [1, 5, 10, 20];
    end
    for ncomb=combs % Number of combinations of options that sampled
        if ncomb > 1
            na = 3;
        end
        rng(100); % Set the seed
        for s = [1, 2, 3] % 1: estimate M, 2 estimate w, 3 both
            % learn(nopts, ncomb, nproblem, na, s, Ns, 1);
            learn(nopts, ncomb, nproblem, na, s, Ns, 0);
        end
    end
end



