clear
clc
rng(100);
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
        Ns = 10^4;
        nproblem = 50;
        C3 = -1/(na-1)*ones(na,na);   % C matrix for na options
        C3 = C3 - diag(diag(C3)) + eye(na);
        simName = sprintf('simMDF_mex_%d', na);

        dataset = {};
        for j = 1:nproblem
            D = zeros(ncomb, na);
            idx = zeros(ncomb, na);
            times = zeros(1, ncomb);
            MM = rand(nopts, 2) * 9 + 1;
            for i=2:nopts
                while(any(all(MM(1:i-1, :) > MM(i, :), 2))) % dominated by any
                    MM(i, :) = rand(1, 2) * 9 + 1;
                end
            end    
            sig2 = rand() + 0.1;
            b = floor(rand() * 15) + 1;
            phi1 = rand()/10 + 0.001;
            phi2 = rand()/10 + 0.001;
            theta = floor(rand() * 10) + 5;
            w = [0.3, 0.5, 0.7]; % random value between 0.3 and 0.7
            w = w(randi([1,3]));
            w = [w; 1 - w];

            for i = 1:ncomb
                id = randperm(nopts);
                id = id(1:na);
                while(any(all(idx == id, 2)))
                    id = randperm(nopts);
                    id = id(1:na);
                end
                M = MM(id, :);
                [G, EG] = distfunct(M,b,phi1,phi2);
                f = sprintf('%s(G,C3,M,w,theta,sig2,Ns)', simName);
                [p3, T] = eval(f); %simMDF_mex(G,C3,M,w,theta,sig2,Ns); 
                times(i) = T;
                D(i, :) = p3;
                idx(i, :) = id;
            end     
            dataset{j}.M = MM;                
            dataset{j}.times = times;
            dataset{j}.params = [b, phi1, phi2, sig2, theta, w(1)];
            dataset{j}.b = b;
            dataset{j}.phi1 = phi1;
            dataset{j}.phi2 = phi2;
            dataset{j}.sig2 = sig2;
            dataset{j}.threshold = theta;
            dataset{j}.w = w;
            dataset{j}.D = D;
            dataset{j}.idx = idx;
            disp("              Idx                              D")
            disp([idx D])

        end
        o = sprintf('../data/set_nopts%d_ncomb%d_nproblem%d_no%d.mat', nopts, ncomb, nproblem, na);
        save(o, 'dataset')
    end
end


