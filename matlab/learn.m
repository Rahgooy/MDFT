function [] = learn(nopts, ncomb, nproblem, na, s, Ns, pref_based)
% s = 1: estimate M, 2 estimate w, 3 both
tic
curr_time = toc;
names = ["M", "w", "Mw"];
type = ["time_based", "pref_based"];
type = type(pref_based + 1);
set = sprintf('set_nopts%d_ncomb%d_nproblem%d_no%d', nopts, ncomb, nproblem, na);
load(strcat('../data/', type, '/', set, '.mat'), 'dataset');

mkdir('../results')
mkdir('../results/MLE')
mkdir('../results/MLE/', type)
mkdir(strcat('../results/MLE/', type, '/', names(s)))
f = fopen(strcat('../results/MLE/', type, '/', names(s), '/', set, '.txt'), 'w');
fprintf(f, set);
fprintf(f, '\nNs: %d\n', Ns);
mse_list = zeros(1, 10);
results = {};
max_trials = 2;
for i = 1:size(dataset,2)
    disp(strcat('Estimating parameters for dataset #', string(i)))
    MM = dataset{i}.M;
    %         fprintf(1, print_3d(MM));
    params = dataset{i}.params;
    D = dataset{i}.D;
    idx = dataset{i}.idx;
    na = size(D,2);   % no of options in choice set
    C3 = -1/(na-1)*ones(na,na);   % C matrix for na options
    C3 = C3 - diag(diag(C3)) + eye(na);
    
    nc = size(D,1);
    na = size(D,2);
    
    options = optimset('Display','off', 'Maxiter', 1000);
    
    if s == 1
        best = 10000;
        for trial=1:max_trials
            M0 = unifrnd(1, 10, size(MM));
            p = log(params);
            [m, mse] = fminsearch(@(m) fitMDFTs_multi(p, D, exp(m), idx, C3, Ns, pref_based), log(M0), options);
            if mse < best
                M = exp(m);
                best = mse;
            end
        end
    end
    
    if s == 2
        best = 10000;
        for trial=1:max_trials
            w0 = log(0.5);
            [w, mse] = fminsearch(@(w) fitMDFTs_multi(([log(params(1:5)) w]), D, MM, idx, C3, Ns, pref_based), w0, options);
            M = MM;
            if mse < best
                params(6) = min(0.99, exp(w));
                best = mse;                
            end
        end
    end
    
    if s == 3
        best = 10000;
        for trial=1:max_trials
            M0 = unifrnd(1, 10, size(MM));
            w0 = log(0.5);
            x0 = [w0 reshape(log(M0), 1, numel(MM))];
            [x, mse] = fminsearch(@(x) fitMDFTs_multi(([log(params(1:5)) x(1)]), D, exp(reshape(x(2:end), size(MM))), idx, C3, Ns, pref_based), x0, options);
            if mse < best
                M = exp(reshape(x(2:end), size(MM)));
                params(6) = min(0.99, exp(x(1)));
                best = mse;
            end
        end
    end
    
    [mse, P3, TV] = fitMDFTs_multi(log(params), D, M, idx, C3, Ns, pref_based);
    mse_list(i) = mse;
    results{i} = {};
    results{i}.M = M;
    results{i}.actual_freq = D;
    results{i}.avg_t = TV;
    results{i}.b = params(1);
    results{i}.freq = P3;
    results{i}.iter = 0;
    results{i}.mse = mse;
    results{i}.phi1 = params(2);
    results{i}.phi2 = params(3);
    results{i}.sig2 = params(4);
    results{i}.threshold = params(5);
    results{i}.w = [params(6), 1-params(6)];
    results{i}.time = toc - curr_time;
    curr_time = toc;
    
    fprintf(f, '=========================================================\n');
    fprintf(f, '          Learning params for data #%d\n', i);
    fprintf(f, '=========================================================\n');
    fprintf(f, 'Actual Params: %s\n', mat2str(dataset{i}.params,4));
    fprintf(f, 'Predicted Params: %s\n', mat2str(params,4));
    fprintf(f, strcat('Actual M:', print_3d(MM), '\n'));
    fprintf(f, strcat('Predicted M:', print_3d(M), '\n'));
    fprintf(f, 'MSE: %f\n', mse);
    
    disp('MSE');
    disp(mse);
    disp('         target data           model predictions');
    disp([D P3 TV]);
end

fprintf(f, '\n\nMSE list: %s\n', mat2str(mse_list, 4));
fprintf(f, 'MSE mean: %f\n', mean(mse_list));
fprintf(f, 'MSE std: %f\n', std(mse_list));

fprintf(f, 'Time elapsed: %f\n', toc);
toc

dataset = strcat(set, '.mat');
nsamples = Ns;
ntest = Ns;
save(strcat('../results/MLE/', type, '/', names(s), '/', set, '.mat'), 'dataset', 'nsamples','ntest', 'results');
end