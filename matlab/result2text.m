clc
clear
Ns = 10000;
name = "Mw";
set = "set4";
load(strcat("results/RNN/", name, "/", set, "_", name, ".mat"))
load(strcat("data/", set, ".mat"))

f = fopen(strcat('results/RNN/', name, '/', set, '.txt'), 'w');
fprintf(f, set);
fprintf(f, '\nNs: %d\n', Ns); 

na = 3;   % no of options in choice set

C3 = -1/(na-1)*ones(na,na);   % C matrix for na options
C3 = C3 - diag(diag(C3)) + eye(na);
params = zeros(1, 6);
mse_list = zeros(1, 10);
for i = 1:size(results, 2)
    r = results{i};
    params = [r.b  r.phi1 r.phi2 r.sig2 double(r.threshold) r.w(1)];
    D = dataset{i}.D;
    M = r.M;
    idx = dataset{i}.idx;
    x0 = log(params);
    [mse, P3, TV] = fitMDFTs_multi(x0,D, M, idx, C3,Ns);
    mse_list(i) = mse;
    fprintf(f, '=========================================================\n');
    fprintf(f, '          Learning params for data #%d\n', i);
    fprintf(f, '=========================================================\n');
    fprintf(f, 'Actual Params: %s\n', mat2str(dataset{i}.params,4));
    fprintf(f, 'Predicted Params: %s\n', mat2str(params,4));
    fprintf(f, strcat('Actual M:', print_3d(dataset{i}.M), '\n'));
    fprintf(f, strcat('Predicted M:', print_3d(M), '\n'));
    fprintf(f, 'MSE: %f\n', mse);
end

fprintf(f, 'MSE list: %s\n', mat2str(mse_list, 4));
fprintf(f, 'MSE mean: %f\n', mean(mse_list));
fprintf(f, 'MSE std: %f\n', std(mse_list));

disp('mse')
disp(mse_list)
disp('mean')
disp(mean(mse_list))
disp('std')
disp(std(mse_list))
