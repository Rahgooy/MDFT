function [mse, P3, TV] = fitMDFTs_multi(params, D, MM, idx, C3, Ns)
% function [sse, P3, T] = fitMDFTs_multi(params, D, MM, idx, C3, Ns)
% ccluates predicted and compare to observed

nc = size(D,1);
na = size(D,2);
P3 = zeros(nc,na);
TV = zeros(nc,1);

wgt = exp(params(1));
phi1 = exp(params(2)); 
phi2 = exp(params(3));
sig2 = exp(params(4));
theta1 = exp(params(5));
w = exp(params(6));
w = [w ; (1-w) ];  % weight vector for m attributes, m=2 in this case


for i = 1:nc
    M3 =MM(idx(i, :), :);
    [G3, ~] = distfunct(M3,wgt,phi1,phi2);  % returns gamma
    simName = sprintf('simMDF_mex_%d', size(G3, 2));
    f = sprintf('%s(G3,C3,M3,w,theta1,sig2,Ns)', simName);
    [p3, T] = eval(f);  %simMDF_mex(G3,C3,M3,w,theta1,sig2,Ns);     
    
   
    P3(i,:) =  p3;
    TV(i) = T;

end

dev = (D-P3);
mse = sum(sum(dev.*dev)) / nc;
   
   
  