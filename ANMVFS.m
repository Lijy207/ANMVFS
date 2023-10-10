function [ WW,theta,alpha ] = ANMVFS( X1,Y1,para)
%% Anti-Noise Muiti-View Feature Selection With Sample Constraints
% by Jiaye Li, Hang Xu, Hao Yu, Weixin Li, Shichao Zhang
%  X1:   training data
%  Y1:   Class labels for training data
%  theta: Weights of features


%% Initialization
Dataa=X1; 

% Data1 = Dataa(:,1:93);
% Data2 = Dataa(:,94:186);
% Data3 = Dataa(:,187:189);
% Data4 = {Data1,Data2,Data3};
%% cornell
% Data1 = Dataa(:,1:1703);
% Data2 = Dataa(:,1704:1898);
% Data3 = Dataa(:,1899:2093);
% Data4 = Dataa(:,2094:2288);
% Data5 = {Data1,Data2,Data3,Data4};

%% texas
% Data1 = Dataa(:,1:1703);
% Data2 = Dataa(:,1704:1890);
% Data3 = Dataa(:,1891:2077);
% Data4 = Dataa(:,2078:2264);
% Data5 = {Data1,Data2,Data3,Data4};

%% washingdon
% Data1 = Dataa(:,1:1703);
% Data2 = Dataa(:,1704:1933);
% Data3 = Dataa(:,1934:2163);
% Data4 = Dataa(:,2164:2393);
% Data5 = {Data1,Data2,Data3,Data4};
%% wisconsin
Data1 = Dataa(:,1:1703);
Data2 = Dataa(:,1704:1968);
Data3 = Dataa(:,1969:2233);
Data4 = Dataa(:,2234:2498);
Data5 = {Data1,Data2,Data3,Data4};


WW = [];
for v = 1:4    % 4 views
    X = cell2mat(Data5(v));
    [n,d] = size(X);
    [~,c] = size(Y1);
    lambda1 = para.lambda1;
    %% Initialization
    W = rand(d,c);
    Theta=eye(d)/d;
    beta = rand(n,1);
    Uv = diag(sqrt(beta));
    onec = ones(c,1);
    alpha = rand(4,1);
    p = 2;
    iter = 1;
     obji = [];   
    %% Initialization
    bv = rand(n,1);
    beta0    = zeros(n,1);
    idx   = randperm(n);
    beta0(idx(1:ceil(n/2)))=1;  
    Uv0 = diag(sqrt(beta0));
    Qv = alpha(v)^p.*Uv0*X;
    Gv = alpha(v)^p.*Uv0*Y1;
    Av = alpha(v)^p.*Uv0;
    W0 = (Qv'*Qv + lambda1*(Theta^-2))\(Qv'*Gv - Qv'*Av*bv*onec');
    res  = abs(sum(alpha(v)^p.*X*Theta*W0 + bv*onec' - Y1,2).^2);
    maxres = max(res);
    L_med    = 1/(2*maxres^2);
    param.gamma = 2*maxres^2;
    param.type = 'mix_var';
    type=param.type;
    switch param.type
        case 'hard'
            K = L_med;
        case 'linear'
            K = L_med;
        case 'log'
            K = L_med;
        case 'mix'
            param.gamma = 2*L_med;
            K           = 1/param.gamma;
        case 'mix_var'
            param.gamma = 2*maxres^2;
            K           = L_med;
    end
 %% Iterative update of various variables
    while 1
        %% Update b      
        Uv = diag(sqrt(beta));
        Qv = alpha(v)^p.*Uv*X;
        Gv = alpha(v)^p.*Uv*Y1;
        Av = alpha(v)^p.*Uv;  
        bv = pinv(c.*Av'*Av)*(Av'*Gv*onec - Av'*Qv*Theta*W*onec);   
        %%  Update W and theta
        W=(Qv'*Qv + lambda1*(Theta^-2))\(Qv'*Gv - Qv'*Av*bv*onec');
        temp=sum(W.*W,2).^(1/(2))+ eps;
        Theta=diag( temp/ sum(temp) ).^(1/2); 
        %%  Calculate lv
        l(v) = norm(Uv*X*Theta*W + Uv*bv*onec' - Uv*Y1, 'fro')^2;
        %% Update beta
        res  = abs(sum(alpha(v)^p.*X*Theta*W + bv*onec' - Y1,2).^2);
        beta = eval_spreg(res, K, param);
        wei0 = find(beta==0);
        if ~isempty(wei0)
        K       =  K/1.35;
        end
        iter = iter + 1;
        if  iter==21,   break,   end
            
    end       
    WW = cat(1,WW,W);
end
%% solving alpha
l(l==0) = eps;
for v1 = 1:4
    alpha(v1) = (l(v1)^(1/(1-p)))/sum(power(l,1/(1-p)));
end
theta=sum(WW.*WW,2).^(1/2);
theta=theta/sum(theta);
% [~, ranked] = sort(theta, 'descend');
end

