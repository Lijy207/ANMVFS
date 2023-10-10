
clear;clc;
%% Load Dataset
FileName = ['wisconsin.mat'];
load(FileName);
Xdata = X_wisconsin;
Data = [Xdata{1}, Xdata{2}, Xdata{3}, Xdata{4}];
 Data = NormalizeFea(Data,0);
[m n] = size(Data);
Y = Y_wisconsin;

%% Parameter settings
opt.nsel = n;  %Number of features
percentage = 0.6; %Percentage of selected features
opt.lambda1 = 1; %Adjustable hyperparameter lambda1
%% Ten fold cross validation to obtain training and testing sets
    ind(:,1) = crossvalind('Kfold',size(find(Y),1),10);
%% 10-fold cross validation results
for k = 1:10
    test = ind(:,1) == k;
    train = ~test;
    %% Calling the ANMVFS function
    [W1,theta,alpha ] = ANMVFS( Data(train,:),Y(train,:),opt );
    %% Using the obtained variables for feature selection
    [~, idx2] = sort(theta, 'descend');
    num = ceil(percentage*opt.nsel);
    theta(idx2(1:num-1)) = 1;
    theta(idx2(num:opt.nsel)) = 0;
    SelectFeaIdx = find(theta~=0); %Index of selected features
end



