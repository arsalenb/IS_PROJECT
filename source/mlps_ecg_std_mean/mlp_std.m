%% Prepare workspace
clear;
clc;
close all;

%% Load data and target
load(".\results\final_data.mat")

INPUTS = final_features_ecg_std_matrix';
TARGETS = final_ecg_std_targets_vector';

%% Parameters definition
% Choosing a Training Function:
trainFcn = 'trainbr';

% Selecting a size for the hidden layer
hiddenLayerSize = 20;

%% Creation of the Neural Network
net = fitnet(hiddenLayerSize,trainFcn);


%% Data splitting & Net parameters setting
% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 20/100;
net.trainParam.epochs=30;

% Choose a Performance Function
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

%% Training & Testing the Network
[net,tr] = train(net,INPUTS,TARGETS);

y = net(INPUTS);

% View the Network
view(net)

% Plotting
figure, plotregression(TARGETS,y)


