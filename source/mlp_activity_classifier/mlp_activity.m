%% Prepare workspace
clear;
clc;
close all;


%% Load data and target
load(".\results\final_data.mat")

INPUT_ACTIVITY = final_features_activities_matrix';
TARGET_ACTIVITY = full(ind2vec(final_activities_targets_vector'));

%% Parameters definition
trainFcn = 'trainlm'; 
% Selecting a size for the hidden layer
hiddenLayerSize = 35;

%% Creation of the Neural Network   
% Create a Pattern Recognition Network
net = patternnet(hiddenLayerSize, trainFcn);
net.trainParam.epochs=30;


%% Data splitting & Net parameters setting
% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose Plot Functions
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

%% Training & Testing the Network
% Train the Network
[net,tr] = train(net,INPUT_ACTIVITY,TARGET_ACTIVITY);

% Test the Network
test_x = INPUT_ACTIVITY(:, tr.testInd);
test_t = TARGET_ACTIVITY(:, tr.testInd);
test_y = net(test_x);
[c, ~] = confusion(test_t, test_y);
correct_classification_percentage = 100 * (1 - c);
disp(correct_classification_percentage)
fprintf("Correction Classification%%: %f",correct_classification_percentage)

% View the Network
%view(net)

% Plots
figure, plotconfusion(test_t,test_y)