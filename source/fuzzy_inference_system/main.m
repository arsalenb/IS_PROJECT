clear;
close all;
clc;

%% Constants

% SEQUENTIALFS
FINAL_SELECTED_FEATURES = 5;
SEQUENTIALFS_HIDDEN_LAYER_SIZE = 18;

% ANFIS
defuzzification_method = "wtaver";
n_membership_functions = 3; % in case of grid partition
input_fuction_type = 'gaussmf';
output_function_type = 'linear';
num_epochs = 15;

% Percentage of data for testing and validation
percentage_training = 0.8; % 80% for testing
percentage_validation = 0.2; % 20% for validation

rng("default");
figure_id = 1;

%% Extract the 5 most relevant features

load('./results/final_data');

opts = statset('Display', 'iter', 'UseParallel', true);

% Select the most relevant features for the activity target
[fs, ~] = sequentialfs( ...
    @(input_train, target_train, hidden_layer_size)feature_selection(input_train, target_train, SEQUENTIALFS_HIDDEN_LAYER_SIZE) , ...
    final_features_activities_matrix, ...
    final_activities_targets_vector, ...
    'cv', 'none', ...
    'opt', opts, ... 
    'nfeatures', FINAL_SELECTED_FEATURES);

% Prepare the fis matrices
fis_features_activities_matrix = final_features_activities_matrix(:, fs);
fis_activities_targets_vector = final_activities_targets_vector;

save('./results/fis_data', ...
    'fis_features_activities_matrix', ...
    'fis_activities_targets_vector');
%% Load and prepare dataset 

load('./results/fis_data');

% Partition the dataset into two equal set for training and checking
dataset = [fis_features_activities_matrix fis_activities_targets_vector];

% Generate random indices for shuffling the rows
numRows = size(dataset, 1);
idx = randperm(numRows);

% Calculate the number of rows for testing and checking datasets
numRows_testing = round(percentage_training * numRows);
numRows_checking = round(percentage_validation * numRows);

% Extract rows for training and validationg dataset
training_data = dataset(idx(1:numRows_testing), :);
checking_data = dataset(idx(numRows_testing+1:numRows_testing+numRows_checking), :);

%% Generate TSK-FIS using gridPartition

options = genfisOptions("GridPartition");
options.NumMembershipFunctions = 2;

% Set input and output memberships fuctions types
options.InputMembershipFunctionType = input_fuction_type;
options.OutputMembershipFunctionType = output_function_type;

fisin = genfis(training_data(:, 1:end-1),training_data(:, end),options);

%% Train the ANFIS

[in,out,rule] = getTunableSettings(fisin);
opt = anfisOptions('EpochNumber',num_epochs);
fisout = tunefis(fisin,[in;out],training_data(:, 1:end-1),training_data(:, end),tunefisOptions("Method","anfis","MethodOptions",opt));

%% Test the ANFIS

fisout.DefuzzificationMethod = defuzzification_method;

% Predict the output
y = evalfis(fisout, checking_data(:, 1:end-1));

% Round the predicted value to the nearest integer value
y_rounded = round(y);

% Clamp the values to be within the range [1, 3]
y_rounded = max(min(y_rounded, 3), 1);

% Encode output and target to generate the confusion matrix
encoded_y = full(ind2vec(y_rounded'));
encoded_t = full(ind2vec(checking_data(:, end)'));

figure(1);
plotconfusion(encoded_t, encoded_y);
saveas(1, './results/anfis_confusion_matrix', 'png');

%% Evaluate and plot results

% Evaluate and print correct classification percentage
[c, ~] = confusion(encoded_t, encoded_y);
correct_classification_percentage = 100 * (1 - c);
fprintf("Correct classification: %f%%\n",correct_classification_percentage);

% Plot input functions
figure;
subplot(2,1,1);
plotmf(fisout, 'input', 1); % Change '1' to the index of your input variable if you have multiple input variables
title('Input Membership Functions');
