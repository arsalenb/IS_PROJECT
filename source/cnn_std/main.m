clear;
close all;
clc;

%% Constants

dataset_folder = '.\..\data';

% CNN Architecture parameters
num_channels = 11;
num_filters = [96 256 348 348];
filter_size = [11 5 3 3];
convolutional_stride = [4 4 4 4];
pooling_compression = [2 2 2];
pooling_stride = [2 2 2];
hidden_layer_size = 256;
output_layer_size = 1;
L2_regularization = 0.01;

% CNN Training parameters
epochs_number = 30;
mini_batch_size = 80;
initial_learn_rate = 0.01;
learn_rate_schedule = 'piecewise';
learn_rate_drop_period = 20;
learn_rate_drop_factor = 0.1;

% Dataset partitioning
window_size = 5000;
testing_set_ratio = 0.2;

rng("default");

%% Generate the dataset and remove outliers

[dataset, targets] = generate_data(dataset_folder, window_size,'true');

% Remove outliers 
[targets, outliers] = rmoutliers(targets);
dataset = dataset(~outliers);

%% Normalise input data
normalized_dataset = cell(size(dataset));

for i = 1:numel(dataset)
    % Extract the matrix from the cell
    current_matrix = dataset{i};

    % Normalize the matrix
    normalized_matrix = normalize(current_matrix, 'range');
    normalized_dataset{i} = normalized_matrix;
end

%% Generate training and test set
partition_data = cvpartition(size(normalized_dataset, 1), "Holdout", testing_set_ratio);

training_set = dataset(training(partition_data), :);
training_targets = targets(training(partition_data), :);

test_set = dataset(test(partition_data), :);
test_targets = targets(test(partition_data), :);

save('./results/cnn_dataset', ...
    'training_set', ...
    'training_targets', ...    
    'test_set', ...
    'test_targets');

%% Define CNN architecture 

load('./results/cnn_dataset');

% Network structure
layers = [
    sequenceInputLayer(num_channels)

    convolution1dLayer(filter_size(1), num_filters(1), 'Stride', convolutional_stride(1), 'Padding', 'same')
    batchNormalizationLayer
    maxPooling1dLayer(pooling_compression(1), 'Stride', pooling_stride(1), 'Padding', 'same')
    reluLayer

    convolution1dLayer(filter_size(2), num_filters(2), 'Stride', convolutional_stride(2), 'Padding', 'same')
    batchNormalizationLayer
    maxPooling1dLayer(pooling_compression(2), 'Stride', pooling_stride(2), 'Padding', 'same')
    reluLayer

    convolution1dLayer(filter_size(3), num_filters(3), 'Stride', convolutional_stride(3), 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    convolution1dLayer(filter_size(4), num_filters(4), 'Stride', convolutional_stride(4), 'Padding', 'same')
    batchNormalizationLayer
    maxPooling1dLayer(pooling_compression(3), 'Stride', pooling_stride(3), 'Padding', 'same')
    reluLayer

    globalAveragePooling1dLayer

    fullyConnectedLayer(hidden_layer_size)
    dropoutLayer(0.3)
    leakyReluLayer 
    fullyConnectedLayer(output_layer_size)

    regressionLayer
];

% Training option
options = trainingOptions( ...
    'adam', ...
    ...
    MaxEpochs = epochs_number, ...
    MiniBatchSize = mini_batch_size, ...
    Shuffle = 'every-epoch' , ...
    ...
    InitialLearnRate = initial_learn_rate, ...
    LearnRateSchedule = learn_rate_schedule, ...
    LearnRateDropPeriod = learn_rate_drop_period, ...
    LearnRateDropFactor = learn_rate_drop_factor, ...
    L2Regularization = L2_regularization, ...
    ...
    ExecutionEnvironment = 'auto', ...
    Plots = 'training-progress', ...
    Verbose = 1, ...
    VerboseFrequency = 1 ...
);

%% View network and Train
% deepNetworkDesigner(layers);
analyzeNetwork(layers);

net = trainNetwork(training_set, training_targets, layers, options);


%% Test the CNN

y_training = predict(net, training_set);
y_test = predict(net, test_set);

% Plot regression and save result
figure(1); 
plotregression(training_targets, y_training);
saveas(1, './results/cnn_training_regression.png');
figure(2); 
plotregression(test_targets, y_test);
saveas(2, './results/cnn_test_regression.png');
