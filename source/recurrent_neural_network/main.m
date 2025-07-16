clear;
close all;
clc;

%% Constants

% Dataset generation parameters
dataset_folder = '.\..\data';
fraction_test_set = 0.2;
ECG_ID = 1;
k_window_size = 40;

% Network layers parameters
num_channels = 1;
max_num_epochs = 10 ;
mini_batch_size = 2500;
lstm_layer_size = 52;
output_layer_size = 1;
dropout_probability = 0.4;

% Training options parameters
learn_rate_schedule = 'piecewise';
learn_rate_drop_period = 10;
learn_rate_drop_factor = 0.1;

rng("default");

%% Generate the dataset

ecg = generate_single_step_data(dataset_folder);

% Normalize dataset with z-score
ecg = ecg{ECG_ID};
ecg = normalize(ecg,'zscore'); 

% Generate windows with input-target couples
dataset = cell(length(ecg) - k_window_size, 1);
targets = zeros(length(ecg) - k_window_size, 1);

start_idx = 1;
end_idx = k_window_size;

while end_idx < length(ecg)
    dataset{start_idx} = ecg(start_idx : end_idx);
    targets(start_idx) = ecg(end_idx + 1);

    start_idx = start_idx + 1;
    end_idx = end_idx + 1;
end

clear start_idx end_idx;

%% Divide training and test set
partition_data = cvpartition(size(dataset, 1), "Holdout", fraction_test_set);

training_set = dataset(training(partition_data), :);
training_targets = targets(training(partition_data), :);

test_set = dataset(test(partition_data), :);
test_targets = targets(test(partition_data), :);

save('./results/rnn_single_step_final_dataset', ...
    'training_set', ...
    'training_targets', ...    
    'test_set', ...
    'test_targets');

%% Define RNN architecture and train it

load('./results/rnn_single_step_final_dataset');

% Network structure
layers = [
    sequenceInputLayer(num_channels)
    lstmLayer(lstm_layer_size, 'OutputMode', 'last')
    fullyConnectedLayer(num_channels)
    dropoutLayer(dropout_probability)
    fullyConnectedLayer(output_layer_size)
    regressionLayer
];

% Training option
options = trainingOptions('rmsprop', ...
    MaxEpochs = max_num_epochs, ...
    MiniBatchSize = mini_batch_size, ...
    ...
    LearnRateSchedule = learn_rate_schedule, ...
    LearnRateDropFactor = learn_rate_drop_factor, ...
    LearnRateDropPeriod = learn_rate_drop_period, ...
    ...
    SequencePaddingDirection = 'left', ...
    ...
    Plots = 'training-progress', ...
    Verbose = 1, ...
    VerboseFrequency = 5, ...
    ExecutionEnvironment = 'auto' ...             
);

net = trainNetwork(training_set, training_targets, layers, options);

save('./results/rnn_single_step_final_net', 'net');

%% Test the RNN

y_training = predict(net, training_set, MiniBatchSize=mini_batch_size);
y_test = predict(net, test_set, MiniBatchSize=mini_batch_size);

%%
targets = test_targets{1};
x = 1:size(targets, 2);
y = y_test{1};

figure;
plot(x(:, 1:10000), targets(:, 1:10000)');
hold on;
plot(x(:, 1:10000), y(:, 1:10000)');

%% Plot regression and save result

figure(1); 
plotregression(training_targets, y_training);
saveas(1, './results/rnn_single_step_training_regression.png');
figure(2); 
plotregression(test_targets, y_test);
saveas(2, './results/rnn_single_step_test_regression.png');

% Draw forecasting plot
figure;
plot(y_test(1:200), '--', 'DisplayName', 'Predicted');
hold on;
plot(test_targets(1:200), 'DisplayName', 'Actual');
hold off;
legend;