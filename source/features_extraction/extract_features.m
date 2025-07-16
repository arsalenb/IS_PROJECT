clear;
close all;
clc;

%% Initialize Constants

dataset_folder = '.\..\data';

% Feature's matrix dimension
num_signals = 11;
num_features = 12;
activities = ["walk", "sit", "run"];

% Sequential feature selection
features_to_select = 10;
sequentialfs_hidden_layer_size = 8;

% Data clean & augmentation
correlation_threshold = 0.9;
windows_number = 7;
augmentation_factor = 40;
num_selected_features = 10;
num_k_folds = 10;

%% Generate Features Matrix containing all features of all signals

[features_matrix, activities_targets_vector] = create_features_matrix(dataset_folder,windows_number, activities);

save('./results/non_normalised_features_matrix', 'features_matrix');
save('./results/non_augmented_activities_targets_vector', 'activities_targets_vector');

%% Normalise and Remove Correlated Features

load('./results/non_normalised_features_matrix');

% Normalise features matrix
normalized_features_matrix = normalize_matrix(features_matrix);

% Remove correlated features
uncorrelated_features_matrix=remove_correlated_features(features_matrix,correlation_threshold);

save('./results/uncorrelated_features_matrix', 'uncorrelated_features_matrix');

%% Get ECG Mean and STD Vectors (Targets)

[ecg_mean_targets, ecg_std_targets] = create_ecg_targets(dataset_folder);

save('./results/ecg_targets_vectors', 'ecg_mean_targets', 'ecg_std_targets');

%% Feature Data Augmentation

load('./results/uncorrelated_features_matrix');
load('./results/ecg_targets_vectors');
load('./results/non_augmented_activities_targets_vector');

% Launch data_augmentation
augmented_features_matrix = augment_data(uncorrelated_features_matrix,augmentation_factor);

% Normalise augmented features matrix
normalized_augmented_features_matrix = normalize_matrix(augmented_features_matrix);

% Replicate targets matrices
augmented_ecg_mean_targets_vector = repmat(ecg_mean_targets, augmentation_factor+1, 1);
augmented_ecg_std_targets_vector = repmat(ecg_std_targets, augmentation_factor+1, 1);
augmented_activities_targets_vector = repmat(activities_targets_vector, augmentation_factor+1, 1);

save('./results/augmented_data', ...
    'normalized_augmented_features_matrix', ...
    'augmented_ecg_mean_targets_vector', ...
    'augmented_ecg_std_targets_vector', ...
    'augmented_activities_targets_vector');

%% Extraction of 10 Best Features

load('./results/augmented_data');

opts = statset('Display', 'iter', 'UseParallel', true);

% Create a random partition for stratified 10-fold cross-validation.
mean_cv = cvpartition(augmented_ecg_mean_targets_vector,"KFold",10);

% Select the most relevant features for the mean ecg targets
fs_mean = sequentialfs( ...
    @(input_train, target_train, input_test, target_test,hidden_layer_size)feature_selection(input_train, target_train, input_test, target_test, sequentialfs_hidden_layer_size) , ...
    normalized_augmented_features_matrix, ...
    augmented_ecg_mean_targets_vector, ...
    'opt', opts, ...
    'CV',mean_cv,...
    'nfeatures', num_selected_features);

% Select the most relevant features for the standard deviation ecg targets
std_cv = cvpartition(augmented_ecg_std_targets_vector,"KFold",num_k_folds);

fs_std = sequentialfs( ...
    @(input_train, target_train, input_test, target_test,hidden_layer_size)feature_selection(input_train, target_train, input_test, target_test, sequentialfs_hidden_layer_size) , ...
    normalized_augmented_features_matrix, ...
    augmented_ecg_std_targets_vector, ...
    'opt', opts, ...
    'CV',std_cv,...
    'nfeatures', num_selected_features);

%% Prepare the data matrices for mean and ecg mlps
fs_mean=false(1,277);
fs_mean([132, 153, 157, 163, 166, 169, 180, 183, 189, 196])=true;
fs_std=false(1,277);
fs_std([39,157, 160, 166, 172, 180, 183, 186, 189, 196])=true;
final_features_ecg_mean_matrix = normalized_augmented_features_matrix(:, fs_mean);
final_features_ecg_std_matrix = normalized_augmented_features_matrix(:, fs_std);

% Prepare data matrix for mlp activity classifier
combined_features = fs_mean|  fs_std;
sum(combined_features(:) == 1)
final_features_activities_matrix = normalized_augmented_features_matrix(:, combined_features);

% Export target matrices
final_ecg_mean_targets_vector = augmented_ecg_mean_targets_vector;
final_ecg_std_targets_vector = augmented_ecg_std_targets_vector;
final_activities_targets_vector = augmented_activities_targets_vector;

save('./results/final_data', ...
    'final_features_ecg_mean_matrix', ...
    'final_features_ecg_std_matrix', ...
    'final_features_activities_matrix', ...
    'final_ecg_mean_targets_vector', ...
    'final_ecg_std_targets_vector', ...
    'final_activities_targets_vector');

