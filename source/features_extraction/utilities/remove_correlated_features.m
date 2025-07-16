function [features_without_correlated_columns] = remove_correlated_features(features_matrix,correlation_threshold)

correlation_matrix = corrcoef(features_matrix);
[correlated_columns_indices, ~] = find(tril((abs(correlation_matrix) > correlation_threshold), -1));
correlated_columns_indices = unique(sort(correlated_columns_indices));

features_without_correlated_columns = features_matrix;
features_without_correlated_columns(:, correlated_columns_indices) = [];