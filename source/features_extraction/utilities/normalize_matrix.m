function [normalized_features] = normalize_matrix(features)
    normalized_features = features - min(features);
    normalized_features = normalized_features ./ max(normalized_features);
end