function data = augment_data(input_matrix,augmentation_factor)

% Constants
hidden_layer_size = round(size(input_matrix, 2) / 2);

% Initialize matrices
input_matrix = input_matrix';
data = zeros(size(input_matrix, 1), size(input_matrix, 2) * augmentation_factor);
data(:, 1:size(input_matrix, 2)) = input_matrix;

% Train autoencode
for i=1:augmentation_factor
    autoencoder = trainAutoencoder(input_matrix,...
                               hidden_layer_size,...
                               'EncoderTransferFunction', 'satlin', ...
                               'DecoderTransferFunction', 'purelin', ...
                               'L2WeightRegularization', 0.001, ...
                               'SparsityRegularization', 1, ...
                               'SparsityProportion', 0.1, ...
                               'ShowProgressWindow', false ...
                               );

    output_autoenc = predict(autoencoder, input_matrix);
    fprintf('Iteration %i: %f\n', i, mse(output_autoenc, input_matrix));
    start_index = size(input_matrix, 2) * i + 1;
    end_index = size(input_matrix, 2) * (i + 1);
    data(:, start_index : end_index) = output_autoenc;

end

% Return output
data = data';

end