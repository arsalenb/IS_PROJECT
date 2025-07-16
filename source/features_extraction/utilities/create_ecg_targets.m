function [ecg_mean_targets_vector, ecg_std_targets_vector] = create_ecg_targets(data_path)

    % List all target files contained in dataset
    csv_targets = dir(fullfile(data_path, '*targets.csv'));
    
    % Initialize targets vectors
    ecg_mean_targets_vector = [];
    ecg_std_targets_vector = [];
    
    % Iterate all targets file
    for i = 1 : length(csv_targets)

        % retrieve the name of the file csv to analyze
        filename = fullfile(csv_targets(i).name);
        disp(filename); % print current file name
    
        % creation of the absolute path
        file_path = fullfile(data_path, filename);
        % Import the targets from the csv and remove timestamp column
        signals_table = readtable(file_path, 'Range', 'B:B');
        % conversion of the table containing the signals value
        raw_data = table2array(signals_table);

            
        % Compute mean and standard deviation of the window
        ecg_mean = mean(raw_data);
        ecg_std = std(raw_data);

        ecg_mean_targets_vector = [ecg_mean_targets_vector; ecg_mean]; 
        ecg_std_targets_vector = [ecg_std_targets_vector; ecg_std];

    end
end