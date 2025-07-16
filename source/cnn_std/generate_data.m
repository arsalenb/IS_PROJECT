function [data, targets] = generate_data(dataset_path, window_size,remove_noisy_recording)
    
    % Retrieve timeseries and targets csv files:
    csv_timeseries = dir(fullfile(dataset_path, '*timeseries.csv'));
    csv_targets = dir(fullfile(dataset_path, '*targets.csv'));

    if (remove_noisy_recording)
    csv_timeseries = csv_timeseries(~contains({csv_timeseries.name}, '13'));
    csv_targets = csv_targets(~contains({csv_targets.name}, '13'));
    end
    % Initialize variables
    data = {};
    targets = [];

    % Iterate timeseries files
    for k = 1 : length(csv_timeseries)
        
        % Read timeseries files and remove timestamp column
        file_path = fullfile(dataset_path, csv_timeseries(k).name);
        raw_timeseries = readtable(file_path);
        raw_timeseries = raw_timeseries(:, 2:end);

        disp(csv_timeseries(k).name); % print to the console the name of the current file we are analyzing

        % Read targets files and remove timestamp column
        file_path = fullfile(dataset_path, csv_targets(k).name);
        raw_targets = readtable(file_path);
        raw_targets = raw_targets(:, 2);
        
        % Windows Iterations
        current_window_iterator = 1;
        start_window_index = 1;
        end_window_index = window_size;

        while ((start_window_index + window_size - 1) < size(raw_timeseries, 1))
            
            % Compute window std and insert it in data
            current_window_data =  table2array(raw_timeseries(start_window_index : end_window_index, :))';
            current_target_data = std(table2array(raw_targets(start_window_index : end_window_index, 1)));

            data = [ data; { current_window_data } ];
            targets = [ targets; current_target_data ];

            current_window_iterator = current_window_iterator + 1;

            % Compute window row indices
            start_window_index = (current_window_iterator - 1) * window_size + 1;
            end_window_index = start_window_index + window_size - 1;
        end
    end
end