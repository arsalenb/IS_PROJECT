function [features_matrix, activities_targets] = create_features_matrix(data_path, num_windows, activities)

    % Configurations parameters
    FEATURES_NUMBER = 12;

    % List all timeseries files contained in dataset
    csv_timeseries = dir(fullfile(data_path, '*timeseries.csv'));

    % Initiliaze matrices
    features_matrix = [];
    activities_targets = [];

    % Iterate through timeseries files
    for i = 1 : length(csv_timeseries)

        % retrieve the name of the file csv to analyze
        filename = fullfile(csv_timeseries(i).name);
        disp(filename); % print to the console the name of the current file we are analyzing
    
        % creation of the absolute path
        file_path = fullfile(data_path, filename);
        % Import the signals from the csv
        signals_table = readtable(file_path, 'Range', 'B:L');
        % conversion of the table containing the signals value
        raw_data = table2array(signals_table);
        
        % extract timeseries parameters
        [signal_length, number_of_signals] = size(raw_data);

        % Extract the activity of the file
        for k = 1 : size(activities, 2)
            if contains(filename, activities(k))
                activity = k;
                break;
            end
        end
        
        % Compute window size and step
        num_contiguos_windows = ceil(num_windows / 2);
        num_windows = num_contiguos_windows * 2 - 1; % if overlapped, the number of windows must be odd!!
        window_size = floor(signal_length / num_contiguos_windows);
        window_step = floor(window_size / 2);

        % Preallocate feature matrix
        extracted_features = zeros(FEATURES_NUMBER, number_of_signals * num_windows);

        % Compute features
        for signal_index = 1 : number_of_signals
            for window_index = 1 : num_windows
                start_timeseries_index = ((window_index - 1) * window_step) + 1;
                end_timeseries_index = start_timeseries_index + window_size - 1;
                
                % Extract raw data from window and compute features
                windows_data = raw_data(start_timeseries_index : end_timeseries_index,  signal_index);
                signal_features = [
                min(windows_data);           % MINIMUM
                max(windows_data);           % MAXIMUM
                mean(windows_data);          % MEAN
                median(windows_data);        % MEDIAN
                var(windows_data);           % VARIANCE
                kurtosis(windows_data);      % KURTOSIS
                skewness(windows_data) ;     % SKEWNESS
                iqr(windows_data);           % INTERQUANTILE_RANGE_GSR
                sum(abs(windows_data) .^ 2); % Energy
                meanfreq(windows_data);      % MEAN_FREQ
                medfreq(windows_data);       % MEDIAN_FREQ
                obw(windows_data)            % OCCUPIED_BANDWIDTH
            ];

                extracted_features(:, (signal_index - 1) * num_windows + window_index) = signal_features;
            end
        end

        % Concatenate the new window features row with the features_matrix and 
        % the activity of the window with the activity vector
        features_matrix = [features_matrix; extracted_features(:)'];
        activities_targets = [activities_targets; activity];
    end

end

