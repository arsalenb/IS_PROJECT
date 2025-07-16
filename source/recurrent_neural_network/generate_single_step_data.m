function ecg = generate_single_step_data(resources_path)

csv_file = dir(fullfile(resources_path,'*targets.csv'));

% Initialize cell array to store ECG signals
ecg = cell(length(csv_file), 1);

% Scan every csv file to extract the ECG signals
for m = 1:length(csv_file)
    % Retrieve the name of the csv file to analyze
    filename = fullfile(csv_file(m).folder, csv_file(m).name);
    disp(filename);
    
    % Read ECG signals directly into a cell array
    ecg{m} = readtable(filename, 'Range', 'B:B').Variables';
end

