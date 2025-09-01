function all_users_data = load_all_users_data()
    % Function to load datasets for all users

    % Folder containing the data
    data_folder = './CW-Data/';
    user_ids = {'U01', 'U02', 'U03', 'U04', 'U05', 'U06', 'U07', 'U08', 'U09', 'U10'};
    feature_types = {'FreqD_FDay', 'TimeD_FDay', 'TimeD_FreqD_FDay', ...
                     'FreqD_MDay', 'TimeD_MDay', 'TimeD_FreqD_MDay'};

    all_users_data = struct();

    for i = 1:numel(user_ids)
        user_id = user_ids{i};
        user_data = struct();
        for j = 1:numel(feature_types)
            feature_type = feature_types{j};
            file_name = [user_id, '_Acc_', feature_type, '.mat'];
            file_path = fullfile(data_folder, file_name);
            try
                data = load(file_path);
                disp(['Loaded file: ', file_name]);
                disp('Variables in file:');
                disp(fieldnames(data));
                user_data.(feature_type) = data;
            catch ME
                warning('Failed to load file: %s. Error: %s', file_name, ME.message);
            end
        end
        all_users_data.(user_id) = user_data;
    end
end