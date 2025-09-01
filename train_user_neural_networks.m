function train_user_neural_networks()
    % Main script to train neural networks for individual users and a combined dataset

    % Load data for all users
    disp('Loading data for all users...');
    all_users_data = load_all_users_data();

    % Combine data for all users
    disp('Combining data for all users...');
    [all_features, all_labels] = combine_all_users_features(all_users_data);

    % Analyze combined data
    disp('Analyzing combined data...');
    analyze_data(all_features, all_labels);

    % Train individual neural networks for each user
    disp('Training individual neural networks for each user...');
    individual_performance = struct();
    user_ids = fieldnames(all_users_data);
    for i = 1:numel(user_ids)
        user_id = user_ids{i};
        disp(['Processing data for ', user_id, '...']);
        [features, labels] = combine_all_users_features(struct(user_id, all_users_data.(user_id)));
        if isempty(features)
            disp(['No data available for user ', user_id]);
            individual_performance.(user_id) = NaN;
            continue;
        end
        hiddenLayerSize = 10; % Set hidden layer size
        [~, accuracy] = train_ffmlp(features, labels, hiddenLayerSize);
        individual_performance.(user_id) = accuracy;
    end

    % Train a combined neural network for all users
    disp('Training a combined neural network for all users...');
    hiddenLayerSize = [50, 20]; % Larger network combined dataset
    [~, combined_performance] = train_ffmlp(all_features, all_labels, hiddenLayerSize);

    % Results
    disp('Individual user network performances:');
    disp(individual_performance);
    disp(['Combined network performance: ', num2str(combined_performance)]);
end