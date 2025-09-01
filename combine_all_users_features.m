function [all_features, all_labels] = combine_all_users_features(all_users_data)
    % Combine features and labels from all users
    user_ids = fieldnames(all_users_data);
    feature_types = {'FreqD_FDay', 'TimeD_FDay', 'TimeD_FreqD_FDay'};
    all_features = [];
    all_labels = [];

    % Initialize a variable to store the maximum feature dimension
    max_feature_dim = 0;

    % Determine the maximum feature dimension
    for i = 1:numel(user_ids)
        user_id = user_ids{i};
        user_data = all_users_data.(user_id);
        for j = 1:numel(feature_types)
            feature_type = feature_types{j};
            if isfield(user_data, feature_type)
                feature_data = user_data.(feature_type);
                if isfield(feature_data, 'Acc_FD_Feat_Vec')
                    features = feature_data.Acc_FD_Feat_Vec;
                elseif isfield(feature_data, 'Acc_TD_Feat_Vec')
                    features = feature_data.Acc_TD_Feat_Vec;
                elseif isfield(feature_data, 'Acc_TDFD_Feat_Vec')
                    features = feature_data.Acc_TDFD_Feat_Vec;
                else
                    warning(['Feature type ', feature_type, ' for ', user_id, ' does not contain expected variables.']);
                    continue;
                end
                max_feature_dim = max(max_feature_dim, size(features, 2));
            end
        end
    end

    % Combine features and labels
    for i = 1:numel(user_ids)
        user_id = user_ids{i};
        user_data = all_users_data.(user_id);
        for j = 1:numel(feature_types)
            feature_type = feature_types{j};
            if isfield(user_data, feature_type)
                feature_data = user_data.(feature_type);
                if isfield(feature_data, 'Acc_FD_Feat_Vec')
                    features = feature_data.Acc_FD_Feat_Vec;
                elseif isfield(feature_data, 'Acc_TD_Feat_Vec')
                    features = feature_data.Acc_TD_Feat_Vec;
                elseif isfield(feature_data, 'Acc_TDFD_Feat_Vec')
                    features = feature_data.Acc_TDFD_Feat_Vec;
                else
                    warning(['Feature type ', feature_type, ' for ', user_id, ' does not contain expected variables.']);
                    continue;
                end

                % Adjust feature dimensions
                [num_samples, current_dim] = size(features);
                if current_dim < max_feature_dim
                    features = [features, zeros(num_samples, max_feature_dim - current_dim)];
                elseif current_dim > max_feature_dim
                    % Trim to match the maximum dimension
                    features = features(:, 1:max_feature_dim);
                end

                labels = i * ones(size(features, 1), 1); % User specific labels
                all_features = [all_features; features];
                all_labels = [all_labels; labels];
            end
        end
    end
end