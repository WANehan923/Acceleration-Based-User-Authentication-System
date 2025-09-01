function [net, performance] = train_ffmlp(features, labels, hiddenLayerSize)
    % Fix random seed for reproducibility
    rng(42); % Set seed to a fixed value
    
    % Normalize features
    features = normalize(features, 'range');
    
    % Convert labels to categorical
    if ~iscategorical(labels)
        labels = categorical(labels); 
    end

    % Stratified splitting
    c = cvpartition(labels, 'HoldOut', 0.2);
    train_idx = training(c);
    test_idx = test(c);

    % Prepare training and testing data
    X_train = features(train_idx, :)';
    y_train = dummyvar(labels(train_idx))'; % Convert categorical labels to binary matrix
    X_test = features(test_idx, :)';
    y_test = labels(test_idx);

    % Configure neural network
    net = patternnet(hiddenLayerSize, 'trainscg'); % Use scaled conjugate gradient
    net.performParam.regularization = 0.2; % Add regularization
    net.layers{1}.transferFcn = 'tansig'; % Replace ReLU with tansig
    net.layers{end}.transferFcn = 'softmax'; % Output layer activation

    % Set random seed for reproducibility in training
    net.trainParam.showWindow = false; % Turn off training window
    net.trainParam.showCommandLine = true;

    % Divide data for training, validation, and testing
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0.1;

    % Training parameters
    net.trainParam.epochs = 1000;
    net.trainParam.max_fail = 30; % Early stopping

    % Train the network
    [net, tr] = train(net, X_train, y_train);

    % Test the network
    y_pred = net(X_test);
    [~, pred_labels] = max(y_pred, [], 1); % Convert to class labels

    % Calculate accuracy
    accuracy = sum(categorical(pred_labels)' == y_test) / length(y_test);
    performance = accuracy;

    % Display accuracy
    disp(['Accuracy: ', num2str(accuracy * 100), '%']);

    % Performance Plot
    figure;
    plotperform(tr);
    title('Training Performance');

    % Confusion Matrix
    figure;
    confusionchart(y_test, categorical(pred_labels));
    title('Confusion Matrix');
end
