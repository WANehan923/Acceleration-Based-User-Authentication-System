function analyze_data(features, labels)
    disp('Feature Mean:');
    disp(mean(features));
    disp('Feature Standard Deviation:');
    disp(std(features));
    disp('Feature Variance:');
    disp(var(features));
    disp('Label Distribution:');
    disp(histcounts(labels, unique(labels)));
end