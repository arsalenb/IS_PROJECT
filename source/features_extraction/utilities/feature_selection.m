function criterion = feature_selection (x_train, y_train, hidden_layer_size)

    % Create a fitnet network
    net = fitnet(hidden_layer_size);
    net.trainFcn = 'trainbr';
    net.trainParam.showWindow = 0;
    net.performFcn = 'mse';

    % Train network
    net = train(net, x_train', y_train');
    
    % Test network
    test_result = net(x_train');
    criterion = perform(net, y_train', test_result);
end
