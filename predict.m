function p = predict(Theta1,Theta2,X)
    %Predict label of a trained NN ; outputs the predicted label of X given
    %the weights of NN : Theta1 and Theta2
    
    m = size(X,1);
    num_labels = size(Theta2,1);
    
    %DIMENSIONS 
    %   theta1 = 25 x 401
    %   theta2 = 10 x 26
    
    %layer1 (input) = 400 nodes & 1 bias
    %layer2 (hidden) = 25 nodes & 1 bias
    %layer3 (output) = 10 nodes
    
    %theta_dimensions = s_(j+1) x ((s_j)+1)
    %theta1 = 25 x 401
    %theta2 = 10 x 26
    % theta1:
    %     1st row indicates: theta corresponding to all nodes from layer1 connecting to for 1st node of layer2
    %     2nd row indicates: theta corresponding to all nodes from layer1 connecting to for 2nd node of layer2
    %     and
    %     1st Column indicates: theta corresponding to node1 from layer1 to all nodes in layer2
    %     2nd Column indicates: theta corresponding to node2 from layer1 to all nodes in layer2
    %     
    % theta2:
    %     1st row indicates: theta corresponding to all nodes from layer2 connecting to for 1st node of layer3
    %     2nd row indicates: theta corresponding to all nodes from layer2 connecting to for 2nd node of layer3
    %     and
    %     1st Column indicates: theta corresponding to node1 from layer2 to all nodes in layer3
    %     2nd Column indicates: theta corresponding to node2 from layer2 to all nodes in layer3
    a1 = X;
    z2 = X*Theta1';%5000 x 25
    a2 = sigmoid(z2);%5000 x 25
    
    a2 =  [ones(size(a2,1),1) a2];  % 5000 x 26
  
    z3 = a2 * Theta2';  % 5000 x 10
    a3 = sigmoid(z3);  % 5000 x 10
  
    [prob, p] = max(a3,[],2); 
    %returns maximum element in each row  == max. probability and its index for each input image
    %p: predicted output (index)
    %prob: probability of predicted output
  
end