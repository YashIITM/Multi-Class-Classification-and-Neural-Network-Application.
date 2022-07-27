function [all_theta] = oneVsAll(X,y,num_labels,lambda)
    %trains multiple logisitic regression classifiers and returns
    %all the classifiers in a matrix all_theta, where tht i-throw
    %of all_theta corresponds to the clasasifier for label i
    
    %num_labels = No. of output classifier (Here it's a 10)
    
    m = size(X,1);%No. of training samples == No. of Images == 5000
    n = size(X,2) - 1;%No. of features == No. of pixels in each Image == 400
    
    all_theta = zeros(num_labels,n+1);
    
    %DIMENSIONS: num_labels x (input_layer_size +1) == num_labels x
    %(no_of_features + 1)==10 x 401
    
    %X = m x input_layer_size + 1
    %Here, 1 row in X represents1 training image of pixel 20*20
    
    
    initial_theta = zeros(n+1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 50);
  
    for c=1:num_labels
       all_theta(c,:) = fmincg (@(t)(lrCostFunction(X, (y == c),t, lambda)),initial_theta, options);
    end
end 