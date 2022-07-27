function p = predictOneVsAll(all_theta,X)
    %predict the label for the trained one-vs-all classifier. The labels
    %are in the range 1 ... K, where K = size(all_theta,1)
    %p is a vector of predictions for each example in the matrix X. Note
    %that X contains the examples in rows. all_theta is a matrix where ith
    %row is s trained logistic regression theta vactor for the i-th class.
    %You should set p to a vector of values from 1...K
    
    m = size(X,1);%No. of input examples to predict
    num_labels = size(all_theta,1);%No. of output classifiers
    
    p = zeros(size(X,1),1);
    
    prob_mat = X*all_theta';
    [prob,p] = max(prob_mat,[],2);
    
    %%%Working of the code :Computation per input image 
    %for i = 1:m
    %   one_image = X(i,:);
    %   prob_mat = one_image*all_theta';
    %   [prob,out] = max(prob_mat);
    %   %out : predicted output
    %   %prob : probability of predicted output
    %   p(i) = out;
    
end