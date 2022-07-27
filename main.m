clc;
clear;
%Multi-class classification 
%Here we use logistic regression and neural networks to recognize
%handwritten digits (0 to 9). We start with one-vs-all classification.

%Load saved matrices from file
load('ex3data1.mat')

%There are 5000 training examples above where each training 
%example is a 20x20 pixel grayscale image of the digit. Each digit
%is represented by a floating point number indicating the 
%grayscale intensity at that location.
%The 20x20- grid pixels is 'unrolled' into a 400-dimensional
%vector.
%Each of these training examples becomes a single row in our data 
%matrix x. This gives us a 5000x400 matrix where every row is
%a training examplefor a handwritten digital image.

m = size(X, 1);
% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);

%theta_t = [-2; -1; 1; 2];
%X_t = [ones(5,1) reshape(1:15,5,3)/10];
%y_t = ([1;0;1;0;1] >= 0.5);
%lambda_t = 3;
%[J, grad] = lrCostFunction( X_t, y_t,theta_t, lambda_t);

%fprintf('Cost: %f | Expected cost: 2.534819\n',J);
%fprintf('Gradients:\n'); fprintf('%f\n',grad);
%fprintf('Expected gradients:\n0.146561  -0.548558  0.724722  1.398003\n');

%One-vs-all Classification: here we train the data set using logistic
%regression for num_labels different classes.
num_labels = 10;
lambda = 0.1;
X = [ones(m,1) X];
[all_theta] = oneVsAll(X,y,num_labels,lambda);

%One-vs-All Prediction:
pred = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);







