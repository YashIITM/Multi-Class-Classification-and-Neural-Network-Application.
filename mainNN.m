clc;
clear;
%NEURAL NETWORK REPRESENTATION
load('ex3data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
random_indices = randperm(m);
sel = X(random_indices(1:100),:);
displayData(sel);
x = [ones(m,1) X];

% Load saved matrices from file
load('ex3weights.mat'); 
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
pred = predict(Theta1, Theta2, x);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


%  Randomly permute examples
rp = randi(m);
% Predict
pred = predict(Theta1, Theta2, x(rp,:));
fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
% Display 
displayData(X(rp, :));