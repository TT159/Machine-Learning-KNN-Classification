% EC 503 Learning from Data
% Fall semester, 2021
% Homework 2
% by (Tian Tan)
%
% Nearest Neighbor Classifier
% each image is 28x28 matrix = 784£¬ 60000 training images,10000 test images
% Problem 2.5e

clc, clear

fprintf("==== Loading data_mnist_train.mat\n");
load("data_mnist_train.mat");
fprintf("==== Loading data_mnist_test.mat\n");
load("data_mnist_test.mat");

% show test image
%imshow(reshape(X_train(200,:), 28,28)') %reshape to 28x28,thus show the image

% determine size of dataset
[Ntrain, dims] = size(X_train); % [# 784]
[Ntest, ~] = size(X_test);

% precompute components

% Note: To improve performance, we split our calculations into
% batches. A batch is defined as a set of operations to be computed
% at once. We split our data into batches to compute so that the 
% computer is not overloaded with a large matrix.
%batch_size = 500;  % fit 4 GB of memory

test_batch_size = 500; %20 batches
test_num_batches = Ntest / test_batch_size;
train_batch_sizee = 1000; % 60 batches
train_num_batches = Ntrain / train_batch_size;

% Using (x - y) * (x - y)' = x * x' + y * y' - 2 x * y'
ypred = zeros(10000,1); % to store the predication of test images
for bn = 1:test_snum_batches
  batch_start = 1 + (bn - 1) * batch_size;
  batch_stop = batch_start + batch_size - 1;
  % calculate cross term 
  % To get the test batch matrix 
  B_test=X_test(batch_start:batch_stop,:); % rows
  
    % compute euclidean distance
  Xdis=zeros(60000,1); % to store the distance
  for i =1: 500
    Xdis=sqrt(B_test(i,:).^2*ones(size(X_train'))+ones(size(B_test(i,:)))*(X_train').^2-2*B_test(i,:)*X_train');  
    % find minimum distance for k = 1
    [mindist,index]=min(Xdis);
    % k=1, 1-nearest neighbor
    ypred(i)=Y_train(index);    % to store the prediacted class of this image
  end
        
    fprintf("==== Doing 1-NN classification for batch %d\n", bn);
end
% compute confusion matrix
disp('Confusion Matrix is following:');
conf_mat = confusionmat(Y_test, ypred);
disp(conf_mat);
% compute CCR from confusion matrix
sum=trace(conf_mat);
disp('The ccr for the test is:');
ccr = sum/10000;
disp(ccr);
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
%   % compute euclidean distance
%   Xdis=zeros(60000,1); % to store the distance
%   for i =1: 500
%     Xdis=sqrt(B_test(i,:).^2*ones(size(X_train'))+ones(size(B_test(i,:)))*(X_train').^2-2*B_test(i,:)*X_train');  
%     % find minimum distance for k = 1
%     [mindist,index]=min(Xdis);
%     % k=1, 1-nearest neighbor
%     ypred(i)=Y_train(index);    % to store the prediacted class of this image
%   end
%         
%     fprintf("==== Doing 1-NN classification for batch %d\n", bn);
% end
% % compute confusion matrix
% disp('Confusion Matrix is following:');
% conf_mat = confusionmat(Y_test, ypred);
% disp(conf_mat);
% % compute CCR from confusion matrix
% sum=trace(conf_mat);
% disp('The ccr for the test is:');
% ccr = sum/10000;
% disp(ccr);








