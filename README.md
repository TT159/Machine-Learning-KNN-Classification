# Machine-Learning-KNN-Classification
In this project, I will try to implement the KNN classifier to two classes based on Euclidean distance with MatLab code.

## Dataset
we consider a simulated data set provided in data knnSimulation.mat. The rows of variable Xtrain are data points and each data point has 2 feature attributes.

## Task
- (a) Create a scatter plot of all the training data with the first column as the horizontal-axis and the second column as the vertical-axis. Color the data points in the 1st, 2nd, and 3rd classes with red, green, and blue colors, respectively.</br> 
Helpful MATLAB function: gscatter.</br>

- (b) For each point on this 2-D grid [−3.5 : 0.1 : 6] × [−3 : 0.1 : 6.5], calculate its probability of being class 2 using k = 10 nearest neighbors, i.e., bp10NN(y = 2|x). Plot the probabilities of these points on a 2-D colored map using the MATLAB default colormap. Repeat this for class 3. Add a colorbar in the figures to indicate the color range and label both axes.</br>
Helpful MATLAB functions: imagesc, contourf, colormap, colorbar.

- (c) For all the points on the same 2-D grid used in part (b), predict its class label using a kNN classifier with k = 1. Color-code the decisions for each grid point using the same color coding scheme used in part (a). Repeat this for k = 5.

- (d) K Perform LOOCV on the training set and plot the average LOOCV CCR (vertical axis) as a unction of k (horizontal axis) for k = 1, 3, 5, 7, 9, 11. Compute and report the smallest value of k for which the LOOCV CCR is maximum (among the 6 choices).</br>

We next look at the classical handwritten digit recognition problem. The dataset is provided in data mnist train.mat and data mnist test.mat. Each data point represents a 28 × 28 grayscale image for hand written digits from 0 to 9. To visualize this data, for example, the 200-th training image.</br> 
Helpful MATLAB functions: imshow(reshape(X train(200,:), 28,28)’).</br>

- (e) K Apply a 1-Nearest Neighbor classifier to this dataset (treat images as vectors of dimension 282 and use Euclidean distance between two vectors to measure the distance between two images).</br>
Hint: The dataset is large. You can improve time efficiency by breaking up computations into batches.
