# Machine-Learning-KNN-Classification
In this project, I will try to implement the KNN classifier to two classes based on Euclidean distance with MatLab code.

## Dataset


- (a) Create a scatter plot of all the training data (with the first column as the horizontal-axis and the
second column as the vertical-axis. Color the data points in the 1st, 2nd, and 3rd classes with red,
green, and blue colors, respectively.
Helpful MATLAB function: gscatter
- (b) For each point on this 2-D grid [−3.5 : 0.1 : 6] × [−3 : 0.1 : 6.5], calculate its probability of
being class 2 using k = 10 nearest neighbors, i.e., bp10NN(y = 2|x). Plot the probabilities of these
points on a 2-D colored map using the MATLAB default colormap. Repeat this for class 3. Print and
include the two figures in your HW-report. Add a colorbar in the figures to indicate the color range
and label both axes. Describe what you observe and how it relates to what you learned in class.
Helpful MATLAB functions: imagesc, contourf, colormap, colorbar.
- (c) For all the points on the same 2-D grid used in part (b), predict its class label using a kNN
classifier with k = 1. Color-code the decisions for each grid point using the same color coding
scheme used in part (a). Repeat this for k = 5. Print and include the two figures in your HW-report.
Comment on how they differ and how it relates to what you learned in class.
- (d) K Perform LOOCV on the training set and plot the average LOOCV CCR (vertical axis) as a
function of k (horizontal axis) for k = 1, 3, 5, 7, 9, 11. Compute and report the smallest value of k for
which the LOOCV CCR is maximum (among the 6 choices). Print and include the figure of the plot
in your HW-report. Comment on how what you observe relates to what you learned in class.
