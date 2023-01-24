#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>

int main() {
    // Load the dataset.
    arma::mat matrix_data;
    matrix_data.load("dataset.csv", arma::csv_ascii);

    // Split the dataset into training and test sets.
    arma::mat trainData = matrix_data.submat(0, 0, matrix_data.n_rows * 0.8 - 1, matrix_data.n_cols - 1);
    arma::mat testData = matrix_data.submat(matrix_data.n_rows * 0.8, 0, matrix_data.n_rows - 1, matrix_data.n_cols - 1);

    // Get the labels for each dataset and convert them to unsigned integers.
    arma::subview<double> trainLabels = trainData.submat(trainData.n_rows - 1, 0, trainData.n_rows - 1, trainData.n_cols - 1);
    arma::subview<double> testLabels = testData.submat(testData.n_rows - 1, 0, testData.n_rows - 1, testData.n_cols - 1);

    // Remove the label column from the training and test data.
    trainData.shed_row(trainData.n_rows - 1);
    testData.shed_row(testData.n_rows - 1);

    // Create the decision tree object.
    mlpack::tree::DecisionTree<arma::mat> dt(trainData, trainLabels);


    // Make predictions on the test data.
    arma::Row<size_t> predictions;
    dt.Classify(testData, predictions);

    // Calculate the accuracy of the model.
    size_t correct = 0;
    for (size_t i = 0; i < predictions.n_elem; i++) {
        if (predictions[i] == testLabels[i]) {
            correct++;
        }
    }
    double accuracy = (double) correct / predictions.n_elem;
    std::cout << "Accuracy: " << accuracy << std::endl;
}
