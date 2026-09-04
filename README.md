# Genetic Algorithm for Feature Selection on Iris Dataset

## Project Overview
In this project, I implemented a Genetic Algorithm (GA) for feature selection on the Iris dataset.
The main goal of this project is to understand the process of the Genetic Algorithm and use it to find a suitable subset of features for classification.

## Dataset
I used the Iris dataset, which contains:
- 150 samples
- 4 numerical features
- 3 classes

The four features are:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## Algorithm
I used a Genetic Algorithm with the following steps:

1. Create an initial population of chromosomes.
2. Calculate the fitness of each chromosome.
3. Select better chromosomes using tournament selection.
4. Apply one-point crossover.
5. Apply mutation.
6. Create a new population.
7. Keep the best chromosome using elitism.
8. Repeat the process for 30 generations.

Each chromosome is a binary array of length 4. A value of `1` means that the corresponding feature is selected, while `0` means that the feature is not selected.

## Classification Model
I used K-Nearest Neighbors (KNN) with 5 neighbors.
The fitness function is based on classification accuracy with a small penalty for selecting more features.

## Parameters
| Parameter | Value |
|---|---:|
| Population Size | 20 |
| Number of Generations | 30 |
| Mutation Rate | 0.1 |
| KNN Neighbors | 5 |
| Number of Features | 4 |

## Results
The best chromosome obtained in my run was:
`[1 1 1 0]`

This means that three features were selected:
- Sepal Length
- Sepal Width
- Petal Length

The Petal Width feature was not selected.
The best fitness value was `0.9925`.
The final test accuracy using the selected features was `93.33%`.
For comparison, the baseline KNN model using all four features achieved `100%` accuracy.
The Genetic Algorithm reduced the number of features by `25%`.

## Conclusion
The Genetic Algorithm was able to remove one of the four features while maintaining a relatively good classification accuracy.
Although the final accuracy was lower than the baseline model, the main purpose of this project was to understand and implement the feature selection process using a Genetic Algorithm.

## Technologies
- Python
- NumPy
- Scikit-learn
- Matplotlib

## Report
The complete project report is included in this repository.
