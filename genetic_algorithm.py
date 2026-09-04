from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

#1. load the Iris dataset
iris = load_iris()

x = iris.data
y = iris.target

print("Dataset shape: ", x.shape)

#2. Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
x_train_ga, x_validation, y_train_ga, y_validation = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)
print("Training shape: ", x_train.shape)
print("Testing shape: ", x_test.shape)


#3. Create Baseline KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

baseline_accuracy = accuracy_score(y_test, y_pred)

print("Baseline accuracy: ", round(baseline_accuracy, 4))


#4. create initial population
population_size = 20
number_of_features = x.shape[1]

population = np.random.randint(
    0,
    2, 
    size=(population_size, number_of_features)
)

print("\nInitial population: \n", population)


#5. create Fitness Func
def fitness_function(chromosome):
    selected_features = chromosome == 1

    if not np.any(selected_features):
        return 0

    x_train_selected = x_train_ga[:, selected_features]
    x_validation_selected = x_validation[:, selected_features]  

    model = KNeighborsClassifier(n_neighbors=5) 
    model.fit(x_train_selected, y_train_ga)

    y_pred = model.predict(x_validation_selected)

    accuracy = accuracy_score(y_validation, y_pred)

    number_of_selected_feature = np.sum(selected_features)
    feature_penalty = 0.01 * (
        number_of_selected_feature / len(chromosome)
    )

    fitness = accuracy - feature_penalty

    return fitness

#output of fitness func
# fitness_value = []

# for chromosome in population:
#     fitness = fitness_function(chromosome)
#     fitness_value.append(fitness)

# print("Fitness value: \n", fitness_value)



#6. Selection
def selection(population, fitness_value):
    selected = []

    for _ in range(len(population)):
        first = np.random.randint(len(population))
        second = np.random.randint(len(population))

        if fitness_value[first] > fitness_value[second]:
            winner = first
        else:
            winner = second

        selected.append(population[winner])

    return np.array(selected)

# #test the selection func
# selected_population = selection(population, fitness_value)

# print("Selected population: \n", selected_population)


#7. Crossover
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))

    child1 = np.concatenate((
        parent1[:crossover_point],
        parent2[crossover_point:]
    ))

    child2 =np.concatenate((
        parent2[:crossover_point],
        parent1[crossover_point:]
    ))

    return child1, child2

# #test the crossover func
# parent1 = selected_population[0]
# parent2 = selected_population[1]

# child1, child2 = crossover(parent1, parent2)

# print("Parent 1: ", parent1)
# print("Parent 2: ", parent2)
# print("Child 1: ", child1)
# print("Child 2: ", child2)


#8. Mutation
def mutation(chromosome, mutation_rate=0.1):
    mutated_chromosome = chromosome.copy()

    for i in range(len(mutated_chromosome)):
        if np.random.random() < mutation_rate:
            mutated_chromosome[i] = 1 - mutated_chromosome[i]

    return mutated_chromosome

# #test the mutation  func
# mutated_child = mutation(child1)

# print("Child before mutation: ", child1)
# print("Child after mutation: ", mutated_child)


#9. create new Population
def create_new_population(selected_population, best_chromosome):
    new_population = []

    new_population.append(best_chromosome.copy())

    for i in range(0, len(selected_population), 2):
        parent1 = selected_population[i]
        parent2 = selected_population[(i + 1) % len(selected_population)]

        child1, child2 = crossover(parent1, parent2)

        child1 = mutation(child1)
        child2 = mutation(child2)

        new_population.append(child1)
        new_population.append(child2)

    return np.array(new_population[:len(selected_population)])

#10. Genetic Algorithm
generations = 30
current_population = population

best_overall_chromosome = None
best_overall_fitness = 0

fitness_history = []

for generation in range(generations):
    fitness_values = []

    for chromosome in current_population:
        fitness_values.append(fitness_function(chromosome))


    best_index = np.argmax(fitness_values)

    best_chromosome = current_population[best_index]
    best_fitness = fitness_values[best_index]

    fitness_history.append(best_fitness)

    if best_fitness > best_overall_fitness:
        best_overall_fitness = best_fitness
        best_overall_chromosome = best_chromosome.copy()

    selected_population = selection(
        current_population, 
        fitness_values
    )

    current_population = create_new_population(selected_population, best_chromosome)

    print(
        "Generation: ", generation + 1,
        "best fitness: ", round(best_fitness,4)
    )

# 11. Best Solution
print("\nBest Overall Solution")
print("---------------------")

print(
    "Best chromosome:",
    best_overall_chromosome
)

print(
    "Best fitness:",
    round(best_overall_fitness, 4)
)



# 12. Final Evaluation on Test Set
selected_features = best_overall_chromosome == 1

x_train_final = x_train[:, selected_features]
x_test_final = x_test[:, selected_features]

final_model = KNeighborsClassifier(n_neighbors=5)

final_model.fit(
    x_train_final,
    y_train
)

y_final_pred = final_model.predict(
    x_test_final
)

final_accuracy = accuracy_score(
    y_test,
    y_final_pred
)

selected_feature_names = [
    iris.feature_names[i]
    for i in range(len(selected_features))
    if selected_features[i]
]

print("\nFinal Evaluation")
print("----------------")

print(
    "Selected features:",
    selected_feature_names
)

print(
    "Number of selected features:",
    int(np.sum(selected_features))
)

print(
    "Final test accuracy:",
    round(final_accuracy, 4)
)

# 13. Comparison
feature_reduction = (
    1 - np.sum(selected_features) / len(selected_features)
) * 100


print("\nComparison")
print("----------")

print(
    "Baseline accuracy:",
    round(baseline_accuracy * 100, 2),
    "%"
)

print(
    "GA selected features accuracy:",
    round(final_accuracy * 100, 2),
    "%"
)

print(
    "Feature reduction:",
    round(feature_reduction, 2),
    "%"
)


# 14. Plot GA Convergence Curve
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, generations + 1),
    fitness_history,
    marker="o"
)

plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Genetic Algorithm Convergence Curve")

plt.grid(True)

plt.tight_layout()

plt.show()
