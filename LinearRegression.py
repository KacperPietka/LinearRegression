#linear regression y = ax + b
import pandas as pd
import matplotlib.pyplot as plt

file_name = input('Enter your file name with .csv\nfor example: data.csv\nFile: ')

data = pd.read_csv(f'./LinearRegression/{file_name}', sep=',')

def gradient(a, b, data_set, learning_rate):
    n = len(data_set)
    new_a = 0
    new_b = 0

    for i in range(n):
        x = data['YearsExperience'].iloc[i]
        y = data['Salary'].iloc[i]

        new_a += -(2/n) * x * (y - (a * x + b))
        new_b += -(2/n) * (y - (a * x + b))
        
    a = a - new_a * learning_rate   
    b = b - new_b * learning_rate
    return a, b

a = 0
b = 0
learning_rates = [0.0001, 0.0005, 0.001]
iteration_counts = [500, 1000, 2000]


def best_parameters(data, learning_rates, iteration_counts):
    best_params = None
    best_loss = float('inf')
    for i in learning_rates:
        for j in iteration_counts:
            a, b = 0, 0
            for _ in range(j):
                a, b = gradient(a, b, data, i)
            y_pred = a * data['YearsExperience'] + b
            loss = ((data['Salary'] - y_pred) ** 2).mean()
            if loss < best_loss:
                best_loss = loss
                best_params = (i, j)
    return best_params


learning_rate, iterations = best_parameters(data, learning_rates, iteration_counts)

a, b = 0, 0

for i in range(iterations + 1):
    a, b = gradient(a, b, data, learning_rate)

    if i % 100 == 0:
        y_pred = a * data['YearsExperience'] + b
        loss = ((data['Salary'] - y_pred) ** 2).mean()
        print(f"Iteration {i}, Loss: {loss}")

print(a, b)

plt.scatter(data[['YearsExperience']], data[['Salary']], color='red')
sorted_x = sorted(data['YearsExperience'])
plt.plot(data['YearsExperience'], [a * x + b for x in sorted_x], color='blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression Fit')
plt.show()