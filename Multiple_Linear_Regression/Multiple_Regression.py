'''
Explicação Matemática

Formula Geral: Y = β0 + β1 * X1 + β2 * X2 + ... + βn * Xn

onde:

Y:  é a variável dependente.
X1, X2, ..., Xn:são as variáveis independentes.
β1, β2, ..., βn: são os coeficientes que estamos tentando encontrar. Esses coeficientes dizem quanto cada variável X contribui para o valor de Y.

Formula dos coeficientes: β = (X^T * X)^-1 * X^T * Y

X: A matriz de variáveis independentes. Cada linha de X representa uma observação (uma linha do DataFrame) e cada coluna representa uma variável (por exemplo, área, quartos, tipo de imóvel, bairro, etc.).
X^T: A transposta de X. Ao fazer XT, estamos trocando as linhas por colunas. Isso é necessário porque o cálculo exige que multipliquemos XT por X.
(X^T * X)^-1: A inversa da matriz X^T * X. Multiplicar uma matriz pela sua inversa é como "desfazer" o efeito da multiplicação inicial.
os erros já foram explicados nas regressões simples e eles não mudam

Explicação Simplificada

Imagine que você quer prever o preço de um imóvel (variável dependente). Para isso, você tem algumas informações sobre o imóvel, como:
Área do imóvel, Número de quartos, Tipo de imóvel, Bairro.
Essas informações são as variáveis independentes que você vai usar para tentar prever o preço. Cada uma dessas informações tem um "peso" que afeta o preço, e o objetico dos cálculos é descobrir esses pesos (coeficientes). 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    'Age': [25, 34, 29, 42, 53, 31, 45, 38, 27, 39, 28, 56, 30, 41, 33, 46, 50, 60, 36, 52],
    'Experience': [5, 12, 3, 18, 20, 7, 15, 10, 9, 14, 8, 25, 17, 13, 22, 19, 6, 11, 23, 24],
    'Salary': [3000, 5000, 4500, 8000, 12000, 3500, 7000, 6500, 4000, 5500, 4200, 10000, 7500, 6800, 9000, 11000, 9500, 9800, 4800, 8700],
    'Hours_Work': [40, 44, 38, 45, 50, 42, 41, 43, 39, 48, 36, 47, 49, 40, 44, 42, 41, 50, 46, 43],
    'Education': ['Undergraduate', 'Postgraduate', 'High School', 'Undergraduate', 'Postgraduate', 'High School', 'Undergraduate', 'High School', 'High School', 'Undergraduate', 'High School', 'Undergraduate', 'Postgraduate', 'Undergraduate', 'High School', 'Postgraduate', 'Undergraduate', 'High School', 'High School', 'Postgraduate'],
    'Productivity': [72.5, 89.1, 65.4, 95.6, 85.7, 77.8, 82.3, 75.4, 69.3, 84.9, 71.2, 92.3, 79.1, 88.4, 86.5, 91.2, 80.8, 78.3, 83.2, 85.1]
}

df = pd.DataFrame(data)

# Converting 'Education' column into boolean dummy variables
df = pd.get_dummies(df, columns=['Education'], drop_first=True)
df = df.astype(int)

# X contains all columns except 'Productivity'
# Y contains only the 'Productivity' column
X = df.drop('Productivity', axis=1).values
Y = df['Productivity'].values

# Obtaining the data to use in the formula
XT = X.T
XTX = XT @ X
XTY = XT @ Y
XTX_inv = np.linalg.pinv(XTX)

# Applying the data to the formula
β = XTX_inv @ XTY
print("Regression coefficients:", β)

# Asking the user for the input values to predict productivity
age = int(input("Age: "))
experience = int(input("Experience (Years): "))
salary = int(input("Salary: "))
hours_work = int(input("Weekly working hours: "))
education = input("Level of education (High School, Undergraduate, Postgraduate): ")
    
# Creating user array
entry = np.array([age, experience, salary, hours_work, 0, 0])

if education.lower() == 'Undergraduate':
    entry[4] = 1
elif education.lower() == 'Postgraduate':
    entry[5] = 1
    
prod_predict = entry @ β
print(f"\nProductivity Predict: {prod_predict:.2f}")

# Linear regression graph

# Real VS Predict
plt.scatter(Y, X @ β, color='blue', label='Predicts')  

# Adding user predict point in graph 
plt.scatter(prod_predict, prod_predict, color='red', label='User Predict', marker='x', s=100)

plt.xlabel("Real Productivity")
plt.ylabel("Estimated Productivity")
plt.title("Productivity Real vs Estimated")
plt.plot([min(Y), max(Y)], [min(Y), max(Y)], color='black', linestyle='--', label='Linha Ideal') 
plt.legend()
plt.show()

    
    
