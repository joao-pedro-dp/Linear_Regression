'''
Formula da Regressão Linear Dicotômica:

Basicamente,a formula é quase idêntica à numérica; no entanto, deve-se criar uma coluna transformando os valores booleanos em 1 e 0 para usá-los na fórmula.

Y = β0 + β1 * D 

Onde:
Y: é o preço da casa (variável dependente).
D: é a variável dicotômica (0 ou 1).
𝛽0: é o intercepto (o preço médio quando D = 0, ou seja, "Ruim").
𝛽1: é o coeficiente da variável dummy (quanto o preço muda quando D = 1, ou seja, "Bom").

β1 = (∑(Di - μD) * (Yi - μY)) / (∑(Di - D)^2)

Onde:
Di: são os valores da variável dummy.
Yi: são os valores da variável dependente (preço).
μD: é a média de D.
μY: é a média de Y.

β0 = μY - β1 * μD

Onde:
B1: é a Formula de cima
μD: é a média de D.
μY: é a média de Y.

MAE = (1/M) * Σ |Yi - Ŷi|
MSE = (1/M) * Σ (Yi - Ŷi)^2
RMSE = sqrt(MSE)
RMSE = sqrt( (1/M) * Σ (Yi - Ŷi)^2 )
R² = 1 - (Σ (Yi - Ŷi)^2 / Σ (Yi - μY)^2)

Onde:

M: é o número de observações,
Yi: são os valores reais,
Ŷi: são os valores previstos pelo modelo.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

houses = [{"rating": "Bad", "price": 240000},{"rating": "Good", "price": 320000},{"rating": "Bad", "price": 400000},
        {"rating": "Good", "price": 485000},{"rating": "Bad", "price": 550000},{"rating": "Good", "price": 625000},
        {"rating": "Bad", "price": 700000},{"rating": "Good", "price": 775000},{"rating": "Good", "price": 850000},
        {"rating": "Good", "price": 925000},{"rating": "Good", "price": 1000000},{"rating": "Bad", "price": 1075000},
        {"rating": "Good", "price": 1150000},{"rating": "Bad", "price": 1225000},{"rating": "Good", "price": 1300000},
        {"rating": "Bad", "price": 1375000},{"rating": "Good", "price": 1450000},{"rating": "Bad", "price": 1525000},
        {"rating": "Good", "price": 1600000},{"rating": "Good", "price": 1800000},{"rating": "Good", "price": 2000000}
        ]

df = pd.DataFrame(houses)

# Converting the 'rating' column into new boolean dummy variable column.
df.loc[df['rating'] == "Bad", 'D'] = 0
df.loc[df['rating'] == "Good", 'D'] = 1

# Obtaining the values to use in the β1 and β0 formula:
mean_D = np.mean(df['D'])
Di_meanD= df['D'] - mean_D
mean_Y = np.mean(df['price'])
Yi = df['price']
mean_Y_Yi = Yi - mean_Y
sum_Di_meanD_mean_Y_Yi = np.sum(Di_meanD * mean_Y_Yi)
squared_sum_Di_meanD = np.sum(Di_meanD**2)

β1 = (sum_Di_meanD_mean_Y_Yi / squared_sum_Di_meanD)

β0 = (mean_Y - β1 * mean_D)

# Calculating errors
df['predict_price'] = β0 + β1 * df['D']
MAE = np.mean(np.abs(df['price'] - df['predict_price']))
MSE = np.mean((df['price'] - df['predict_price']) ** 2)
RMSE = np.sqrt(MSE)
SSE = sum((df['price'] - df['predict_price']) ** 2)
SST = sum((df['price'] - mean_Y) ** 2)
R2 = 1 - (SSE / SST)

# Input receives the user's rating (D)
D = int(input("Enter the rating of the house for which you want to know the price (0: 'Bad' and 1: 'Good'): "))

# Calculating the approximate price of the house with user input
Y = β0 + β1 * D

print(f"RMSE: {RMSE}")
print(f"R^2: {R2}")
print(f"The approximate price of the house with a rating of {D} is: R${Y}")

# Linear regression graph with errors

# Adding the data points 
plt.figure(figsize=(10,5))
plt.scatter(df['D'], df['price'], label="Real data", color="blue")  
plt.plot(df['D'], df['predict_price'], label="Dichotomous Linear Regression", color="red")  

# Adding line with errors
for i in range(len(df)):
    plt.vlines(x=df['D'][i], ymin=df['predict_price'][i], ymax=df['price'][i], color='gray', linestyle='dotted')

# Adding the user's predicted point to the plot
plt.scatter(D, Y, color='green', s=200, label=f"Result {D}", edgecolors='black', zorder=5)  

plt.title("Simple Linear Regression on House Prices using Dichotomous Variables")
plt.xlabel("Rating")
plt.ylabel("Price")
plt.legend()
plt.show()
