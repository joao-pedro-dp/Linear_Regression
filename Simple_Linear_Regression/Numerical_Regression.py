'''
Formula da Regressão Linear Numérica:

Y = A * X + B

Onde:
Y: é o preço da casa (variável dependente).
A: coeficiente angular (ou inclinação) da reta.
X: variável independente, que é o valor de entrada.
B: coeficiente linear (ou intercepto).

A = (M * Σ(X*Y) - ΣX * ΣY)/(M * Σ (X^2) - (ΣX)^2)

Onde:
M: é o numero de dados(linhas).
Σ: somatorio.

B = μY - A * μX

Onde:
μY: média do preço da casa.
μX: média do valor de entrada.

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

houses = [{"area": 50, "price": 240000},{"area": 70, "price": 320000},{"area": 90, "price": 400000},
        {"area": 110, "price": 485000},{"area": 130, "price": 550000},{"area": 150, "price": 625000},
        {"area": 170, "price": 700000},{"area": 190, "price": 775000},{"area": 210, "price": 850000},
        {"area": 230, "price": 925000},{"area": 250, "price": 1000000},{"area": 270, "price": 1075000},
        {"area": 290, "price": 1150000},{"area": 310, "price": 1225000},{"area": 330, "price": 1300000},
        {"area": 350, "price": 1375000},{"area": 370, "price": 1450000},{"area": 390, "price": 1525000},
        {"area": 410, "price": 1600000},{"area": 450, "price": 1800000},{"area": 480, "price": 2000000}
        ]

df = pd.DataFrame(houses)

# Obtaining the values to use in the formula:
m = len(df)
sum_x = sum(df['area'])
sum_y = sum(df['price'])
sum_xy = sum(df['area'] * df['price'])
sum_squared_x = sum(df['area'] ** 2)  
squared_x = sum_x ** 2
mean_x = np.mean(df['area'])
mean_y = np.mean(df['price'])

A = (m * sum_xy - sum_x * sum_y) / (m * sum_squared_x - squared_x)

B = mean_y - A * mean_x

# Calculating errors
df['predict_price'] = A * df['area'] + B
MAE = np.mean(np.abs(df['price'] - df['predict_price']))
MSE = np.mean((df['price'] - df['predict_price']) ** 2)
RMSE = np.sqrt(MSE)
SSE = sum((df['price'] - df['predict_price']) ** 2)
SST = sum((df['price'] - mean_y) ** 2)
R2 = 1 - (SSE / SST)

# Input receives the user's area (X)
X = float(input("Enter the area of the house for which you want to know the price: "))

# Calculating the approximate price of the house with user input
Y = A * X + B  

print(f"RMSE: {RMSE}")
print(f"R^2: {R2}")
print(f"The approximate price of the house with a rating of {X} is: R${Y}")

# Linear regression graph with errors

# Adding the data points 
plt.figure(figsize=(10,5))
plt.scatter(df['area'], df['price'], label="Real Data", color="blue")  
plt.plot(df['area'], df['predict_price'], label="Numerical Linear Regression", color="red")  

# Adicionando linha de erro
for i in range(len(df)):
    plt.vlines(x=df['area'][i], ymin=df['predict_price'][i], ymax=df['price'][i], color='gray', linestyle='dotted')

# Adicionando ponto de previsão do usuário
plt.scatter(X, Y, color='green', s=200, label=f"Result {X}m²", edgecolors='black', zorder=5)  

plt.title("Simple Linear Regression on House Prices using Numerical Variables")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.show()

