'''
Formula da Regressão Linear Categórica:

Basicamente,a formula é idêntica dicotômica; no entanto, Deve-se criar colunas correspondentes à quantidade de variáveis dummy a serem utilizadas, transformando os valores booleanos em 1 e 0 para usá-los na fórmula.

O valor final obtido pode parecer um pouco estranho, mas o objetivo principal, que eram os cálculos, foi alcançado. 
Para resultados mais precisos, basta manter os cálculos e utilizar uma base de dados real, renomeando as colunas conforme necessário para corresponder às usadas no código e realizar os testes. 
No entanto, como mencionado, meu objetivo foi atingido, então vou encerrar por aqui.

Y = β0 + β1 * D1 + β2 * D2

Onde:
Y: é o preço da casa (variável dependente).
D1: 1 se a observação é de nível "médio" e D1 = 0 caso contrário.
D2: 1 se a observação é de nível "alto" e D2 = 0 caso contrário.
A categoria "baixo" será o grupo de referência (ou seja, quando D1 e D2 = 0).
𝛽0: é o intercepto (o preço médio quando D = 0, ou seja, "Ruim").
𝛽1: é o coeficiente da variável dummy (quanto o preço muda quando D1 = 1, ou seja, "Médio").
𝛽2: é o coeficiente da variável dummy (quanto o preço muda quando D2 = 1, ou seja, "Alto").

β1 = (∑(Di1 - μD1) * (Yi - μY)) / (∑(Di1 - D1)^2)

β2 = (∑(Di2 - μD2) * (Yi - μY)) / (∑(Di2 - D2)^2)

Onde:
Di: são os valores da variável dummy.
Yi: são os valores da variável dependente (preço).
μD: é a média de D.
μY: é a média de Y.

β0 = μY - β1 * μD1 - β2 * μD2

Onde:
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

houses = [{"class": "Low", "price": 240000},{"class": "Low", "price": 320000},{"class": "Low", "price": 400000},
        {"class": "Low", "price": 485000},{"class": "Middle", "price": 550000},{"class": "Middle", "price": 625000},
        {"class": "Middle", "price": 700000},{"class": "Middle", "price": 775000},{"class": "Middle", "price": 850000},
        {"class": "Middle", "price": 925000},{"class": "Middle", "price": 1000000},{"class": "High", "price": 1075000},
        {"class": "High", "price": 1150000},{"class": "High", "price": 1225000},{"class": "High", "price": 1300000},
        {"class": "High", "price": 1375000},{"class": "High", "price": 1450000},{"class": "High", "price": 1525000},
        {"class": "High", "price": 1600000},{"class": "High", "price": 1800000},{"class": "High", "price": 2000000}]

df = pd.DataFrame(houses)

# Converting the 'class' column into new boolean dummy variable columns.
df.loc[df['class'] != "Middle", 'D1'] = 0
df.loc[df['class'] == "Middle", 'D1'] = 1
df.loc[df['class'] != "High", 'D2'] = 0
df.loc[df['class'] == "High", 'D2'] = 1

# Obtaining the D1 values to use in the β1 formula
mean_D1 = np.mean(df['D1'])
Di_meanD1= df['D1'] - mean_D1
mean_Y1 = np.mean(df['price'])
Yi1 = df['price']
mean_Y_Yi1 = Yi1 - mean_Y1
sum_Di1_meanD1_mean_Y_Yi = np.sum(Di_meanD1 * mean_Y_Yi1)
squared_sum_Di1_meanD1 = np.sum(Di_meanD1**2)

β1 = (sum_Di1_meanD1_mean_Y_Yi / squared_sum_Di1_meanD1)

# Obtaining the D2 values to use in the β2 formula
mean_D2 = np.mean(df['D2'])
Di_meanD2= df['D2'] - mean_D2
mean_Y2 = np.mean(df['price'])
Yi2 = df['price']
mean_Y_Yi2 = Yi2 - mean_Y2
sum_Di2_meanD2_mean_Y_Yi = np.sum(Di_meanD2 * mean_Y_Yi2)
squared_sum_Di2_meanD2 = np.sum(Di_meanD2**2)

β2 = (sum_Di2_meanD2_mean_Y_Yi / squared_sum_Di2_meanD2)

β0 = (mean_Y1 - β1 * mean_D1 - β2 * mean_D2)

# Calculating errors
df['predict_price'] = β0 + β1 * df['D1'] + β2 * df['D2']
MAE = np.mean(np.abs(df['price'] - df['predict_price']))
MSE = np.mean((df['price'] - df['predict_price']) ** 2)
RMSE = np.sqrt(MSE)
SSE = sum((df['price'] - df['predict_price']) ** 2)
SST = sum((df['price'] - mean_Y1) ** 2)
R2 = 1 - (SSE / SST)

# The input receives a rating (D) using a basic if statement, because the objective is linear regression.
D = int(input("Enter the class of the house for which you want to know the price (0 - Low, 1 - Middle, 2 - High): "))

D1 = 1 if D == 1 else 0
D2 = 1 if D == 2 else 0

# Calculating the approximate price of the house
Y = β0 + β1 * D1 + β2 * D2
print(f"RMSE: {RMSE}")
print(f"R^2: {R2}")
print(f"The approximate price of the house with a class {D} is: R${Y}")

# Linear regression graph with errors

# Adding the data points 
plt.figure(figsize=(10,5))
plt.scatter(df['D1'] + 2 * df['D2'], df['price'], label="Real data", color="blue")  
plt.plot(df['D1'] + 2 * df['D2'], df['predict_price'], label="Categorical Linear Regression", color="red")  

# Adding line with errors
for i in range(len(df)):
    plt.vlines(x=df['D1'][i] + 2 * df['D2'][i], ymin=df['predict_price'][i], ymax=df['price'][i], color='gray', linestyle='dotted')

# Adding the user's predicted point to the plot
plt.scatter(D, Y, color='green', s=200, label=f"Result {D}", edgecolors='black', zorder=5)  

plt.title("Simple Linear Regression on House Prices using Categorical Variables")
plt.xlabel("Class")
plt.ylabel("Price")
plt.legend()
plt.show()

