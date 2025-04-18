import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

df = pd.read_csv('Power_Plant.csv')
df = df.rename(columns={'AT': 'Temperature', 'V': 'Air Pressure','RH': 'Air Humidity','AP': 'ATM Pressure','PE': 'Energy Produced',})

y = df['Energy Produced']
X = df.drop(columns='Energy Produced')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=250)

df_train = pd.DataFrame(data= X_train)
df_train['Energy Produced'] = y_train

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.OLS(y_train,
                  X_train[['const', 'Air Pressure', 'ATM Pressure', 'Air Humidity']]).fit()

predict = model.predict(X_test[['const', 'Air Pressure', 'ATM Pressure', 'Air Humidity']])

new_data = pd.DataFrame({ 'const': [1],
                              'Air Pressure': [32.1],
                              'ATM Pressure': [1008.20],
                              'Air Humidity':[70.99]
})

print("R2: ",model.rsquared)
print("Energy values obtained: ", model.predict(new_data)[0])