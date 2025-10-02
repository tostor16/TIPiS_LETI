from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt

# Получаем DataSet
adult = fetch_ucirepo(id=2) 
  
# Извлекаем DataFrame
X = adult.data.features # DataFrame с исхожными данными
y = adult.data.targets  # DataFrame с полученными данными

df = pd.concat([X, y], axis=1)


print(f"кол-во столбцов (объединенной DataFrame):\t {df.shape[1]}\n") # кол-во столбцов (объединенной DataFrame)
print(f"столбцы где пропущеены зн-ия:\n {df.isnull().sum()}\n") # столбцы где пропущеены зн-ия
print(f"кол-во уникальных значений в столбце race:\t {df.race.unique()}\n") # кол-во уникальных значений в столбце race
print(f"медиана hours-per-week:\t{df['hours-per-week'].median()}\n") # медиана hours-per-week
df["hours-per-week"].hist()
plt.show()

df_select = df[df['income'] == '>50K']
man = len(df_select[df_select['sex'] == "Male"])
female = len(df_select[df_select['sex'] == "Female"])
difference = man - female
print(f"Мужчин зарабатывающик >50K больше на:\t{difference}\n")

mode_workclass = df["workclass"].mode()[0] # находим самое распространенное зн-ие
df["workclass"] = df["workclass"].fillna(mode_workclass) # заполняем пропуски
mode_native_country = df["native-country"].mode()[0] # находим самое распространенное зн-ие
df["native-country"] = df["native-country"].fillna(mode_native_country) # заполняем пропуски
mode_occupation = df["occupation"].mode()[0] # находим самое распространенное зн-ие
df["occupation"] = df["occupation"].fillna(mode_occupation) # заполняем пропуски
print(f"Пропущенные зн-ия после заполнения mode: \n{df.isnull().sum()}", sep = "\n") # столбцы где пропущеены зн-ия
