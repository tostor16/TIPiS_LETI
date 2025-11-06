# bank_marketing_hw.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_classif
from sklearn.feature_extraction import DictVectorizer

# Загрузка данных
df = pd.read_csv('bank-full.csv', sep=';')

# Выбор нужных столбцов
columns_needed = [
    'age', 'job', 'marital', 'education', 'balance', 'housing',
    'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
    'previous', 'poutcome', 'y'
]
df = df[columns_needed]

print("=" * 70)
print("АНАЛИЗ ДАННЫХ BANK MARKETING - РЕЗУЛЬТАТЫ")
print("=" * 70)

# Вопрос 1
education_mode = df['education'].value_counts().index[0]
education_counts = df['education'].value_counts()
print("\nВОПРОС 1: Какое самое частое значение для столбца education?")
print("Распределение значений education:")
for education, count in education_counts.items():
    print(f"  {education}: {count}")
print(f"Ответ: {education_mode}")

# Вопрос 2
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
corr_matrix = df[numeric_cols].corr()
corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs != 1.0]
top_corr_pair = corr_pairs.index[0]
top_corr_value = corr_pairs.iloc[0]

print("\nВОПРОС 2: Какие два признака имеют наибольшую корреляцию?")
print("Топ-3 пар по корреляции:")
for i, ((col1, col2), corr) in enumerate(corr_pairs.head(3).items(), 1):
    print(f"  {i}. {col1} & {col2}: {corr:.4f}")
print(f"Ответ: {top_corr_pair[0]} и {top_corr_pair[1]} (корреляция: {top_corr_value:.4f})")

# Кодируем y и разделяем данные
df['y'] = (df['y'] == 'yes').astype(int)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

# Вопрос 3
categorical = ['job', 'marital', 'education', 'housing', 'contact', 'poutcome', 'month']

def calculate_mi(series):
    return mutual_info_classif(series.values.reshape(-1, 1), df_train['y'].values, random_state=42)[0]

mi_scores = {}
for col in categorical:
    series = df_train[col].astype('category').cat.codes
    mi_scores[col] = calculate_mi(series)

max_mi_feature = max(mi_scores, key=mi_scores.get)
max_mi_value = mi_scores[max_mi_feature]

print("\nВОПРОС 3: Какая категориальная переменная имеет наибольшую взаимную информацию?")
print("Взаимная информация с целевой переменной:")
for k, v in sorted(mi_scores.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:.4f}")
print(f"Ответ: {max_mi_feature} (MI = {max_mi_value:.4f})")

# Подготовка данных для моделей
numerical = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
categorical = ['job', 'marital', 'education', 'housing', 'contact', 'month', 'poutcome']

train_dict = df_train[numerical + categorical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dict)

val_dict = df_val[numerical + categorical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_train = df_train['y'].values
y_val = df_val['y'].values

# Вопрос 4
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print("\nВОПРОС 4: Точность логистической регрессии на валидационном наборе")
print(f"Точность модели: {accuracy:.4f}")
print(f"Ответ: {accuracy:.1f}")

# Вопрос 5
features = numerical + categorical
base_accuracy = accuracy

diff = {}
for feature in features:
    reduced_features = [f for f in features if f != feature]
    train_dict_red = df_train[reduced_features].to_dict(orient='records')
    dv_red = DictVectorizer(sparse=False)
    X_train_red = dv_red.fit_transform(train_dict_red)
    
    val_dict_red = df_val[reduced_features].to_dict(orient='records')
    X_val_red = dv_red.transform(val_dict_red)
    
    model_red = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model_red.fit(X_train_red, y_train)
    
    y_pred_red = model_red.predict(X_val_red)
    acc_red = accuracy_score(y_val, y_pred_red)
    diff[feature] = base_accuracy - acc_red

min_diff_feature = min(diff, key=lambda x: abs(diff[x]))
min_diff_value = diff[min_diff_feature]

print("\nВОПРОС 5: Какой признак имеет наименьшую разницу при исключении?")
print("Изменение точности при исключении признаков:")
for k, v in sorted(diff.items(), key=lambda x: abs(x[1])):
    print(f"  {k}: {v:+.4f}")
print(f"Ответ: {min_diff_feature} (разница: {min_diff_value:.4f})")

# Вопрос 6
best_C = None
best_acc = 0
results = {}

for C in [0.01, 0.1, 1, 10]:
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    results[C] = acc
    
    if acc > best_acc:
        best_acc = acc
        best_C = C
    elif acc == best_acc and C < best_C:
        best_C = C

print("\nВОПРОС 6: Какое значение C приводит к наилучшей точности?")
print("Точность при разных значениях C:")
for C, acc in results.items():
    print(f"  C = {C}: {acc:.4f}")
print(f"Ответ: {best_C} (точность: {best_acc:.4f})")

print("\n" + "=" * 70)
print("ИТОГОВЫЕ ОТВЕТЫ:")
print("=" * 70)
print(f"1. Самое частое образование: {education_mode}")
print(f"2. Наибольшая корреляция: {top_corr_pair[0]} и {top_corr_pair[1]}")
print(f"3. Наибольшая взаимная информация: {max_mi_feature}")
print(f"4. Точность модели: {accuracy:.1f}")
print(f"5. Наименее влиятельный признак: {min_diff_feature}")
print(f"6. Оптимальное значение C: {best_C}")
print("=" * 70)