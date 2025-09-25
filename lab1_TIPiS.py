from ucimlrepo import fetch_ucirepo 
import pandas as pd

def greater_income():
    full_df = pd.concat([X_df, Y_df])
    count_income = full_df[full_df['income'] == '>50K']['sex'].value_counts()
    return "мужчин" if count_income.get('Male', 0) > count_income.get('Female', 0) else "женщин"

def replace_with_mode():
    for column in X_df.columns:
        if X_df[column].isnull().any():
            mode_values = X_df[column].mode()
            if not mode_values.empty:
                mode_value = mode_values.iloc[0]
                X_df[column] = X_df[column].fillna(mode_value)
                print(f"Заполнено {column}: мода = {mode_value}")
    
adult = fetch_ucirepo(id=2)  
X_df = pd.DataFrame(adult.data.features) 
Y_df = pd.DataFrame(adult.data.targets) 
print(f"1. Число столбцов {X_df.shape[1]}", end='\n\n\n') 
print("2. Пропуски в данных:", X_df.isnull().sum(), sep='\n', end='\n\n\n')
print(f"3. Количество уникальных значений в race {X_df['race'].nunique()}", end='\n\n\n')
print(f"4. Медиана hours-per-week {X_df['hours-per-week'].mean()}", end='\n\n\n')
print(f"5. Зарплата больше у {greater_income()}", end='\n\n\n')
print("6. Замена пустых значений на моду в каждом столбце")
replace_with_mode()