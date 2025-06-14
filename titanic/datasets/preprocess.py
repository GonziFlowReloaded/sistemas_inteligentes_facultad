import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

# 1) Carga de datos
df = pd.read_csv('tp2/datasets/train.csv')

# 2) Separar features y target
X = df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'])
y = df['Survived']

# 3) Definir columnas por tipo
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
cat_cols = ['Pclass', 'Sex', 'Embarked']

# 4) Pipelines de preprocesamiento
#   - Numérico: imputar mediana + escalar
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

#   - Categórico: imputar moda + one-hot
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
# 5) ColumnTransformer que une ambos
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# 6) Balanceador
ros = RandomOverSampler(random_state=42)

# 7) Pipeline completo: preprocesamiento + balanceo
X_pre = preprocessor.fit_transform(X)
X_balanced, y_balanced = ros.fit_resample(X_pre, y)

# 8) (Opcional) dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

print("Shape original X/y:", X.shape, y.value_counts().to_dict())
print("Shape tras balanceo X/y:",  X_balanced.shape, pd.Series(y_balanced).value_counts().to_dict())

# ——————————————
# 9) Reconstruir DataFrame limpio y exportar a CSV

# Obtener nombres de las nuevas columnas del one-hot encoder
onehot_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)

# Combinar nombres de columnas numéricas + categóricas
all_cols = num_cols + list(onehot_cols)

# Crear DataFrame con X_balanced
df_clean = pd.DataFrame(X_balanced, columns=all_cols)

# Agregar la columna target
df_clean['Survived'] = y_balanced

# Exportar a CSV listo para entrenamiento
output_path = 'tp2/datasets/train_cleaned.csv'
df_clean.to_csv(output_path, index=False)

print(f"CSV limpio exportado en: {output_path}")
