import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

RANDOM_STATE = 42
DEFAULT_DATA_PATH = "Amazon_BestSelling_Books_500.xls"

st.set_page_config(page_title="Amazon Books ML Demo", layout="wide")

st.title("ML-демонстрация: классификация книг Amazon")
st.write("Модель предсказывает категорию книги: Fiction или Non-Fiction.")

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(DEFAULT_DATA_PATH)

uploaded_file = st.sidebar.file_uploader("Загрузить CSV/XLS-файл", type=["csv", "xls"])

try:
    df = load_data(uploaded_file)
except Exception as exc:
    st.error(f"Не удалось загрузить данные: {exc}")
    st.stop()

df = df.drop_duplicates()

required_col = "Category"
if required_col not in df.columns:
    st.error("В датасете должен быть столбец Category.")
    st.stop()

st.subheader("Первые строки датасета")
st.dataframe(df.head(20), use_container_width=True)

data = df.copy()
current_year = 2026

data["Book Age"] = current_year - data["Year Published"]
data["Reviews Log"] = np.log1p(data["Reviews"])
data["Price Per Rating"] = data["Price (USD)"] / data["Rating"].replace(0, np.nan)
data["BSR Log"] = np.log1p(data["Amazon BSR"])
data["Is Paperback"] = (data["Format"] == "Paperback").astype(int)

drop_cols = ["Category", "Title", "ISBN", "Amazon URL"]
drop_cols = [col for col in drop_cols if col in data.columns]

X = data.drop(columns=drop_cols)
y = data["Category"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

st.sidebar.header("Гиперпараметры Random Forest")
n_estimators = st.sidebar.slider("n_estimators", min_value=10, max_value=500, value=100, step=10)
max_depth_value = st.sidebar.slider("max_depth", min_value=1, max_value=30, value=10, step=1)
use_unlimited_depth = st.sidebar.checkbox("max_depth = None", value=False)
min_samples_split = st.sidebar.slider("min_samples_split", min_value=2, max_value=20, value=2, step=1)
test_size = st.sidebar.slider("Размер тестовой выборки", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

max_depth = None if use_unlimited_depth else max_depth_value

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=test_size,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE
    ))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
    "F1-macro": f1_score(y_test, y_pred, average="macro"),
    "ROC AUC": roc_auc_score(y_test, y_proba)
}

st.subheader("Качество модели")
cols = st.columns(4)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.3f}")

st.subheader("Отчет классификации")
report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)
st.dataframe(pd.DataFrame(report).T, use_container_width=True)

st.subheader("Матрица ошибок")
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(ax=ax)
st.pyplot(fig)

st.subheader("Важность признаков Random Forest")
rf = model.named_steps["model"]
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
importances = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False).head(20)

st.dataframe(importances, use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.barh(importances["feature"][::-1], importances["importance"][::-1])
ax2.set_title("Топ-20 важных признаков")
ax2.set_xlabel("Важность")
st.pyplot(fig2)

st.subheader("Проверка на одной книге из датасета")
row_index = st.slider("Номер строки", min_value=0, max_value=len(X) - 1, value=0)
sample = X.iloc[[row_index]]
pred = model.predict(sample)[0]
proba = model.predict_proba(sample)[0]

st.write("Исходная книга:")
st.dataframe(df.iloc[[row_index]], use_container_width=True)

st.write(f"Предсказанный класс: **{label_encoder.inverse_transform([pred])[0]}**")
st.write(pd.DataFrame({
    "class": label_encoder.classes_,
    "probability": proba
}))
