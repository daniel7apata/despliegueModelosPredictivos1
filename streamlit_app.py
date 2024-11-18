import streamlit as st
from openai import OpenAI
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score,accuracy_score,recall_score,precision_score,classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score, mean_absolute_percentage_error




# Show title and description.
st.title("Trabajo Final Grupo 6")
st.write(
    "Esta página hace predicciones de probabilidad de ocurrencia de accidentes de tránsito mediante modelos de Machine Learning DecisionTreeRegressor y RandomForestRegressor"
)


# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Sube un archivo, xlsx o csv", type=("xlsx", "csv")
)


#procesamiento de archivos subidos
if uploaded_file and question:
    document = uploaded_file.read().decode()




# Cargar los modelos serializados
best_model_tree = joblib.load('best_model_tree.pkl')
best_model_rf = joblib.load('best_model_rf.pkl')
#Cargar conjunto de datos test
df_test = pd.read_pickle('df_final_tablon_completo_test_encoded.pickle')
#Separar features de target
X_test = df_test.drop(columns='target_total_mes_accidentes')
y_test = df_test['target_total_mes_accidentes']
#Realizar prediccion
tree_prediction = best_model_tree.predict(X_test)
rf_prediction = best_model_rf.predict(X_test)
#Mostrar prediccion
st.write("Predicción con DecisionTreeRegressor:", tree_prediction)
st.write("Predicción con RandomForestRegressor:", rf_prediction)


