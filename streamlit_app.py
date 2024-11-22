import streamlit as st
import pandas as pd
import openpyxl
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score,accuracy_score,recall_score,precision_score,classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score, mean_absolute_percentage_error



# Show title and description.
st.title("Trabajo Final Grupo 6")
st.write(
    "Esta página hace predicciones de probabilidad de ocurrencia de accidentes de tránsito mediante modelos de Machine Learning"
)


# Subida de archivo
uploaded_file = st.file_uploader("Sube un archivo CSV o XLSX", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Verificar el tipo de archivo y leerlo apropiadamente
        if uploaded_file.name.endswith(".csv"):
            # Leer archivo CSV
            df_data = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
            st.write("Vista previa del archivo CSV:")
        elif uploaded_file.name.endswith(".xlsx"):
            # Leer archivo XLSX
            df_data = pd.read_excel(uploaded_file, engine="openpyxl")
            st.write("Vista previa del archivo XLSX:")
        
        # Mostrar el DataFrame
        st.write("Datos originales")
        st.dataframe(df_data)

            #cambiar nombres de las columnas
        cambiar_nombres = {
          'AÑO': 'año',
          'MES': 'mes',
          'CONCESION': 'concesion',
          'ENTIDAD PRESTADORA': 'entidad_prestadora',
          'INGRESOS TOTAL': 'ingreso_total',
          'TRAFICO TOTAL': 'trafico_total',
          'TOTAL RECLAMOS RESUELTOS': 'total_reclamos_resueltos',
          'TOTAL RECLAMOS PRESENTADOS': 'total_reclamos_presentados',
          'TOTAL RECLAMOS EN PROCESO': 'total_reclamos_enproceso',
          'TOTAL LLAMADAS EMERGENCIAS': 'total_llamadas_emerg',
          'TOTAL ASISTENCIA MECANICA': 'total_asistencia_mecanica',
          'TOTAL MES ACCIDENTES': 'target_total_mes_accidentes',
        }
        df_data = df_data.rename (columns=cambiar_nombres)
        
        
        #Traer los encoders
        
        # Cargar el objeto `encoderMMS` desde el archivo
        encoderMMS = joblib.load('encoderMMS.pkl')
        
        # Cargar el objeto `oneHE` desde el archivo
        oneHE = joblib.load('oneHE.pkl')
        
        
        
        
        #aislar features año y mes (*1)
        aniomes = df_data[['año', 'mes']]
        
        
        
        #Aplicar el encoder y aislar las features OneHotEncoder (2*)
        columnas = ['concesion','entidad_prestadora']
        features_one_hot = pd.DataFrame(oneHE.transform(df_data[columnas]),
                                      columns=oneHE.get_feature_names_out(columnas),
                                      index=df_data.index)
        
        
        
        #Aplica el encoder y aislar las features MinMaxScaler (3*)
        columnas = ['ingreso_total',	'trafico_total',	'total_reclamos_resueltos',	'total_reclamos_presentados',
                    'total_reclamos_enproceso','total_llamadas_emerg',	'total_asistencia_mecanica']
        
        features_min_max = pd.DataFrame(encoderMMS.transform(df_data[columnas]),
                                     columns=columnas,
                                     index=df_data.index)
        
        
        #aislar target
        columna_target = df_data['target_total_mes_accidentes']
        
        
        # 4 cosas:
        # año y mes tal cual los datos originales (*1)
        # transformacion de One Hot encoder (2*)
        # transformacion de Min Max Scaler (3*)
        # columna de target bajo el nombre "target_total_mes_accidentes" (*4)
        df_tablon_completo_test = pd.concat([aniomes   , features_one_hot,   features_min_max,  columna_target],axis=1)
        
        st.write("Datos transformados")
        st.dataframe(df_tablon_completo_test)
        
        
        #Traer modelos ya entrenados
        tree_model = joblib.load('tree_model.pkl')
        rf_model = joblib.load('rf_model.pkl')
        #LogisticRegression = joblib.load('LogisticRegression.pkl')
        
        best_model_tree = joblib.load('best_model_tree.pkl')
        best_model_rf = joblib.load('best_model_rf.pkl')
        
        
        
        
        #Separar features de target
        las_features = df_tablon_completo_test.drop(columns='target_total_mes_accidentes')
        la_target = df_tablon_completo_test['target_total_mes_accidentes']
        
        
        st.write("Las columnas features")
        st.dataframe(las_features)
        
        st.write("La columna target")
        st.dataframe(la_target)
        
        
        #Realizar prediccion
        tree_prediction = tree_model.predict(las_features)
        rf_prediction = rf_model.predict(las_features)
        #LogisticRegression_prediction = LogisticRegression.predict(las_features)
        
        best_tree_prediction = best_model_tree.predict(las_features)
        best_rf_prediction = best_model_rf.predict(las_features)
        
        
        
        #Mostrar predicción
        
        df_comparar = pd.DataFrame()

        df_comparar['target_original'] = la_target
        #df_comparar['LogisticRegression'] = LogisticRegression_prediction
        df_comparar['DecisionTreeRegressor'] = tree_prediction
        df_comparar['Optimizado_DecisionTreeRegressor'] = best_tree_prediction
        df_comparar['RandomForestRegressor'] = rf_prediction
        df_comparar['Optimizado_RandomForestRegressor'] = best_rf_prediction

        
        
        st.write("Comparación de predicciones")
        st.dataframe(df_comparar)


    except Exception as e:
        st.error(f"Hubo un error al leer el archivo: {e}")
else:
    st.write("Sube un archivo CSV o XLSX para empezar.")

