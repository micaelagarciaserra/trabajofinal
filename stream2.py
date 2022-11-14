import streamlit as st 
from joblib import load 
import shap
import streamlit.components.v1 as components
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])
    

def load_model():
    model = load('modeloxgb.joblib') 
    return model


# funcion que me permite integrar un grafico de shap con streamlit
def st_shap(plot, height=None):
    js=shap.getjs()
    shap_html = f"<head>{js}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    
# aca empieza la 'pagina'
st.title("Predictor de Videojuegos")
#st.write("Esta es mi primera pagina")

# entreno el modelo
model=load_model()

st.header('Elija las variables del juego que quiere saber si llega al millon de ventas:')

"""
tipo = st.selectbox(
     'Tipo de inmueble?',['PH', 'apartment', 'house', 'store'])
"""

options_regiones = st.multiselect(
    '¿En que regiones contiene ventas?',
    ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])

st.write('Seleccionaste:', options_regiones)

options_certificaciones = st.multiselect(
    '¿Que certificacion tiene?',
    ['certification_AO', 'certification_E', 'certification_E10+', 'certification_EC', 'certification_M', 'certification_RP', 'certification_T'])

st.write('Seleccionaste:', options_certificaciones)

pred=[options_regiones,options_certificaciones]

df_pred=pd.DataFrame([pred], columns=['options_regiones','options_certificaciones'])
#X_pred=transformer.transform(df_pred)
X_pred=model.transform(df_pred)


lista_features=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales','certification_AO', 'certification_E', 
'certification_E10+', 'certification_EC', 'certification_M', 'certification_RP', 'certification_T']
# printeamos el grafico
st.subheader('Analizando la prediccion:')


explainer = shap.TreeExplainer(model)
# genero la expliacion para los datos del test

shap_value = explainer.shap_values(X_pred)
st_shap(shap.force_plot(explainer.expected_value, shap_value, X_pred,feature_names=lista_features))


