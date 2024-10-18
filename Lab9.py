import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
import string

# Inyectar CSS personalizado para aplicar la paleta de colores
st.markdown("""
    <style>
    body {
        background-color: #F9F8F2; /* Fondo blanco cálido */
    }
    .css-1d391kg { /* Fondo principal de la app */
        background-color: #F9F8F2 !important;
    }
    .css-18e3th9 { /* Pestañas de la app */
        background-color: #165A2B !important;
        color: white !important;
    }
    .stButton>button { /* Estilo de los botones */
        background-color: #5D8D6C !important;
        color: white !important;
        border-radius: 8px !important;
    }
    h1, h2, h3 { /* Colores para los encabezados */
        color: #165A2B !important;
    }
    .stTabs { /* Estilo para las pestañas */
        color: #5D8D6C !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Descargar el léxico de VADER
nltk.download('vader_lexicon')

# Inicializar el analizador de sentimiento VADER
sia = SentimentIntensityAnalyzer()

# Cargar datos
data = pd.read_csv("train.csv")

# Función para limpiar el texto
def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Eliminar URLs
    text = re.sub(r'\@\w+|\#|\d+', '', text)  # Quitar hashtags, menciones, y números
    text = text.translate(str.maketrans('', '', string.punctuation))  # Eliminar signos de puntuación
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])  # Eliminar stopwords
    return text

# Aplicar la función de limpieza a la columna de texto antes de cualquier operación
data['cleaned_text'] = data['text'].apply(clean_text)

# Separar en tweets de desastres y no desastres para obtener palabras más comunes
disaster_tweets = data[data['target'] == 1]['cleaned_text']
non_disaster_tweets = data[data['target'] == 0]['cleaned_text']

# Calcular la frecuencia de las palabras
disaster_words = Counter(" ".join(disaster_tweets).split())
non_disaster_words = Counter(" ".join(non_disaster_tweets).split())

# Aplicar el análisis de sentimiento a cada tweet después de la limpieza
data['vader_scores'] = data['cleaned_text'].apply(sia.polarity_scores)

# Extraer las puntuaciones de negatividad, neutralidad, positividad y compound
data['negativity_vader'] = data['vader_scores'].apply(lambda x: x['neg'])
data['neutrality_vader'] = data['vader_scores'].apply(lambda x: x['neu'])
data['positivity_vader'] = data['vader_scores'].apply(lambda x: x['pos'])
data['compound_vader'] = data['vader_scores'].apply(lambda x: x['compound'])

# Clasificar los tweets en positivo, negativo o neutral basado en el score compound
data['sentiment_vader'] = data['compound_vader'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Preprocesamiento de textos para TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_vectorized = vectorizer.fit_transform(data['cleaned_text'])
y = data['target']

# Split del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Definir modelos
logreg = LogisticRegression()
dtree = DecisionTreeClassifier()
rf = RandomForestClassifier()

# Entrenar modelos
logreg.fit(X_train, y_train)
dtree.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Realizar predicciones
logreg_pred = logreg.predict(X_test)
dtree_pred = dtree.predict(X_test)
rf_pred = rf.predict(X_test)

# Calcular la precisión de los modelos
logreg_acc = accuracy_score(y_test, logreg_pred)
dtree_acc = accuracy_score(y_test, dtree_pred)
rf_acc = accuracy_score(y_test, rf_pred)

# Configurar el dashboard con pestañas
st.title("Disaster Tweet Analysis Dashboard")

# Crear las pestañas
tab1, tab2, tab3, tab4 = st.tabs(["Exploración de Datos", "Comparación de Modelos", "Matriz de Confusión", "Gráficos Adicionales"])

# Pestaña 1: Exploración de Datos
with tab1:
    st.header("Exploración de Datos")
    keyword_filter = st.selectbox("Seleccione Keyword", options=[None] + list(data['keyword'].dropna().unique()), index=0)
    target_filter = st.selectbox("Seleccione Target (1: Desastre, 0: No Desastre)", options=[None, 0, 1], index=1)

    if keyword_filter or target_filter is not None:
        filtered_data = data.copy()
        if keyword_filter:
            filtered_data = filtered_data[filtered_data['keyword'] == keyword_filter]
        if target_filter is not None:
            filtered_data = filtered_data[filtered_data['target'] == target_filter]

        st.write("### Datos filtrados")
        st.write(filtered_data.head())

# Pestaña 2: Comparación de Modelos
with tab2:
    st.header("Comparación de Desempeño de Modelos")
    # Opción para seleccionar los modelos
    models_selected = st.multiselect(
        "Modelos a comparar:",
        ["Logistic Regression", "Decision Tree", "Random Forest"],
        default=["Logistic Regression", "Decision Tree", "Random Forest"]
    )

    # Mostrar el desempeño de los modelos seleccionados
    performance_data = []
    if "Logistic Regression" in models_selected:
        performance_data.append({"Model": "Logistic Regression", "Accuracy": logreg_acc})
    if "Decision Tree" in models_selected:
        performance_data.append({"Model": "Decision Tree", "Accuracy": dtree_acc})
    if "Random Forest" in models_selected:
        performance_data.append({"Model": "Random Forest", "Accuracy": rf_acc})

    performance_df = pd.DataFrame(performance_data)
    st.write(performance_df)

# Pestaña 3: Matriz de Confusión
with tab3:
    st.header("Matriz de Confusión de Modelos")

    if "Logistic Regression" in models_selected:
        st.write("### Matriz de Confusión: Logistic Regression")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, logreg_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    if "Decision Tree" in models_selected:
        st.write("### Matriz de Confusión: Decision Tree")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, dtree_pred), annot=True, fmt="d", cmap="Greens", ax=ax)
        st.pyplot(fig)

    if "Random Forest" in models_selected:
        st.write("### Matriz de Confusión: Random Forest")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt="d", cmap="Oranges", ax=ax)
        st.pyplot(fig)

# Pestaña 4: Gráficos Adicionales
with tab4:
    # Gráfico de cantidad de tweets de No Desastres vs Desastres con Plotly
    st.subheader("Cantidad de Tweets de No Desastres vs Desastres")

    # Contar el número de tweets relacionados con desastres y los que no lo son
    disaster_counts = data['target'].value_counts().reset_index()
    disaster_counts.columns = ['Target', 'Count']
    disaster_counts['Target'] = disaster_counts['Target'].map({0: 'No Desastre', 1: 'Desastre'})

    # Crear el gráfico de barras interactivo
    fig_disaster = px.bar(disaster_counts, x='Target', y='Count', 
                          color='Target', 
                          color_discrete_sequence=['#2F2777', '#38B586'], 
                          title='Cantidad de Tweets de No Desastres vs Desastres',
                          labels={'Target': 'Categoría de Tweet', 'Count': 'Cantidad'})
    fig_disaster.update_layout(yaxis_range=[0, 5000])  # Límite del eje y

    # Mostrar el gráfico interactivo en Streamlit
    st.plotly_chart(fig_disaster)

    # Gráfico de distribución de sentimientos con Plotly
    st.subheader("Distribución de Sentimientos en los Tweets")

    # Crear la columna 'sentiment_category' basada en compound_vader
    data['sentiment_category'] = pd.cut(data['compound_vader'], 
                                        bins=[-float('inf'), -0.75, -0.25, 0.25, 0.75, float('inf')],
                                        labels=['Más Negativos', 'Negativos', 'Neutrales', 'Positivos', 'Más Positivos'],
                                        include_lowest=True)

    # Contar la cantidad de tweets por categoría de sentimiento
    category_counts = data['sentiment_category'].value_counts().reset_index()
    category_counts.columns = ['Sentiment', 'Count']

    # Crear el gráfico de barras interactivo
    fig_sentiment = px.bar(category_counts, x='Sentiment', y='Count', 
                           color='Sentiment', 
                           color_discrete_sequence=['#D0021B', '#F49045', '#AEB8BC', '#8BD646', '#5AC864'], 
                           title='Distribución de Sentimientos en los Tweets',
                           labels={'Sentiment': 'Categoría de Sentimiento', 'Count': 'Número de Tweets'})

    # Mostrar el gráfico interactivo en Streamlit
    st.plotly_chart(fig_sentiment)

    # Distribución Global de Tweets sobre Desastres con Plotly
    st.subheader("Distribución Global de Tweets sobre Desastres")
    manual_locations = {
        'New York, USA': [40.7128, -74.0060],
        'London, UK': [51.5074, -0.1278],
        'Los Angeles, California': [34.0522, -118.2437],
        'United States': [37.0902, -95.7129],
        'Canada': [56.1304, -106.3468],
        'Nigeria': [9.0820, 8.6753],
        'India': [20.5937, 78.9629],
        'Mumbai': [19.0760, 72.8777],
        'Washington, DC': [38.9072, -77.0369],
        'Kenya': [-1.2921, 36.8219]
    }

    # Crear el DataFrame con estas ubicaciones manuales y sus coordenadas
    manual_location_df = pd.DataFrame.from_dict(manual_locations, orient='index', columns=['Latitude', 'Longitude']).reset_index()
    manual_location_df.columns = ['Location', 'Latitude', 'Longitude']

    # Asignar un conteo manual de tweets basado en los datos originales
    manual_location_df['Count'] = [71, 45, 26, 104, 29, 28, 24, 22, 21, 20]

    # Crear un mapamundi interactivo con colores personalizados
    fig_map = px.scatter_geo(manual_location_df,
                             lat='Latitude',
                             lon='Longitude',
                             hover_name='Location',
                             size='Count',
                             color='Count',  # Usar la cantidad para definir el color
                             color_continuous_scale=[[0, 'lightblue'], [0.25, 'green'], [0.5, 'yellow'], [1, 'red']],  # Escala personalizada
                             scope='world',  # Cambiamos a escala mundial
                             title='Distribución Global de Tweets sobre Desastres')

    # Mostrar el gráfico interactivo en Streamlit
    st.plotly_chart(fig_map)
    
    # Gráfico de palabras más frecuentes en tweets de desastres
    st.subheader("Palabras más comunes en tweets de desastres")
    
    # Obtener las palabras más comunes en tweets de desastres (simulando disaster_words)
    common_disaster_words = disaster_words.most_common(20)  # Simula tener los datos de las palabras más frecuentes
    
    # Convertir las palabras en DataFrame para usar con Plotly
    disaster_words_df = pd.DataFrame(common_disaster_words, columns=['Word', 'Count'])
    
    # Crear el gráfico de barras interactivo para palabras más comunes en tweets de desastres
    fig_disaster_words = px.bar(disaster_words_df, x='Count', y='Word', 
                                orientation='h', 
                                title='Palabras más frecuentes en tweets de desastres',
                                labels={'Count': 'Frecuencia', 'Word': 'Palabra'})
    
    # Mostrar el gráfico interactivo en Streamlit
    st.plotly_chart(fig_disaster_words)

    # Gráfico de palabras más frecuentes en tweets que no son de desastres
    st.subheader("Palabras más comunes en tweets que no son de desastres")
    
    # Obtener las palabras más comunes en tweets que no son de desastres (simulando non_disaster_words)
    common_non_disaster_words = non_disaster_words.most_common(20)  # Simula tener los datos de las palabras más frecuentes
    
    # Convertir las palabras en DataFrame para usar con Plotly
    non_disaster_words_df = pd.DataFrame(common_non_disaster_words, columns=['Word', 'Count'])
    
    # Crear el gráfico de barras interactivo para palabras más comunes en tweets que no son de desastres
    fig_non_disaster_words = px.bar(non_disaster_words_df, x='Count', y='Word', 
                                    orientation='h', 
                                    title='Palabras más frecuentes en tweets que no son de desastres',
                                    labels={'Count': 'Frecuencia', 'Word': 'Palabra'})
    
    # Mostrar el gráfico interactivo en Streamlit
    st.plotly_chart(fig_non_disaster_words)

