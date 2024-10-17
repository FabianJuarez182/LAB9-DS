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

# Descargar el léxico de VADER
nltk.download('vader_lexicon')

# Inicializar el analizador de sentimiento VADER
sia = SentimentIntensityAnalyzer()

# Cargar datos
data = pd.read_csv("train.csv")

# Función para obtener los puntajes de sentimiento con VADER
def get_vader_sentiment_scores(tweet):
    return sia.polarity_scores(tweet)

# Aplicar el análisis de sentimiento a cada tweet
data['vader_scores'] = data['text'].apply(get_vader_sentiment_scores)

# Extraer las puntuaciones de negatividad, neutralidad, positividad y compound
data['negativity_vader'] = data['vader_scores'].apply(lambda x: x['neg'])
data['neutrality_vader'] = data['vader_scores'].apply(lambda x: x['neu'])
data['positivity_vader'] = data['vader_scores'].apply(lambda x: x['pos'])
data['compound_vader'] = data['vader_scores'].apply(lambda x: x['compound'])

# Clasificar los tweets en positivo, negativo o neutral basado en el score compound
data['sentiment_vader'] = data['compound_vader'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Inicializar listas vacías para los tweets
negative_tweets = []
positive_tweets = []
neutral_tweets = []

# Clasificar los tweets en las listas según el sentimiento
for index, row in data.iterrows():
    if row['sentiment_vader'] == 'Negative':
        negative_tweets.append(row['text'])
    elif row['sentiment_vader'] == 'Positive':
        positive_tweets.append(row['text'])
    else:
        neutral_tweets.append(row['text'])

# Preprocesamiento de textos
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_vectorized = vectorizer.fit_transform(data['text'])
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
    st.header("Distribución de Sentimientos en los Tweets")

    # Crear la columna 'sentiment_category' basada en compound_vader
    data['sentiment_category'] = pd.cut(data['compound_vader'], 
                                        bins=[-float('inf'), -0.75, -0.25, 0.25, 0.75, float('inf')],
                                        labels=['Más Negativos', 'Negativos', 'Neutrales', 'Positivos', 'Más Positivos'],
                                        include_lowest=True)

    # Contar la cantidad de tweets por categoría de sentimiento
    category_counts = data['sentiment_category'].value_counts().sort_index()

    # Crear el gráfico de barras
    fig, ax = plt.subplots(figsize=(8, 5))
    category_counts.plot(kind='bar', color=['#D0021B', '#F49045', '#AEB8BC', '#8BD646', '#5AC864'], ax=ax)
    ax.set_title('Distribución de Sentimientos en los Tweets')
    ax.set_xlabel('Categoría de Sentimiento')
    ax.set_ylabel('Número de Tweets')
    ax.set_xticklabels(category_counts.index, rotation=45)
    plt.tight_layout()

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

