import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle
# Commentez la ligne suivante si vous ne prévoyez pas d'utiliser le profilage
# from ydata_profiling import ProfileReport

# ------------------Exploration des données-----------------------#
data = pd.read_csv('/Users/mac/Downloads/Financial_inclusion_dataset.csv')
data.head()
data.describe()
data.info()
data.isna().sum()
data.shape
# --------------------Rapport de profilage-------------------#
# Commentez la ligne suivante si vous ne prévoyez pas d'utiliser le profilage
# Profil = ProfileReport(data, title='Pandas Profiling')
# Profil.to_file("your_report_name.html")
#---------------Gestion des valeurs manquantes et des doublons------------#
data = data.drop_duplicates()
# Encogade des variables
for column in data.columns:
    if data[column].isin(['Yes', 'No']).all():
        data[f'{column}'] = data[column].map({'Yes': 1, 'No': 0})
#-------------------Encodage des variables-----------------------#
data['gender_of_respondent'] = data['gender_of_respondent'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, columns=['country', 'location_type', 'relationship_with_head', 'marital_status', 'job_type'], prefix=['country', 'location_type', 'relationship_with_head', 'marital_status', 'job_type'])

label_encoder = LabelEncoder()
data['education_level'] = label_encoder.fit_transform(data['education_level'])
# Preparation des donnees
df = data.drop(['uniqueid'], axis=1)
X = df.drop('bank_account', axis=1)
y = df['bank_account']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialiser le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Entraîner le modèle
model.fit(X_train, y_train)
# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)
# Évaluer la performance du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy}")
# Charger le modèle préalablement entraîné
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Fonction de prédiction
def make_prediction(df):
    required_columns = df.columns.tolist()

    # Vérifier si toutes les colonnes nécessaires sont présentes dans le DataFrame
    for col in required_columns:
        if col not in df.columns:
            st.error(f"La colonne '{col}' est manquante dans le DataFrame.")
            return None

    # Effectuer la prédiction
    prediction = model.predict(df)

    return prediction
def main():
    st.title("Application de Prédiction Simplifiée")


    # Ajouter des champs de saisie pour chaque variable
    gender = st.radio("Genre", ['Male', 'Female'])
    year = st.selectbox("Année", df['year'].unique())
    # Ajouter d'autres champs de saisie pour les autres variables



    # Convertir le dictionnaire en un DataFrame pour la prédiction
    user_data = pd.DataFrame(df)

    # Ajouter un bouton de prédiction
    if st.button("Lancer la Prédiction"):
        # Effectuer la prédiction avec la fonction définie précédemment
        prediction_result = make_prediction(user_data)

        # Vérifier si la prédiction a réussi
        if prediction_result is not None:
            # Afficher le résultat de la prédiction
            st.write(f"Résultat de la prédiction : {prediction_result[0]}")
        else:
            st.error("Erreur lors de la prédiction. Veuillez vérifier les entrées utilisateur.")


if __name__ == "__main__":
    main()
