import pandas as pd
from sklearn.preprocessing import LabelEncoder

def extract_date_features(data):
    """
    Extrait les caractéristiques temporelles de la colonne date
    """
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])
    data['day_of_week'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    return data

def encode_categorical_variables(data):
    """
    Encode les variables catégorielles
    """
    encoder = LabelEncoder()
    data['weather_condition'] = encoder.fit_transform(data['weather_condition'])
    data['public_transport'] = encoder.fit_transform(data['public_transport'])
    return data

def create_location_features(data):
    """
    Crée des variables dummy pour les caractéristiques de localisation
    """
    return pd.get_dummies(data, columns=['city', '5g_phase'], drop_first=False)

def engineer_features(data):
    """
    Applique toutes les transformations de feature engineering
    """
    data = extract_date_features(data)
    data = encode_categorical_variables(data)
    data = create_location_features(data)
    return data


if __name__ == "__main__":
    # Charger les données prétraitées
    data = pd.read_csv("./data/processed/telecom_sales_data_processed.csv")
    
    # Appliquer le feature engineering
    data = engineer_features(data)
    
    # Sauvegarder les données finales
    data.to_csv("./data/final/telecom_sales_data_final.csv", index=False)