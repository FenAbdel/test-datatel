import pandas as pd

def load_data(file_path):
    """
    Charge les données à partir d'un fichier CSV
    """
    data = pd.read_csv(file_path)
    data.rename(columns={"Unnamed: 0": "date"}, inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    return data

def handle_missing_values(data):
    """
    Gère les valeurs manquantes dans le dataset
    """
    # Variables numériques à imputer avec la médiane
    numerical_vars = [
        'marketing_score', 'competition_index', 'customer_satisfaction',
        'purchasing_power_index', 'store_traffic', 
        'jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue'
    ]
    
    for var in numerical_vars:
        data[var].fillna(data[var].median(), inplace=True)
    
    # Variables catégorielles à imputer avec le mode
    categorical_vars = ['weather_condition', '5g_phase', 'public_transport']
    
    for var in categorical_vars:
        data[var].fillna(data[var].mode()[0], inplace=True)
    
    return data

def process_tech_events(data):
    """
    Crée des variables dummy pour les événements technologiques
    """
    return pd.get_dummies(data, columns=['tech_event'], prefix='tech_event', drop_first=False)



if __name__ == "__main__":
    # Chemins des fichiers
    input_path = "./data/raw/telecom_sales_data.csv"
    output_path = "./data/processed/telecom_sales_data_processed.csv"
    
    # Pipeline de traitement
    data = load_data(input_path)
    data = handle_missing_values(data)
    data = process_tech_events(data)
    
    data.to_csv(output_path, index=False)

