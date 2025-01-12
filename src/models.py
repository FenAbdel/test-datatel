import pandas as pd
import numpy as np
import lightgbm as lgb


def create_future_features(last_date, X_features, periods=90):
    """
    Crée un DataFrame avec les features pour les dates futures
    """
    # Générer les dates futures
    future_dates = pd.date_range(start=last_date, periods=periods + 1)[1:]
    
    # Créer les features numériques
    future_df = pd.DataFrame({
        'marketing_score': np.linspace(X_features['marketing_score'].iloc[-1], 
                                    X_features['marketing_score'].iloc[-1] * 1.05, 
                                    len(future_dates)),
        'competition_index': np.linspace(X_features['competition_index'].iloc[-1],
                                    X_features['competition_index'].iloc[-1] * 1.05,
                                    len(future_dates)),
        'customer_satisfaction': [X_features['customer_satisfaction'].iloc[-1]] * len(future_dates),
        'purchasing_power_index': np.linspace(X_features['purchasing_power_index'].iloc[-1],
                                            X_features['purchasing_power_index'].iloc[-1] * 1.05,
                                            len(future_dates)),
        'weather_condition': np.random.randint(0, 3, size=len(future_dates)),
        'store_traffic': np.random.uniform(-0.5, 1.5, size=len(future_dates)),
        'public_transport': np.random.randint(0, 4, size=len(future_dates))
    })
    
    # Ajouter les colonnes catégorielles (dummies)
    for col in X_features.columns:
        if col not in future_df.columns and col in X_features.columns:
            future_df[col] = 0
    
    # S'assurer que toutes les colonnes sont dans le même ordre
    future_df = future_df[X_features.columns]
    
    return future_df, future_dates

def forecast_revenues(data, target_phone, future_periods=90):
    """
    Fait des prévisions pour un modèle de téléphone spécifique
    """
    # Séparer les features et la cible
    target_columns = ["jPhone_Pro_revenue", "Kaggle_Pixel_5_revenue", "Planet_SX_revenue"]
    feature_columns = [col for col in data.columns if col not in target_columns + ['date']]
    
    X = data[feature_columns]
    y = data[target_phone]
    
    # Entraîner le modèle
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42,
        verbose=-1  # Réduire les messages de warning
    )
    model.fit(X, y)
    
    # Créer les features futures
    last_date = pd.to_datetime(data['date'].iloc[-1])
    future_features, future_dates = create_future_features(last_date, X, future_periods)
    
    # Faire des prévisions pour chaque ville
    all_predictions = []
    cities = ['Marseille','Paris']
    
    for city in cities:
        future_features_city = future_features.copy()
        
        # Mettre à jour les colonnes de ville
        for other_city in cities:
            col = f'city_{other_city}'
            future_features_city[col] = (other_city == city)
        
        # Faire la prédiction
        predictions = model.predict(future_features_city)
        
        # Créer un DataFrame avec les résultats
        city_predictions = pd.DataFrame({
            'date': future_dates,
            'city': city,
            'predicted_revenue': predictions
        })
        
        all_predictions.append(city_predictions)
    
    # Combiner toutes les prédictions
    final_predictions = pd.concat(all_predictions, ignore_index=True)
    return final_predictions

def forecast_all_phones(data):
    """
    Fait des prévisions pour tous les modèles de téléphone
    """
    phones = ["jPhone_Pro_revenue", "Kaggle_Pixel_5_revenue", "Planet_SX_revenue"]
    all_forecasts = {}
    
    for phone in phones:
        print(f"Génération des prévisions pour {phone}...")
        forecasts = forecast_revenues(data, phone)
        all_forecasts[phone] = forecasts
    
    return all_forecasts