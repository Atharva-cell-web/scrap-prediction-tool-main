import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_importance():
    print("📊 Generating Feature Importance Plot...")
    
    # 1. Load the trained model and feature names
    model_path = os.path.join(MODEL_DIR, "final_scrap_model.pkl")
    features_path = os.path.join(MODEL_DIR, "final_model_features.pkl")
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Train it first.")
        return

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    
    # 2. Extract Importance
    importances = model.feature_importances_
    
    # 3. Create a DataFrame
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(15) # Top 15
    
    print("\n🏆 TOP 5 PREDICTORS OF SCRAP:")
    print(df_imp.head(5))

    # 4. Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis')
    plt.title('Top 15 Sensors Predicting High Scrap Risk', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Sensor Name', fontsize=12)
    plt.tight_layout()
    
    # 5. Save
    save_path = os.path.join(OUTPUT_DIR, "final_feature_importance.png")
    plt.savefig(save_path)
    print(f"\n✅ Graph saved to: {save_path}")

if __name__ == "__main__":
    plot_importance()