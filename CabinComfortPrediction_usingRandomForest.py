import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

# Sentetik Veri Oluşturma (NumPy)

np.random.seed(42)
N_samples = 1000

# Girdi Değişkenleri
Temperature = np.random.uniform(18, 26, N_samples) # °C
Humidity = np.random.uniform(20, 70, N_samples)    # %
Noise = np.random.uniform(50, 80, N_samples)       # dB
Airflow = np.random.uniform(0.5, 5, N_samples)     # m³/h
SeatZone_map = {0: 'Front', 1: 'Mid', 2: 'Aft'}
SeatZone = np.random.randint(0, 3, N_samples)
IFE_Brightness = np.random.uniform(10, 100, N_samples) # %
IFE_UsageTime = np.random.uniform(0, 300, N_samples)  # min

# Konfor İndeksi (1-5) için Basit Regresyon Formülü (sentetik)
# Düşük Sıcaklık, Orta Nem, Düşük Gürültü, Orta Hava Akımı, Düşük Parlaklık/Kullanım Süresi = Yüksek Konfor
ComfortIndex_raw = (
    5 - 0.2 * (Temperature - 22)  # Sıcaklık 22'den uzaklaştıkça konfor düşer
    - 0.05 * abs(Humidity - 45)  # Nem 45'ten uzaklaştıkça konfor düşer
    - 0.03 * (Noise - 50)
    + 0.1 * (Airflow - 2)
    - 0.01 * (IFE_Brightness - 50)
    - 0.005 * IFE_UsageTime
    + np.random.normal(0, 0.5, N_samples) # Gürültü/hata
)

# Konfor İndeksi'ni 1-5 aralığına sıkıştıralım
ComfortIndex = np.clip(ComfortIndex_raw, 1, 5).round(0)

# DataFrame oluşturma
data = pd.DataFrame({
    'Temperature': Temperature,
    'Humidity': Humidity,
    'Noise': Noise,
    'Airflow': Airflow,
    'SeatZone': [SeatZone_map[z] for z in SeatZone],
    'IFE_Brightness': IFE_Brightness,
    'IFE_UsageTime': IFE_UsageTime,
    'ComfortIndex': ComfortIndex
})

print("Veri Setinin İlk 5 Satırı:")
print(data.head())

# Özellikler (X) ve Hedef (y) ayırma
X = data.drop('ComfortIndex', axis=1)
y = data['ComfortIndex']

# Eğitim ve Test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ön İşleme Hattı (Pipeline) Oluşturma
# Kategorik ve Sayısal Sütunları Tanımlama
categorical_features = ['SeatZone'] #
numerical_features = ['Temperature', 'Humidity', 'Noise', 'Airflow', 'IFE_Brightness', 'IFE_UsageTime'] #

# Dönüştürücüleri Tanımlama
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features), # Normalizasyon (Min-Max)
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # One-hot encoding
    ],
    remainder='passthrough'
)

# Ön İşlemeyi Uygulama
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# İşlenmiş veriyi DataFrame'e geri dönüştürme (gerekirse)
feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

# Sadece sayısal sütunları içeren bir DataFrame oluşturma
# 'SeatZone' sütununu ve One-Hot-Encoding yapılmamış kategorik sütunları çıkartma.

numerical_data = data.select_dtypes(include=np.number)

# Korelasyon Isı Haritası
plt.figure(figsize=(10, 8))
# Artık korelasyonu sadece sayısal veriler üzerinde hesaplıyoruz
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heat Map (Only Numerical Data)')
plt.show()

# Model Seçimi: Random Forest Regressor
rfr = RandomForestRegressor(random_state=42)

# Hiperparametre Optimizasyonu için Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train_df, y_train)

best_rfr = grid_search.best_estimator_

print("\nEn İyi Hiperparametreler:", grid_search.best_params_)
print("En İyi MSE (Grid Search):", -grid_search.best_score_)

# Tahmin Yapma
y_pred = best_rfr.predict(X_test_df)

# Performans Metrikleri
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performansı (Test Seti):")
print(f"Mean Squared Error (MSE): {mse:.4f}")  # Minimize edilmesi gereken
print(f"R² Skoru: {r2:.4f}")  # Maksimize edilmesi gereken

# 1. Tahmin Edilen vs. Gerçek Konfor Dağılım Grafiği
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Mükemmel Tahmin Çizgisi
plt.xlabel('Actual Comfort Index (y_test)')
plt.ylabel('Predicted Comfort Index (y_pred)')
plt.title('Predicted vs. Actual Comfort Index')
plt.show()

# 2. Önemli Özellikler Grafiği (Feature Importance Plot)
if hasattr(best_rfr, 'feature_importances_'):
    feature_importances = pd.Series(best_rfr.feature_importances_, index=feature_names)
    feature_importances.sort_values(ascending=False, inplace=True)

    plt.figure(figsize=(10, 6))

    # !!! Düzeltilmiş Kod !!!
    # FutureWarning'ı çözmek için 'hue' parametresi eklendi ve 'legend=False' ayarlandı.
    sns.barplot(
        x=feature_importances.values,
        y=feature_importances.index,
        palette="viridis",
        hue=feature_importances.index,  # Y eksenindeki değişkeni hue olarak atıyoruz
        legend=False  # Göstergeyi kapatıyoruz
    )
    # !!! Düzeltilmiş Kod Sonu !!!

    plt.title('Feature Importance Ranking (Random Forest)')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Feature')
    plt.show()