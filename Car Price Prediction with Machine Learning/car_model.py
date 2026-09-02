import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

cedvel = pd.read_csv('car data.csv')

print('Ilk 5 masinin melumati:')
print(cedvel.head())
print('-' * 40)

giris_melumatlari = cedvel[['Year', 'Present_Price', 'Driven_kms']]
hedef_qiymet = cedvel['Selling_Price']

oyrenme_giris, yoxlama_giris, oyrenme_qiymet, yoxlama_qiymet = (
    train_test_split(giris_melumatlari, hedef_qiymet, test_size=0.2, random_state=42)
)

masin_modeli = LinearRegression()
masin_modeli.fit(oyrenme_giris, oyrenme_qiymet)

tahmin_qiymetler = masin_modeli.predict(yoxlama_giris)

deqiqlik = r2_score(yoxlama_qiymet, tahmin_qiymetler)
print(f'Modelin Deqiqlik Faiz Gostericisi (R2): {deqiqlik:.2f}')

plt.figure(figsize=(8, 6))
plt.scatter(yoxlama_qiymet, tahmin_qiymetler, color='blue', alpha=0.6)
plt.xlabel('Heqiqi Qiymetler')
plt.ylabel('Modelin Proqnozlasdirdigi Qiymetler')
plt.title('Avtomobil Qiymeti Tahmini: Heqiqi vs Proqnoz')
plt.plot(
    [yoxlama_qiymet.min(), yoxlama_qiymet.max()],
    [yoxlama_qiymet.min(), yoxlama_qiymet.max()],
    'r--',
    lw=2,
)
plt.show()