# -*- coding: utf-8 -*-
"""Program_Proyek1_ML_Terapan_Predictive_Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1g91ctqmd6sidXO3SD19dJELuN_5o3bY6

**Dicoding Proyek Pertama ML Terapan: Predictive Analysis**

Notebook kali ini adalah program dan dokumentasi untuk proyek pertama untuk kelas Machine Learning Terapan. Modul ini akan berisi
1. Domain Proyek: Berisi Latar Belakang dan alasan masalah tersebut harus diselesaikan.
2. Business Understanding: berisi Problem Statement, Goals, dan Solution Statement.
3. Data Understanding: berisi tentang penjelasan informasi dari data yang digunakan.
4. Data Preparation: berisi tentang teknik persiapan data sebelum modeling.
5. Modeling: berisi tentan model ML yang digunakan untuk memecahkan masalah.
6. Evaluasi: berisi tentang penjelasan metriks evaluasi yang digunakan.

**1. Domain Proyek**

**- Tema : Kesehatan**

**- Judul: Prediksi Penyakit Stroke Menggunakan KNN, Random Forest, Logistic Regression**

**- Latar Belakang**

Stroke adalah salah satu penyakit tidak menular yang paling banyak dialami oleh manusia. Menurut World Health Organization, pada tahun 2017 stroke menjadi penyebab kematian ketiga di dunia. Di Indonesia, stroke menjadi penyebab pertama kematian di rumah sakit. Sayangnya, penyakit ini kebanyakan diketahui setelah penderita menderita stroke tingkat akhir yang memiliki kemungkinan untuk sembuh sangat kecil.

Meskipun menjadi suatu ancaman penyakit yang banyak menjangkit manusia, penelitian mengenai prediksi penyakit ini masih sedikit di Indonesia.  Berdasarkan data di atas, maka diperlukan suatu sistem yang dapat memprediksi kemungkinan penyakit stroke berdasarkan gejala-gejala yang dirasakan oleh pasien. Dengan adanya solusi ini, diharapkan dapat membantu dokter untuk mendiagnosis penyakit lebih awal dan memberikan terapi serta edukasi tentang penyakit stroke.

**- Alasan Penyelesaian Masalah**

Penyakit stroke adalah salah satu penyakit mematikan yang sering menjangkit manusia. Kebanyakan kasus stroke diketahui setelah pasien menderita stroke tahap akhir. Padahal, penyakit ini memerlukan deteksi awal dan penanganan yang cepat agar stroke tidak memengaruhi bagian syaraf dan tubuh yang lain. Oleh karena itu, prediksi awal sangat menentukan dalam penanganan penyakit ini serta meningkatkan kemungkinan hidup dari penderita.

**- Referensi**

Referensi dari permasalahan ini berasal dari jurnal "Prediksi Risiko Kematian Pasien Stroke Perdarahan dengan Menggunakan Teknik Klasifikasi Data Mining" oleh Indarto, Ema Utami, dan Suwanto Raharjo. Jurnal dapat diakses pada : http://www.e-journal.janabadra.ac.id/index.php/informasiinteraktif/article/view/1172/790

**2. Business Understanding**

**Problem Statement**

1. Penyakit stroke adalah penyakit yang sering diderita oleh manusia namun masih jarang terdeteksi saat gejala awal sehingga dapat menyebabkan komplikasi lebih lanjut.

**Goals**

1. Diperlukan suatu sistem yang dapat memprediksi apakah seseorang menderita penyakit stroke berdasarkan gejala ataupun situasi yang dialami oleh penderita agar dapat dilakukan penanganan secepat mungkin.


**Solution Statement**
1. Untuk membuat sistem prediksi kali ini, saya menggunakan 3 algoritma sebagai perbandingan yaitu Logistic Regression, KNN, dan Random Forest. Untuk menentukan algoritma yang lebih baik, saya menggunakan metriks evaluasi yaitu MSE(Mean Squared Error). Saya memilih MSE karena poin penting dari prediksi penyakit stroke adalah mengurangi kemungkinan salah diagnosis atau mengurangi False Negative dan False Positive.

**3. Data Understanding**

3.1. Loading Data

3.2. EDA-Deskripsi Variabel

3.3. EDA-Menangani Missing Value dan Outliers

3.4. EDA-Univariate Analysis

3.5. EDA-Multivariate Analysis

**3.1. Loading Data**

**- Import package yang diperlukan**
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

"""**- Membaca data CSV dan mengubahnya menjadi DataFrame**"""

# Mengambil data dari dataset
url = 'https://raw.githubusercontent.com/jonywony/Dicoding-ML-Terapan-Submission1/main/Dataset/healthcare-dataset-stroke-data.csv'
df = pd.read_csv(url)
df

#Menghapus kolom id karena tidak akan masuk ke dalam proses klasifikasi
df.drop('id', axis=1, inplace=True)
df

"""**Penjelasan**

Dari dataset yang saya ambil dari Kaggle dengan judul "Stroke Prediction Dataset" (https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) dapat dilihat bahwa terdapat 5110 data dan 12 kolom yang diguanakan yaitu:
1. **id**: merupakan identifikasi unik untuk setiap pasien (numerik)
2. **gender**: merupakan data jenis kelamin pasien apakah laki-laki('Male'), perempuan('Female'), atau lainnya ('Other') (kategorik)
3. **age**: umur dari pasien (numerik)
4. **hypertension**: merupakan status apakah pasien memiliki hipertensi(1) atau tidak(0) (kategorik)
5. **heart_disease**: merupakan status apakah pasien memiliki penyakit jantung(1) atau tidak(0) (kategorik)
6. **ever_married**: merupakan status pernikahan apakah sudah pernah menikah('Yes') atau belum('no') (kategorik)
7. **work_type**: merupakan status pekerjaan apakah anak-anak('children'), pegawai pemerintahan('Govt_jov'), tidak pernah bekerja('Never_worked'), bekerja di perusahaan('private'), atau self-employed('self-employed') (kategorik)
8. **Residence_type**: merupakan status dimana pasien tinggal apakah di desa('Rural') atau kota('Urban') (kategorik)
9. **avg_glucose_level**: merupakan rata-rata level glukosa dalam darah (numerik)
10. **bmi**: merupakan indeks massa tubuh (numerik)
11. **smoking_status**:merupakan status perokok pasien apakah berhenti merokok('formerly_smoked'), tidak merokok('never_smoked'), merokok('smoked'), atau tidak tahu('Unknown')
12. **stroke**: merupakan status apakah seseorang terkena stroke(1) atau tidak(0) (kategorik)

***note**: kolom id dihilangkan karena tidak akan masuk ke dalam proses klasifikasi nanti

**3.2. EDA-Deskripsi Variabel**
"""

#Mengecek informasi tentang tipe data yang terdapat pada setiap kolom di dataset
df.info()

#Mengecek ringkasan statistik dataset
df.describe()

#Mengubah tipe data hypertension, heart_disease, dan stroke menjadi object
df['hypertension'] = df['hypertension'].astype(dtype='object')
df['heart_disease'] = df['heart_disease'].astype(dtype='object')
df['stroke'] = df['stroke'].astype(dtype='object')
df.describe()

"""**Penjelasan**

Eksplorasi variabel data menggunakan fungsi **info()** dari dataframe akan mengeluarkan informasi tentang banyaknya data yang tidak null dan tipe data yang ada pada data, yaitu:
1. **gender**: memiliki tipe data object
2. **age**: memiliki tipe data float
3. **hypertension**: memiliki tipe data int
4. **heart_disease**: memiliki tipe data int
5. **ever_married**: memiliki tipe data object
6. **work_type**: memiliki tipe data object
7. **Residence_type**: memiliki tipe data object
8. **avg_glucose_level**: memiliki tipe data float
9. **bmi**: memiliki tipe data int
10. **smoking_status**:memiliki tipe data object
11. **stroke**: memiliki tipe data int

Pada bagian Loading Data dijelaskan bahwa atribut hypertension, heart_disease, dan stroke adalah data kategorik namun pada dataframe memiliki nilai int. Oleh karena itu, pertama-tama kita akan mengubah tipe data ketiganya menjadi object dengan menggunakan fungsi **astype(dtype='object')**. Hasil dari transformasi ini adalah:
1. **gender**: memiliki tipe data object
2. **age**: memiliki tipe data float
3. **hypertension**: memiliki tipe data **object**
4. **heart_disease**: memiliki tipe data **object**
5. **ever_married**: memiliki tipe data object
6. **work_type**: memiliki tipe data object
7. **Residence_type**: memiliki tipe data object
8. **avg_glucose_level**: memiliki tipe data float
9. **bmi**: memiliki tipe data int
10. **smoking_status**:memiliki tipe data object
11. **stroke**: memiliki tipe data **object**

***note**: dari sini juga dapat dilihat bahwa atribut bmi memiliki missing value, hal ini akan ditangani pada proses selanjutnya

**3.3. EDA-Penanganan Missing Value dan outliers**

**Penanganan Missing Value**
"""

#Cek apakah ada missing Value atau tidak
df.isna().sum().sort_values(ascending=False)[:10]

#Menangani Missing Value pada kolom bmi dengan menggantinya dengan median untuk mempertahankan distribusi data
df['bmi'].fillna(value=df['bmi'].median() ,inplace=True)

#Cek apakah ada missing Value atau tidak
df.isna().sum().sort_values(ascending=False)[:10]

#Cek kembali deskripsi statistik data
df.describe()

#Cek bentuk data
df.shape

"""**Penanganan Outlier**"""

#Cek apakah ada outlier pada kolom age
sns.boxplot(x=df['age'])

#Cek apakah ada outlier pada kolom age
sns.boxplot(x=df['avg_glucose_level'])

#Cek apakah ada outlier pada kolom age
sns.boxplot(x=df['bmi'])

#Menghitung Interuartile Range sebagai acuan untuk menghapus outlier
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR=Q3-Q1

#Melakukan subset dan menghapus data outlier
df=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]

#Cek ukuran dataset setelah proses drop outliers
df.shape

"""**Penjelasan**


**- Penanganan Missing Value**
Untuk menangani Missing Value, sebelumnya kita akan mengidentifikasi apakah ada missing value pada data menggunakan fungsi **isna().sum().sort_values(ascending=False)**. Dapat dilihat bahwa terdapat 201 missing value pada atribut bmi. Setelah itu, saya menggunakan fungsi **fillna()** dengan value median dari bmi. Saya memutuskan untuk mengisi data yang kosong tersebut dengan mengunakan median Karena memiliki jumlah yang sangat banyak dan untuk mempertahankan distribusi data.



**- Penanganan Outlier**
Dengan menggunakan boxplot pada data numerik ('bmi', 'avg_glucose_level', dan 'age') dapat dilihat bahwa pada atribut **bmi** dan **avg_glucose_level** memiliki outlier yang ditandai dengan titik-titik hitam diluar boxplot. Untuk menanganinya, saya menggunakan IQR(Inter Quartile Range) sebagai parameter untuk menghapus outlier, jika ada data yang nilainya kurang dari Q1-1.5XIQR atau data yang nilainya lebih dari Q3+1.5XIQR maka data tersebut akan dianggap outlier dan dihilangkan.

Setelah melalui dua proses tersebut data yang telah dibersihkan menjadi 4391 data.

**3.4. EDA-Univariate Analysis**

**- Data Kategorik**
"""

#Membagi kolom pada dataframe menjadi kolom kategorik dan numerik
numerical_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']

feature = categorical_features[0]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df1 = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df1)
count.plot(kind='bar', title=feature);

#Menghilangkan nilai other dari dataset
df.drop(df[df['gender']=='Other'].index,inplace=True)

feature = categorical_features[1]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df1 = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df1)
count.plot(kind='bar', title=feature);

feature = categorical_features[2]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df1 = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df1)
count.plot(kind='bar', title=feature);

feature = categorical_features[3]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df1 = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df1)
count.plot(kind='bar', title=feature);

feature = categorical_features[4]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df1 = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df1)
count.plot(kind='bar', title=feature);

feature = categorical_features[5]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df1 = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df1)
count.plot(kind='bar', title=feature);

feature = categorical_features[6]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df1 = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df1)
count.plot(kind='bar', title=feature);

feature = categorical_features[7]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df1 = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df1)
count.plot(kind='bar', title=feature);

"""**- Data Numerik**"""

#Melihat persebaran data numerik
df[numerical_features].hist(bins=50, figsize=(20,15))
plt.show()

"""**Penjelasan**

**-Data Kategorik**

Pada data kategorik yang terdiri dari atribut ('gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke') akan dilakukan univariate analysis dengan mengetaui sebarannya. Didapatkan hasil sebagai berikut:
1. **gender**: memiliki sebaran yang hampir merata antara kategori "Male" dan "Female". Kategori "Other" akan dihilangkan karena hanya memiliki 1 data saja.
2. **hypertension**: memiliki sebaran yang tidak merata antara pasien yang memiliki hipertensi(1) dengan yang tidak(0). Pasien yang tidak memiliki hipertensi jauh lebih banyak.
3. **heart_disease**: memiliki sebaran yang tidak merata antara pasien yang memiliki penyakit jantung(1) dengan yang tidak(0). Pasien yang tidak memiliki penyakit jantung jauh lebih banyak.
4. **ever_married**: memiliki sebaran yang hampir merata antara kategori "Yes" dan "No".
5. **work_type**: memiliki sebaran yang tidak rata antara setiap data kategorik dimana kebanyakan data ada pada satu kategori yaitu "private" sementara kategori "never_worked" sangat sedikit.
6. **Residence_type**: memiliki sebaran yang hampir merata antara kategori "Urban" dan "Rural".
7. **smoking_status**:memiliki sebaran yang hampir merata antara setiap kategori.
8. **stroke**: memiliki sebaran yang tidak merata dimana kebanyakan data adalah pasien yang tidak mengalami stroke(0).

**- Data Numerik**

1. **age**: memiliki sebaran data yang random dimana hampir semua distribusi memiliki nilai yang hampir sama.
2. **avg_glucose_level**: memiliki sebaran yang condong/skewed ke kiri.
3. **bmi**: memiliki sebaran yang mendekati normal.

**3.5. EDA-Multivariate Analysis**

**- Fitur Kategorik**
"""

cat_features = df.select_dtypes(include=['object', 'int64']).columns.to_list()

for col in cat_features:
  sns.catplot(x=col, y="stroke", kind="bar", dodge=False, height = 4, aspect = 3,  data=df, palette="Set3")
  plt.title("Rata-rata 'stroke' Relatif terhadap - {}".format(col))

"""
**- Fitur Numerik**"""

sns.pairplot(df[numerical_features], diag_kind = 'kde')

plt.figure(figsize=(10, 8))

correlation_matrix = df.corr().round(2)
 
# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

#Kolom avg_glucose_level di drop karena memiliki korelasi yang sangat rendah(-0.02)
df.drop(['avg_glucose_level'], inplace=True, axis=1)
df.head()

"""**Penjelasan**

Multi Variate Analysis digunakan untuk mengetahui hubungan antara 2 atribut atau lebih.

**- Fitur Kategorik**
Saya akan mengecek rata-rata terkena penyakit stroke terhadap masing-masing fitur kategorik. Didapatkan hasil sebagai berikut:
1. **gender**: Pada atribut gender, rata-rata nilai pada kategori "Male' dan "Female" hampir mirip pada rentang 0.3-0.4.
2. **hypertension**: Pada atribut hypertension, apabila memiliki hipertensi(1) maka akan semakin tinggi rata-rata.
3. **heart_disease**: Pada atribut heart_disease, apabila memiliki penyakit jantung(1) maka akan semakin tinggi rata-rata.
4. **ever_married**: Pada atribut ever_married, apabila sudah pernah menikah, maka akan semakin tinggi rata-rata.
5. **work_type**: Pada atribut work_type, rata-rata nilai pada setiap kategori hampir sama kecuali untuk kategori "Never_worked" dan "children" yang mendekati 0. apabila memiliki kategori self_employed maka akan semakin tinggi rata-rata.
6. **Residence_type**: Pada atribut Residence_type, rata-rata nilai pada kategori "Rural' dan "Urban" hampir mirip.
7. **smoking_status**: Pada atribut gender, rata-rata nilai pada setiap kategori hampir sama. Apabila memiliki kategori "formerly_smoked" maka rata-rata akan semakin tinggi.

**- Fitur Numerik**

Pada data numerik, akan dihitung korelasi antara setiap bagian fitur. Didapatkan hasil:
1. Atribut age dan bmi memiliki korelas yang bagus sehingga nilai tersebut akan tetap dipertahankan.
2. Atribut avg_glucose_level memiliki corelasi yang buruk yaitu -0.02, karena itu atribut avg_glucose_level akan dihapus.

**4. Data Preparation**

**- Melakukan Encoding Fitur Kategorik**
"""

#Mengubah kolom gender menjadi int
df['gender'].replace({'Male':1,'Female':0},inplace = True)

#Mengubah kolom ever_married menjadi int biner(0 atau 1)
df['ever_married'].replace({'Yes':1,'No':0},inplace = True)

#Melakukan one-hot encoding pada kolom work_type karena memiliki lebih dari 2 kategori (multi category)
df = pd.get_dummies(df, columns = ['work_type'])

#mengubah kolom Residence_type menjadi int
df['Residence_type'].replace({'Urban':1,'Rural':0},inplace = True)

#Melakukan proses one-hot encoding pada kolom smoking_status
df = pd.get_dummies(df, columns = ['smoking_status'])

#Mengubah tipe data atribut hypertension, heart_disease, dan stroke menjadi int
df['hypertension'].replace({'1':1,'0':0},inplace = True)
df['heart_disease'].replace({'1':1,'0':0},inplace = True)
df['stroke'].replace({'1':1,'0':0},inplace = True)

#Mengecek bentuk dataframe setelah preprocess dan memastikan semua data adalah numerik
df.shape
df.info()

"""**- Train-Test Split**"""

#Membagi dataset menjadi data training dan data latih
from sklearn.model_selection import train_test_split
 
X = df.drop(["stroke"],axis =1)
y = df["stroke"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""**- Standarisasi**"""

#Melakukan normalisasi pada data numerik
from sklearn.preprocessing import StandardScaler
 
numerical_features = ['age', 'bmi']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

X_train[numerical_features].describe().round(4)

"""**Penjelasan**

**- Encoding Fitur Kategorik**

- Proses

Pada dataset kali ini, akan dilakukan proses encoding data kategorik menjadi numerik yaitu one-hot encoding. One-hot encoding adalah proses merepresentasikan data kategorik sebagai vektor biner yang bernilai integer 0 atau 1. 

Apabila data kategorik adalah biner(terdiri dari 2 kategori) maka atribut akan tetap dan salah satu kategori merepresentasikan nilai 1 dan kategori lainnya merepresentasikan nilai 0. Dalam dataset ini, atribut yang memiliki kategori biner adalah atribut **gender, ever_married, Residence_type, hypertension, heart_disease, dan stroke**

Apabila data kategorik memiliki lebih dari 2 kategori, maka akan dibuat kolom baru sesuai dengan banyaknya kategori yang ada. pada atribut tersebut nilai 1 merepresentasikan bahwa data termasuk kategori tersebut sementara nilai 0 merepresentasikan bahwa data bukanlah termasuk kategori tersebut. Dala dataset ini, atribut yang memiliki kategori lebih dari 2 adalah **work_type dan smoking_status**

- Alasan Penggunaan

Alasan penggunaan one-hot encoding adalah untuk mengubah data kategorik menjadi data numerik namun masih tetap merepresentasikan data kategorik tersebut. Hal ini dilakukan karena akan membantu model untuk memahami data menjadi lebih baik dimana mesin hanya mengerti angka dan bukan teks/huruf. Selain itu, one-hot encoding akan memudahkan penentuan probabilitas untuk setiap value.

**- Membagi Data Train dan Test**
- Proses

Sebelum data digunakan untuk membuat model, data harus terlebih dulu dibagi menjadi data latih dan data uji. Saya menggunakan rasio 80:20 untuk data latih dan data uji. pembagian data latih dan data uji menggunakan fungsi **train_test_split(X, y, test_size = 0.2, random_state = 123)** dimana X dan y adalah data yang akan di pisah, test_size adalah ukuran data latih, dan random_state adalah inisiasi internal random generator yang akan menentukan indeks dalam pembagian dataset.
- Alasan Penggunaan

Pembagian dataset menjadi data latih dan data uji bertujuan untuk memudahkan mengevaluasi seberapa baik generalisasi model terhadap data baru. Selain itu, hal ini dilakukan untuk tidak mengotori data uji dengan informasi yang kita dapat dari data latih.

**- Standarisasi**

- Proses

Standarisasi adalah proses yang dilakukan pada data numerik untuk mengurangkan mean kemudian membaginya dengan standar deviasi untuk menggeser distribusi. Pada data kali ini, saya menggunakan StandardScaler untuk melakukan standarisasi. StandardScaler akan menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0.

- Alasan Penggunaan

Standarisasi digunakan untuk membuat model machine learning memiliki performa lebih baik dan konvergen karena dimodelkan dengan data yang mendekati distribusi normal atau data dengan skala relatif.

**5. Pembuatan Model**
"""

#Membuat model untuk prediksi
# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN', 'RandomForest', 'logistic_Regression'])

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
 
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
 
models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

# Impor library yang dibutuhkan
from sklearn.ensemble import RandomForestRegressor
 
# buat model prediksi
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
 
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

# Impor library yang dibutuhkan
from sklearn.linear_model import LogisticRegression
 
# buat model prediksi
logisticRegr = LogisticRegression(C=0.1, random_state=0)
logisticRegr.fit(X_train, y_train)
 
models.loc['train_mse','logistic_Regression'] = mean_squared_error(y_pred=logisticRegr.predict(X_train), y_true=y_train)

"""**Penjelasan**

Modeling adalah proses untuk membuat model yang dapat melakukan tugas sesuai dengan keinginan kita. Pada proyek kali ini, saya menggunakan 3 algoritma sebagai perbandingan yaitu KNN, Random Forest, dan Logistic Regression. 
- Algoritma KNN

Pada implementasi algoritma KNN saya menggunakan fungsi **KNeighborsRegressor** dari **sklearn.neighbors** dengan menggunakan parameter n_neighbors sama dengan 10 yang menunjukan banyaknya tetangga terdekat yang ikut dalam penentuan kelas. Secara default, metriks penentuan jaraknya adalah euclidean.

**- Kelebihan KNN**
1. Tidak memiliki periode training, sehingga algoritma ini sangat efisien dalam hal waktu dan kompleksitas.
2. Implementasi yang mudah, KNN mudah untuk diimplementasikan karena hanya mengukur jarak antar data.
3. Karena tidak memiliki periode training, data baru bisa ditambahkan kapan saja.

**- Kekurangan KNN**
1. Tidak bekerja dengan baik saat menggunakan dataset yang besar karena sangat boros dalam hal komputasi.
2. Tidak bekerja dengan baik pad adataset dengan dimensi tinggi karena menyusahkan dalam perhitungan jarak antar data.
3. Sensitif terhadap outlier dan missing data.
4. Seluruh data harus dilakukan scaling atau normalisasi dan standarisasi.

- Algoritma Random Forest

Pada implementasi Random Forest saya menggunakan fungsi **RandomForestRegressor** dari package **sklearn.ensemble**. Dengan menggunakan parameter n_estimators sama dengan 50 yang menunjukan jumlah tree di forest, max_depth sama dengan 16 yang menunjukan kedalaman atau panjang pohon, random_state sama dengan 55 yang digunakan untuk mengontrol random number generator, dan n_jobs sama dengan -1 yang berarti semua proses dilaksanakan secara paralel.

**- Kelebihan Random Forest**
1. Dapat mengurangi variasi data sehingga dapat memperbaiki akurasi
2. Dapat menyelesaikan permasalahan data kategorik dan numerik sekaligus.
3. Dapat menangani missing values.
4. Tidak sensitif terhadap outlier dan dapat menanganinya secara otomatis.

**- Kekurangan Random Forest**
1. Kompleksitas yang tinggi karena menggunakan banyak tree dan menggabungkan berbagai output dari tree tersebut.
2. Memiliki periode training yang lebih lama karena akan menghasilkan banyak tree dan membuat keputusan berdasarkan mayoritas dari tree.

- Algoritma Logistic Regression

Pada implementasi Logistic Regression saya menggunakan fungsi **LogisticRegression** dari package **sklearn.linear_model**. Dengan menggunakan parameter C sama dengan 0.1 yang menunjukan regularization yang kuat karena nilainya yang semakin kecil (regularisasi digunakan untuk menghindari overfitting), random_state sama dengan 123 yang digunakan untuk mengontrol random number generator, dans ecara default menetapkan penalti dengan aturan L2 sebagai cara untuk belajar.

**- Kelebihan Logistic Regression**
1. Mudah untuk diimplementasikan, dijabarkan, dan efisien saat pelatihan data.
2. Dapat bekerja dengan baik saat dataset dapat dipisahkan secara linear.
3. Lebih tahan terhadap overfitting karena menerapkan Regularization.
4. Tidak hanya menunjukan relevansi predictor(coefficient size) tetapi juga menunjukan arah asosiasi(negatif atau positif).
**- Kekurangan Logistic Regression**
1. Terbatas pada data yang dapat dipisahkan secara linear.
2. Hanya dapat digunakan untuk memprediksi data yang bersifat diskrit dan memiliki permasalahan dalam menganalisa data yang kontinu.

**Algoritma Terbaik**

Berdasarkan kekurangan dan kelebihan yang dijabarkan, menurut saya algoritma yang mungkin bekerja dengan sangat baik pada data kali ini adalah algoritma KNN. Hal ini disebabkan karena dataset yang memiliki dimensionality rendah, jumlah data yang cenderung kecil, dan telah dilakukan proses untuk menangani missing value serta outlier. Selain itu, dilihat dari metriks MSE pada data latih dan data uji KNN adalah model yang menunjukan error yang kecil pada keduanya dan tidak terindikasi overfitting.

**6. Evaluasi**

**Penjelasan**

Metriks evaluasi yang saya gunakan pada proyek kali ini adalah MSE (Mean Squared Error). Sesuai dengan konteks data, problem statement, dan solusi yang diinginkan yaitu lebih menekankan untuk mengurangi kesalahan prediksi(error). MSE bekerja dengan cara mengurangi nilai aktual dengan nilai prediksi dan hasilnya dikuadratkan kemudian dijumlahkan secara keseluruhan dan membaginya dengan banyaknya data yang ada. Secara umum, formula untuk MSE adalah sebagai berikut:
![MSE_Formula.png](data:image/png;base64,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)

Hasil yang didapatkan pada MSE data latih dan data uji adalah sebagai berikut:
- Algoritma KNN

**- MSE Data latih:** 0.000031

**- MSE Data uji:** 0.000032

**- Analisis Hasil:** Dapat dilihat bahwa MSE yang didapatkan sangat kecil pada kedua data latih dan data uji, selain itu tidak terindikasi adanya overfitting pada data latih.


- Algoritma Random Forest

**- MSE Data latih:** 0.000006

**- MSE Data uji:** 0.000035

**- Analisis Hasil:** Dapat dilihat bahwa MSE yang didapatkan sangat kecil pada kedua data latih dan data uji, namun terdapat kemungkinan adanya overfitting pada data latih karena MSE pada data latih jauh lebih kecil dibandingkan MSE pada data uji.


- Algoritma Logistic Regression

**- MSE Data latih:** 0.000039

**- MSE Data uji:** 0.000033

**- Analisis Hasil:** Dapat dilihat bahwa MSE yang didapatkan sangat kecil pada kedua data latih dan data uji, namun terdapat kemungkinan adanya overfitting pada data latih karena MSE pada data latih sedikit lebih kecil dibandingkan MSE pada data uji.



**Kesimpulan:** Pada saat percobaan prediksi pada dataset, terlihat bahwa ketiganya sama-sama menunjukan hasil yang benar. Namun berdasarkan evaluasi mertriks MSE, didapatkan hasil bahwa algoritma terbaik pada model kali ini adalah KNN.
"""

# Lakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','logisticRegr'])
acc = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','logisticRegr'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'logisticRegr': logisticRegr}
 
# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
 
# Panggil mse
mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(0)
 
pd.DataFrame(pred_dict)
