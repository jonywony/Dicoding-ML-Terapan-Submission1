# Dicoding Proyek Pertama ML Terapan: Predictive Analysis

Notebook kali ini adalah program dan dokumentasi untuk proyek pertama untuk kelas Machine Learning Terapan. Modul ini akan berisi
1. Domain Proyek: Berisi Latar Belakang dan alasan masalah tersebut harus diselesaikan.
2. Business Understanding: berisi Problem Statement, Goals, dan Solution Statement.
3. Data Understanding: berisi tentang penjelasan informasi dari data yang digunakan.
4. Data Preparation: berisi tentang teknik persiapan data sebelum modeling.
5. Modeling: berisi tentan model ML yang digunakan untuk memecahkan masalah.
6. Evaluasi: berisi tentang penjelasan metriks evaluasi yang digunakan.

 ## **1. Domain Proyek**

##### **- Tema : Kesehatan**

##### **- Judul: Prediksi Penyakit Stroke Menggunakan KNN, Random Forest, Logistic Regression**

##### **- Latar Belakang**
Stroke adalah salah satu penyakit tidak menular yang paling banyak dialami oleh manusia. Menurut World Health Organization, pada tahun 2017 stroke menjadi penyebab kematian ketiga di dunia. Di Indonesia, stroke menjadi penyebab pertama kematian di rumah sakit. Sayangnya, penyakit ini kebanyakan diketahui setelah penderita menderita stroke tingkat akhir yang memiliki kemungkinan untuk sembuh sangat kecil.
Meskipun menjadi suatu ancaman penyakit yang banyak menjangkit manusia, penelitian mengenai prediksi penyakit ini masih sedikit di Indonesia.  Berdasarkan data di atas, maka diperlukan suatu sistem yang dapat memprediksi kemungkinan penyakit stroke berdasarkan gejala-gejala yang dirasakan oleh pasien. Dengan adanya solusi ini, diharapkan dapat membantu dokter untuk mendiagnosis penyakit lebih awal dan memberikan terapi serta edukasi tentang penyakit stroke.

##### **- Alasan Penyelesaian Masalah**
Penyakit stroke adalah salah satu penyakit mematikan yang sering menjangkit manusia. Kebanyakan kasus stroke diketahui setelah pasien menderita stroke tahap akhir. Padahal, penyakit ini memerlukan deteksi awal dan penanganan yang cepat agar stroke tidak memengaruhi bagian syaraf dan tubuh yang lain. Oleh karena itu, prediksi awal sangat menentukan dalam penanganan penyakit ini serta meningkatkan kemungkinan hidup dari penderita.

##### **- Referensi**

Referensi dari permasalahan ini berasal dari jurnal "Prediksi Risiko Kematian Pasien Stroke Perdarahan dengan Menggunakan Teknik Klasifikasi Data Mining" oleh Indarto, Ema Utami, dan Suwanto Raharjo. Jurnal dapat diakses pada : http://www.e-journal.janabadra.ac.id/index.php/informasiinteraktif/article/view/1172/790

## **2. Business Understanding**

##### **Problem Statement**
1. Penyakit stroke adalah penyakit yang sering diderita oleh manusia namun masih jarang terdeteksi saat gejala awal sehingga dapat menyebabkan komplikasi lebih lanjut.
##### **Goals**
1. Diperlukan suatu sistem yang dapat memprediksi apakah seseorang menderita penyakit stroke berdasarkan gejala ataupun situasi yang dialami oleh penderita agar dapat dilakukan penanganan secepat mungkin.
##### **Solution Statement**
1. Untuk membuat sistem prediksi kali ini, saya menggunakan 3 algoritma sebagai perbandingan yaitu Logistic Regression, KNN, dan Random Forest. Untuk menentukan algoritma yang lebih baik, saya menggunakan metriks evaluasi yaitu MSE(Mean Squared Error). Saya memilih MSE karena poin penting dari prediksi penyakit stroke adalah mengurangi kemungkinan salah diagnosis atau mengurangi False Negative dan False Positive.

## **3. Data Understanding**
#### **3.1. Loading Data**
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

***note**: kolom id dihilangkan karena tidak akan masuk ke dalam proses klasifikasi nanti.

#### **3.2. EDA-Deskripsi Variabel**
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

#### **3.3. EDA-Menangani Missing Value dan Outliers**
##### **- Penanganan Missing Value**
Untuk menangani Missing Value, sebelumnya kita akan mengidentifikasi apakah ada missing value pada data menggunakan fungsi **isna().sum().sort_values(ascending=False)**. Dapat dilihat bahwa terdapat 201 missing value pada atribut bmi. Setelah itu, saya menggunakan fungsi **fillna()** dengan value median dari bmi. Saya memutuskan untuk mengisi data yang kosong tersebut dengan mengunakan median Karena memiliki jumlah yang sangat banyak dan untuk mempertahankan distribusi data.

##### **- Penanganan Outlier**
[![Box_Age](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/dist_age.png?raw=true)
[![Box_BMI](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/dist_bmi.png?raw=true)
[![Box_Avg_glucose_level](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/dist_avg_glucose_level.png?raw=true)
Dengan menggunakan boxplot pada data numerik ('bmi', 'avg_glucose_level', dan 'age') dapat dilihat bahwa pada atribut **bmi** dan **avg_glucose_level** memiliki outlier yang ditandai dengan titik-titik hitam diluar boxplot. Untuk menanganinya, saya menggunakan IQR(Inter Quartile Range) sebagai parameter untuk menghapus outlier, jika ada data yang nilainya kurang dari Q1-1.5XIQR atau data yang nilainya lebih dari Q3+1.5XIQR maka data tersebut akan dianggap outlier dan dihilangkan.
Setelah melalui dua proses tersebut data yang telah dibersihkan menjadi 4391 data.

#### **3.4. EDA-Univariate Analysis**
##### **-Data Kategorik**
Pada data kategorik yang terdiri dari atribut ('gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke') akan dilakukan univariate analysis dengan mengetaui sebarannya. Didapatkan hasil sebagai berikut:
[![Box_gender](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/dist_gender.png?raw=true)
1. **gender**: memiliki sebaran yang hampir merata antara kategori "Male" dan "Female". Kategori "Other" akan dihilangkan karena hanya memiliki 1 data saja.
[![Box_hypertension](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/dist_hypertension.png?raw=true)
2. **hypertension**: memiliki sebaran yang tidak merata antara pasien yang memiliki hipertensi(1) dengan yang tidak(0). Pasien yang tidak memiliki hipertensi jauh lebih banyak.
[![Box_HD](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/dist_heart_disease.png?raw=true)
3. **heart_disease**: memiliki sebaran yang tidak merata antara pasien yang memiliki penyakit jantung(1) dengan yang tidak(0). Pasien yang tidak memiliki penyakit jantung jauh lebih banyak.
[![Box_EM](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/dist_ever_married.png?raw=true)
4. **ever_married**: memiliki sebaran yang hampir merata antara kategori "Yes" dan "No".
[![Box_WT](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/dist_work_type.png?raw=true)
5. **work_type**: memiliki sebaran yang tidak rata antara setiap data kategorik dimana kebanyakan data ada pada satu kategori yaitu "private" sementara kategori "never_worked" sangat sedikit.
[![Box_RT](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/dist_Residence_type.png?raw=true)
6. **Residence_type**: memiliki sebaran yang hampir merata antara kategori "Urban" dan "Rural".
[![Box_Smoking-Status](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/dist_smoking_status.png?raw=true)
7. **smoking_status**:memiliki sebaran yang hampir merata antara setiap kategori.
[![Box_stroke](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/dist_stroke.png?raw=true)
8. **stroke**: memiliki sebaran yang tidak merata dimana kebanyakan data adalah pasien yang tidak mengalami stroke(0).

##### **- Data Numerik**
[![Box_NUM](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/dist_numerik.png?raw=true)
1. **age**: memiliki sebaran data yang random dimana hampir semua distribusi memiliki nilai yang hampir sama.
2. **avg_glucose_level**: memiliki sebaran yang condong/skewed ke kiri.
3. **bmi**: memiliki sebaran yang mendekati normal.

#### **3.5. EDA-Multivariate Analysis**
Multi Variate Analysis digunakan untuk mengetahui hubungan antara 2 atribut atau lebih.
##### **- Fitur Kategorik**
[![Box_CorrKategorik](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/corr_kategorik.png?raw=true)
Saya akan mengecek rata-rata terkena penyakit stroke terhadap masing-masing fitur kategorik. Didapatkan hasil sebagai berikut:
1. **gender**: Pada atribut gender, rata-rata nilai pada kategori "Male' dan "Female" hampir mirip pada rentang 0.3-0.4.
2. **hypertension**: Pada atribut hypertension, apabila memiliki hipertensi(1) maka akan semakin tinggi rata-rata.
3. **heart_disease**: Pada atribut heart_disease, apabila memiliki penyakit jantung(1) maka akan semakin tinggi rata-rata.
4. **ever_married**: Pada atribut ever_married, apabila sudah pernah menikah, maka akan semakin tinggi rata-rata.
5. **work_type**: Pada atribut work_type, rata-rata nilai pada setiap kategori hampir sama kecuali untuk kategori "Never_worked" dan "children" yang mendekati 0. apabila memiliki kategori self_employed maka akan semakin tinggi rata-rata.
6. **Residence_type**: Pada atribut Residence_type, rata-rata nilai pada kategori "Rural' dan "Urban" hampir mirip.
7. **smoking_status**: Pada atribut gender, rata-rata nilai pada setiap kategori hampir sama. Apabila memiliki kategori "formerly_smoked" maka rata-rata akan semakin tinggi.

##### **- Fitur Numerik**
[![Box_KorrNum](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/matriks_korelasi_numerik.png?raw=true)
Pada data numerik, akan dihitung korelasi antara setiap bagian fitur. Didapatkan hasil:
1. Atribut age dan bmi memiliki korelas yang bagus sehingga nilai tersebut akan tetap dipertahankan.
2. Atribut avg_glucose_level memiliki corelasi yang buruk yaitu -0.02, karena itu atribut avg_glucose_level akan dihapus.
 
## **4. Data Preparation**
#### **- Encoding Fitur Kategorik**
- Proses

Pada dataset kali ini, akan dilakukan proses encoding data kategorik menjadi numerik yaitu one-hot encoding. One-hot encoding adalah proses merepresentasikan data kategorik sebagai vektor biner yang bernilai integer 0 atau 1. 
Apabila data kategorik adalah biner(terdiri dari 2 kategori) maka atribut akan tetap dan salah satu kategori merepresentasikan nilai 1 dan kategori lainnya merepresentasikan nilai 0. Dalam dataset ini, atribut yang memiliki kategori biner adalah atribut **gender, ever_married, Residence_type, hypertension, heart_disease, dan stroke**
Apabila data kategorik memiliki lebih dari 2 kategori, maka akan dibuat kolom baru sesuai dengan banyaknya kategori yang ada. pada atribut tersebut nilai 1 merepresentasikan bahwa data termasuk kategori tersebut sementara nilai 0 merepresentasikan bahwa data bukanlah termasuk kategori tersebut. Dala dataset ini, atribut yang memiliki kategori lebih dari 2 adalah **work_type dan smoking_status**
- Alasan Penggunaan

Alasan penggunaan one-hot encoding adalah untuk mengubah data kategorik menjadi data numerik namun masih tetap merepresentasikan data kategorik tersebut. Hal ini dilakukan karena akan membantu model untuk memahami data menjadi lebih baik dimana mesin hanya mengerti angka dan bukan teks/huruf. Selain itu, one-hot encoding akan memudahkan penentuan probabilitas untuk setiap value.

#### **- Membagi Data Train dan Test**
- Proses

Sebelum data digunakan untuk membuat model, data harus terlebih dulu dibagi menjadi data latih dan data uji. Saya menggunakan rasio 80:20 untuk data latih dan data uji. pembagian data latih dan data uji menggunakan fungsi **train_test_split(X, y, test_size = 0.2, random_state = 123)** dimana X dan y adalah data yang akan di pisah, test_size adalah ukuran data latih, dan random_state adalah inisiasi internal random generator yang akan menentukan indeks dalam pembagian dataset.
- Alasan Penggunaan

Pembagian dataset menjadi data latih dan data uji bertujuan untuk memudahkan mengevaluasi seberapa baik generalisasi model terhadap data baru. Selain itu, hal ini dilakukan untuk tidak mengotori data uji dengan informasi yang kita dapat dari data latih.

#### **- Standarisasi**

- Proses

Standarisasi adalah proses yang dilakukan pada data numerik untuk mengurangkan mean kemudian membaginya dengan standar deviasi untuk menggeser distribusi. Pada data kali ini, saya menggunakan StandardScaler untuk melakukan standarisasi. StandardScaler akan menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0.

- Alasan Penggunaan

Standarisasi digunakan untuk membuat model machine learning memiliki performa lebih baik dan konvergen karena dimodelkan dengan data yang mendekati distribusi normal atau data dengan skala relatif.

## **5. Modeling**

Modeling adalah proses untuk membuat model yang dapat melakukan tugas sesuai dengan keinginan kita. Pada proyek kali ini, saya menggunakan 3 algoritma sebagai perbandingan yaitu KNN, Random Forest, dan Logistic Regression. 
- **Algoritma KNN**

Pada implementasi algoritma KNN saya menggunakan fungsi **KNeighborsRegressor** dari **sklearn.neighbors** dengan menggunakan parameter n_neighbors sama dengan 10 yang menunjukan banyaknya tetangga terdekat yang ikut dalam penentuan kelas. Secara default, metriks penentuan jaraknya adalah euclidean.

##### **- Kelebihan KNN**
1. Tidak memiliki periode training, sehingga algoritma ini sangat efisien dalam hal waktu dan kompleksitas.
2. Implementasi yang mudah, KNN mudah untuk diimplementasikan karena hanya mengukur jarak antar data.
3. Karena tidak memiliki periode training, data baru bisa ditambahkan kapan saja.

##### **- Kekurangan KNN**
1. Tidak bekerja dengan baik saat menggunakan dataset yang besar karena sangat boros dalam hal komputasi.
2. Tidak bekerja dengan baik pad adataset dengan dimensi tinggi karena menyusahkan dalam perhitungan jarak antar data.
3. Sensitif terhadap outlier dan missing data.
4. Seluruh data harus dilakukan scaling atau normalisasi dan standarisasi.

- **Algoritma Random Forest**

Pada implementasi Random Forest saya menggunakan fungsi **RandomForestRegressor** dari package **sklearn.ensemble**. Dengan menggunakan parameter n_estimators sama dengan 50 yang menunjukan jumlah tree di forest, max_depth sama dengan 16 yang menunjukan kedalaman atau panjang pohon, random_state sama dengan 55 yang digunakan untuk mengontrol random number generator, dan n_jobs sama dengan -1 yang berarti semua proses dilaksanakan secara paralel.

##### **- Kelebihan Random Forest**
1. Dapat mengurangi variasi data sehingga dapat memperbaiki akurasi
2. Dapat menyelesaikan permasalahan data kategorik dan numerik sekaligus.
3. Dapat menangani missing values.
4. Tidak sensitif terhadap outlier dan dapat menanganinya secara otomatis.

##### **- Kekurangan Random Forest**
1. Kompleksitas yang tinggi karena menggunakan banyak tree dan menggabungkan berbagai output dari tree tersebut.
2. Memiliki periode training yang lebih lama karena akan menghasilkan banyak tree dan membuat keputusan berdasarkan mayoritas dari tree.

- **Algoritma Logistic Regression**

Pada implementasi Logistic Regression saya menggunakan fungsi **LogisticRegression** dari package **sklearn.linear_model**. Dengan menggunakan parameter C sama dengan 0.1 yang menunjukan regularization yang kuat karena nilainya yang semakin kecil (regularisasi digunakan untuk menghindari overfitting), random_state sama dengan 123 yang digunakan untuk mengontrol random number generator, dans ecara default menetapkan penalti dengan aturan L2 sebagai cara untuk belajar.

##### **- Kelebihan Logistic Regression**
1. Mudah untuk diimplementasikan, dijabarkan, dan efisien saat pelatihan data.
2. Dapat bekerja dengan baik saat dataset dapat dipisahkan secara linear.
3. Lebih tahan terhadap overfitting karena menerapkan Regularization.
4. Tidak hanya menunjukan relevansi predictor(coefficient size) tetapi juga menunjukan arah asosiasi(negatif atau positif).
##### **- Kekurangan Logistic Regression**
1. Terbatas pada data yang dapat dipisahkan secara linear.
2. Hanya dapat digunakan untuk memprediksi data yang bersifat diskrit dan memiliki permasalahan dalam menganalisa data yang kontinu.

##### **Algoritma Terbaik**
Berdasarkan kekurangan dan kelebihan yang dijabarkan, menurut saya algoritma yang mungkin bekerja dengan sangat baik pada data kali ini adalah algoritma KNN. Hal ini disebabkan karena dataset yang memiliki dimensionality rendah, jumlah data yang cenderung kecil, dan telah dilakukan proses untuk menangani missing value serta outlier. Selain itu, dilihat dari metriks MSE pada data latih dan data uji KNN adalah model yang menunjukan error yang kecil pada keduanya dan tidak terindikasi overfitting.

## **6. Evaluasi**
Metriks evaluasi yang saya gunakan pada proyek kali ini adalah MSE (Mean Squared Error). Sesuai dengan konteks data, problem statement, dan solusi yang diinginkan yaitu lebih menekankan untuk mengurangi kesalahan prediksi(error). MSE bekerja dengan cara mengurangi nilai aktual dengan nilai prediksi dan hasilnya dikuadratkan kemudian dijumlahkan secara keseluruhan dan membaginya dengan banyaknya data yang ada. Secara umum, formula untuk MSE adalah sebagai berikut:
[![MSE_Formula](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/MSE_Formula.png?raw=true)

Hasil yang didapatkan pada MSE data latih dan data uji adalah sebagai berikut:
[![Box_MSE](https://github.com/jonywony/Dicoding-ML-Terapan-Submission1/blob/main/Gambar/Result_MSE.png?raw=true)
- **Algoritma KNN**

##### **- MSE Data latih:** 0.000031
##### **- MSE Data uji:** 0.000032
##### **- Analisis Hasil:** 
Dapat dilihat bahwa MSE yang didapatkan sangat kecil pada kedua data latih dan data uji, selain itu tidak terindikasi adanya overfitting pada data latih.

- **Algoritma Random Forest**
##### **- MSE Data latih:** 0.000006
##### **- MSE Data uji:** 0.000035
##### **- Analisis Hasil:**
Dapat dilihat bahwa MSE yang didapatkan sangat kecil pada kedua data latih dan data uji, namun terdapat kemungkinan adanya overfitting pada data latih karena MSE pada data latih jauh lebih kecil dibandingkan MSE pada data uji.

- **Algoritma Logistic Regression**
##### **- MSE Data latih:** 0.000039
##### **- MSE Data uji:** 0.000033
##### **- Analisis Hasil:**
Dapat dilihat bahwa MSE yang didapatkan sangat kecil pada kedua data latih dan data uji, namun terdapat kemungkinan adanya overfitting pada data latih karena MSE pada data latih sedikit lebih kecil dibandingkan MSE pada data uji.

##### **Kesimpulan:** 
Pada saat percobaan prediksi pada dataset, terlihat bahwa ketiganya sama-sama menunjukan hasil yang benar. Namun berdasarkan evaluasi mertriks MSE, didapatkan hasil bahwa algoritma terbaik pada model kali ini adalah KNN.
