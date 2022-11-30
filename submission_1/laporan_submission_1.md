# Laporan Proyek Machine Learning - Rama Tri Agung
## Domain Proyek
Asuransi kesehatan merupakan jenis asuransi yang paling mendasar, atau asuransi yang menjadi prioritas utama untuk dimiliki. Karena asuransi ini akan menanggung biaya pengobatan. Namun bagi segelintir masyarakat, tagihan Asuransi bukanlah nominal yang kecil. Rutin membayar tagihan asuransi tentu bisa mengganggu kondisi keuangan jika pengeluarannya cukup signifikan. Hal ini juga didukung oleh tidak adanya jumlah yang ideal untuk tagihan asuransi. Setiap orang menghadapi risiko keuangan yang berbeda; semakin lengkap manfaat asuransinya, semakin tinggi pula tagihan yang harus dibayar.

Untuk mengembangkan produk asuransi kesehatan terbaik yang diminati oleh masyarakat, perusahaan asuransi harus memiliki akses ke data historis untuk memperkirakan biaya medis setiap pengguna. Perusahaan asuransi medis dapat menggunakan data ini untuk mengembangkan model penetapan harga yang lebih akurat, merencanakan hasil asuransi tertentu, atau mengelola portofolio besar. Tujuan dari semua kasus ini adalah untuk memprediksi biaya asuransi secara akurat.

## Business Understanding
### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem prediksi biaya kesehatan untuk menjawab permasalahan berikut:
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap biaya kesehatan?
- Berapa prediksi biaya kesehatan dengan karakteristik atau fitur tertentu?
- Berapa besar tingkat eror dari hasil prediksi yang dilakukan?

### Goals
Untuk menjawab pertanyaan tersebut, perusahaan membuat predictive modelling dengan tujuan atau goals sebagai berikut:
- Mengetahui fitur yang memliki korelasi tinggi dengan biaya kesehatan
- Membuat model machine learning yang dapat memprediksi biaya kesehatan seakurat mungkin berdasarkan fitur-fitur yang ada
- Melakukan evaluasi model yang telah dibangun untuk mengetahui tingkat kesalahan model

### Solution Statements
Dari tujuan atau goals yang ditentukan, berikut ini solusi atau cara untuk meraih goals tersebut:
- Analisis data dilakukan lebih detail dengan membersihkan data dan beberapa visualisasi data sebelum mencari nilai korelasi
- Menggunakan beberapa model machine learning untuk memperoleh hasil prediksi yang paling akurat dan setiap model tersebut dilakukan hyperparameter tuning untuk memperoleh model terbaik
- Seluruh hasil pelatihan model akan dievaluasi berdasarkan metrik Mean Squared Error (mse)

## Data Understanding
Pembuatan model machine learning pada proyek menggunakan 1338 data dari Medical Cost Personal Datasets, data dapat diunduh pada tautan ini.

### Variabel-variabel pada Medical Cost Personal Datasets adalah sebagai berikut:
- age: usia penerima asuransi. (numerikal)
- sex: jenis kelamin. (kategorikal)['female', 'male']
- bmi: Body mass index, memberikan pengertian tentang tubuh, bobot yang tinggi/rendah relatif terhadap tinggi badan. (numerikal)
- children: Jumlah anak yang ditanggung. (numerikal)
- smoker: status merokok. (kategorikal)['yes', 'no']
- region: daerah penerima asuransi di AS. (kategorikal)['northeast', 'southeast', 'southwest', 'northwest']
- charges: Biaya medis individu yang ditagih oleh asuransi kesehatan. (numerikal)

Pada bagian ini dilakukan beberapa proses analisis untuk melihat bagaimana kondisi dari dataset. Hal pertama yang dilakukan adalah melihat apakah ada nilai yang kosong pada data dengan cara cek null data dan cek data nol (0) pada fitur bmi dan age, alhasil tidak ada nilai yang kosong pada data. 

Kemudian pencarian outlier dilakukan dengan melihat distribusi data menggunakan boxplot pada data numerik seperti berikut.

![Box Plot](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/DU_boxplot.png)

Terlihat pada fitur BMI terdapat beberapa data yang outlier, sehingga diterapkan penghapusan outlier menggunakan metode IQR, alhasil terdapat 9 data yang terdeteksi sebagai outlier seperti berikut. 

![IQR](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/DU_IQR.png)

Kemudian pada fitur charges sebagai target model terdapat banyak data yang diluar boxplot, namun sangat beresiko apabila semua data tersebut dihapus, sehingga hanya data terpencil yang akan dihapus secara manual, yaitu menghapus data yang nilainya diatas 50000. Hasil penghapusan kedua outlier tersebut jumlah dataset sekarang menjadi 1323.

Kemudian univariate analysis dilakukan pada satuan data untuk melihat bagaimana persebaran dan jumlah datanya setiap fitur, analisis ini hanya menggunakan beberapa teknik visualisasi yaitu histogram chart dan bar chart.

Terakhir multivariate analysis dilakukan untuk melihat hubungan setiap fitur variabel dengan fitur target (charges). Pada fitur kategorikal menggunakan categorical plot dengan sumbu x sebagai nilai kategori dan sumbu y sebagai rata-rata dari nilai charges.

![Category 1](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/DU_MA_CAT_1.png)
![Category 2](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/DU_MA_CAT_2.png)
![Category 3](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/DU_MA_CAT_3.png)

Kemudian pada fitur numerikal menggunakan scatter plot yang dihubungkan satu sama lain.

![Numeric 1](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/DU_MA_NUM_1.png)

Berikut juga hasil nilai korelasi antar variabel.

![Numeric 2](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/DU_MA_NUM_2.png)

Dari dua data visualisasi diatas menunjukkan bahwa hubungan antara variabel charges dan children tidak memiliki korelasi ataupun pola yang baik. Sehingga variabel children dapat dihapuskan, jadi saat ini terdapat enam variabel yang akan digunakan termasuk variabel target yaitu age, bmi, sex, smoker, region, dan charges.

## Data Preparation
Setelah analisis dilakukan, pada bagian ini data akan dilakukan beberapa proses teknik data preparation diantaranya yaitu encoding, splitting dataset, dan standardization.

Pada tahap pertama yaitu encoding dilakukan dengan mengubah nilai dari fitur kategorikal menjadi fitur numerik karena model machine learning hanya dapat menerima inputan data numerik. Terdapat tiga fitur kategorikal pada proyek ini yaitu sex, smoker, dan region. Sex dan smoker memiliki nilai yang binary, sehingga pada fitur tersebut encoding dilakukan dengan teknik Ordinal Encoder untuk mengubah nilai menjadi 0 dan 1 pada satu kolom. Berbeda dengan fitur region yang memiliki nilai non binary, pendekatan encoding dilakukan menggunakan teknik One Hot Encoding untuk memecah nilai region menjadi empat kolom baru yang berisi nilai 0 dan 1. Hasil encoding tersebut dapat dilihat sebagai berikut.

![Encoding](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/DP_ENC.png)

Setelah itu, splitting dataset dilakukan untuk membagi dataset menjadi data training dan data testing dengan rasio perbandingan 85% data training dan 15% data testing. Tahapan ini dilakukan untuk mempertahankan beberapa data sehingga sebagian data akan dilakukan training pada model kemudian sebagian data lainnya dapat dilakukan testing untuk evaluasi terhadap model yang telah di-training. Sehingga total jumlah data hasil splitting ini sebagai berikut.

![Splitting](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/DP_SPLIT.png)

Tahapan terakhir akan dilakukan proses standardization pada data numerikal yaitu fitur age dan bmi. Proses ini dilakukan karena model machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada tahap ini standardization pada data dilakukan dengan menggunakan fungsi StandarScaler pada library Scikit-learn. Hasil proses tersebut dapat dilihat sebagai berikut pada fitur age dan bmi.

![Standardization](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/DP_STD.png)

## Modeling
Selanjutnya ketika data sudah siap untuk digunakan maka pengembangan model machine learning akan dilakukan. Proyek ini mengembangkan model machine learning sesuai dengan kasus yang dihadapi yaitu regresi. Terdapat lima model yang dikembangkan diantaranya K Nearest Neighbors, Random Forest, Ada Boost, Support Vector Machine, dan Neural Network Tensorflow. Selain mencari model yang terbaik diantara kelimanya, setiap model juga dilakukan hyperparameter tuning untuk mencari parameter terbaik pada setiap model. Hyperparameter tuning dilakukan menggunakan fungsi GridSearchCV dengan parameter scoring = 'neg_mean_squared_error'.

### K Nearest Neighbors
Pada model ini tuning dilakukan terhadap parameter : 
- 'n_neighbors': [1, 3, 6, 10, 15, 21]

Berdasarkan hasil pelatihan yang dilakukan menggunakan parameter tersebut,
- Best Estimator: KNeighborsRegressor(n_neighbors=21)
- Best Score: -126865357.842845

Kemudian berikut hasil scoring terhadap seluruh kombinasi parameter

![KNN](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/ML_KNN.png)

### Random Forest
Pada model ini tuning dilakukan terhadap parameter : 
- 'n_estimators': [30, 50, 100, 150, 200]
- 'max_depth': [None, 16, 32, 64]

Berdasarkan hasil pelatihan yang dilakukan menggunakan parameter tersebut,
- Best Estimator: RandomForestRegressor(max_depth=16, n_estimators=150)
- Best Score: -143738339.9118495

Kemudian berikut hasil scoring terhadap seluruh kombinasi parameter

![RF](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/ML_RF.png)

### Ada Boost
Pada model ini tuning dilakukan terhadap parameter : 
- 'n_estimators': [15, 30, 50, 100]
- 'learning_rate': [0.005, 0.05, 0.5, 1]

Berdasarkan hasil pelatihan yang dilakukan menggunakan parameter tersebut,
- Best Estimator: AdaBoostRegressor(learning_rate=0.005, n_estimators=100)
- Best Score: -122510808.63991885

Kemudian berikut hasil scoring terhadap seluruh kombinasi parameter

![AB](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/ML_AB.png)

### Support Vector Machine
Pada model ini tuning dilakukan terhadap parameter : 
- 'kernel': ['rbf', 'sigmoid']
- 'C': [1, 3, 6, 10, 15, 20]

Berdasarkan hasil pelatihan yang dilakukan menggunakan parameter tersebut,
- Best Estimator: SVR(C=6, kernel='sigmoid')
- Best Score: -151112531.66046908

Kemudian berikut hasil scoring terhadap seluruh kombinasi parameter

![SVM](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/ML_SVM.png)

### Neural Network Tensorflow
Pada model ini tuning dilakukan terhadap parameter : 
- 'optimizer': ['RMSprop', 'Adam']
- 'loss': ['mse', 'huber']
- 'epochs': [50, 100, 150]

Berdasarkan hasil pelatihan yang dilakukan menggunakan parameter tersebut,
- Best Estimator: KerasRegressor(optimizer=RMSprop, loss=mse, epochs=50)
- Best Score: -120666910.91840395

Kemudian berikut hasil scoring terhadap seluruh kombinasi parameter

![TF](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/ML_TF.png)

Selanjutnya dari kelima model machine learning tersebut akan dibandingkan bagaimana tingkat nilai mse yang dihasilkan. Berikut tabel dan visualisasi hasil perbandingannya.

![Evaluation 1](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/ML_EVAL_1.png)
![Evaluation 2](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/ML_EVAL_2.png)

Dari hasil perbandingan kelima model tersebut, model dari KNN, Tensorflow, dan Boosting memiliki nilai evaluasi mse yang hampir sama, sedangkan SVM memiliki nilai mse yang sangat tinggi, dan Random Forest memiliki nilai mse yang sangat rendah pada data training namun pada data testing nilai mse yang diperoleh paling tinggi diantara model yang lain. Sehingga kandidat model terbaik yang akan digunakan yaitu pada model KNN, Tensorflow, dan Boosting. Namun secara kompleksitas algoritma KNN merupakan model yang ringan dan kompleksitas rendah, sehingga perhitungan model dapat dijalankan dengan cepat dan akurat. Maka dari itu kesimpulan yang diperoleh dari perbandingan terhadap lima metode tersebut, model KNN dipilih dan digunakan pada proyek ini.

## Evaluation
Tahapan terakhir yang perlu dilakukan adalah evaluasi model machine learning. Seperti yang sudah dijelaskan pada bagian-bagian sebelumnya bahwa proyek ini akan menghitung evaluasi model menggunakan mean squared error (mse). Mean Squared Error adalah Rata-rata Kesalahan kuadrat diantara nilai aktual dan nilai peramalan. Metode Mean Squared Error secara umum digunakan untuk mengecek estimasi berapa nilai kesalahan pada peramalan. Nilai Mean Squared Error yang rendah atau nilai mean squared error mendekati nol menunjukkan bahwa hasil peramalan sesuai dengan data aktual dan bisa dijadikan untuk perhitungan peramalan di periode mendatang. Berikut ini cara menghitung nilai mse.

![MSE](https://github.com/RTAgung/Machine-Learning-Terapan/blob/master/assets/submission_1/EV_MSE.png)

Dimana:
- At = Nilai Aktual
- Ft = Nilai Prediksi
- n = Jumlah Data

Dari hasil pemodelan yang telah dilakukan sebelumnya, model KNN digunakan dalam proyek dengan tingkat nilai mse sebesar 113656.32 pada data training dan 115089.32 pada data testing. Dilihat dari nilai standar deviasi dari variabel target (charges) yaitu sebesar 12110.01 nilai mse pada model KNN bukanlah tinkat error  yang buruk, namun juga bukan tingkat error yang baik. Perolehan hasil evaluasi dari model yang telah dibangun tidak hanya bergantung pada model, namun juga memiliki pengaruh besar terhadap data yang digunakan. Sehingga pada proyek ini, hasil dari proses yang telah dilakukan sudah cukup baik.
