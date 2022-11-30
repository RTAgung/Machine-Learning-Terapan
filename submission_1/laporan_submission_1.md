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

![Box Plot](https://raw.githubusercontent.com/RTAgung/Machine-Learning-Terapan/master/assets/submission_1/DU_boxplot.png)

Terlihat pada fitur BMI terdapat beberapa data yang outlier, sehingga diterapkan penghapusan outlier menggunakan metode IQR, alhasil terdapat 9 data yang terdeteksi sebagai outlier seperti berikut. 

| index | age |    sex |   bmi | children | smoker |    region |     charges |
|------:|----:|-------:|------:|---------:|-------:|----------:|------------:|
|  116  |  58 |   male | 49.06 |        0 |     no | southeast | 11381.32540 |
|  286  |  46 | female | 48.07 |        2 |     no | northeast |  9432.92530 |
|  401  |  47 |   male | 47.52 |        1 |     no | southeast |  8083.91980 |
|  543  |  54 | female | 47.41 |        0 |    yes | southeast | 63770.42801 |
|  847  |  23 |   male | 50.38 |        1 |     no | southeast |  2438.05520 |
|  860  |  37 | female | 47.60 |        2 |    yes | southwest | 46113.51100 |
|  1047 |  22 |   male | 52.58 |        1 |    yes | southeast | 44501.39820 |
|  1088 |  52 |   male | 47.74 |        1 |     no | southeast |  9748.91060 |
|  1317 |  18 |   male | 53.13 |        0 |     no | southeast |  1163.46270 |

Kemudian pada fitur charges sebagai target model terdapat banyak data yang diluar boxplot, namun sangat beresiko apabila semua data tersebut dihapus, sehingga hanya data terpencil yang akan dihapus secara manual, yaitu menghapus data yang nilainya diatas 50000. Hasil penghapusan kedua outlier tersebut jumlah dataset sekarang menjadi 1323.

Kemudian univariate analysis dilakukan pada satuan data untuk melihat bagaimana persebaran dan jumlah datanya setiap fitur, analisis ini hanya menggunakan beberapa teknik visualisasi yaitu histogram chart dan bar chart.

Terakhir multivariate analysis dilakukan untuk melihat hubungan setiap fitur variabel dengan fitur target (charges). Pada fitur kategorikal menggunakan categorical plot dengan sumbu x sebagai nilai kategori dan sumbu y sebagai rata-rata dari nilai charges.

![Category 1](https://raw.githubusercontent.com/RTAgung/Machine-Learning-Terapan/master/assets/submission_1/DU_MA_CAT_1.png)

![Category 2](https://raw.githubusercontent.com/RTAgung/Machine-Learning-Terapan/master/assets/submission_1/DU_MA_CAT_2.png)

![Category 3](https://raw.githubusercontent.com/RTAgung/Machine-Learning-Terapan/master/assets/submission_1/DU_MA_CAT_3.png)

Kemudian pada fitur numerikal menggunakan scatter plot yang dihubungkan satu sama lain.

![Numeric 1](https://raw.githubusercontent.com/RTAgung/Machine-Learning-Terapan/master/assets/submission_1/DU_MA_NUM_1.png)

Berikut juga hasil nilai korelasi antar variabel.

![Numeric 2](https://raw.githubusercontent.com/RTAgung/Machine-Learning-Terapan/master/assets/submission_1/DU_MA_NUM_2.png)

Dari dua data visualisasi diatas menunjukkan bahwa hubungan antara variabel charges dan children tidak memiliki korelasi ataupun pola yang baik. Sehingga variabel children dapat dihapuskan, jadi saat ini terdapat enam variabel yang akan digunakan termasuk variabel target yaitu age, bmi, sex, smoker, region, dan charges.

## Data Preparation
Setelah analisis dilakukan, pada bagian ini data akan dilakukan beberapa proses teknik data preparation diantaranya yaitu encoding, splitting dataset, dan standardization.

Pada tahap pertama yaitu encoding dilakukan dengan mengubah nilai dari fitur kategorikal menjadi fitur numerik karena model machine learning hanya dapat menerima inputan data numerik. Terdapat tiga fitur kategorikal pada proyek ini yaitu sex, smoker, dan region. Sex dan smoker memiliki nilai yang binary, sehingga pada fitur tersebut encoding dilakukan dengan teknik Ordinal Encoder untuk mengubah nilai menjadi 0 dan 1 pada satu kolom. Berbeda dengan fitur region yang memiliki nilai non binary, pendekatan encoding dilakukan menggunakan teknik One Hot Encoding untuk memecah nilai region menjadi empat kolom baru yang berisi nilai 0 dan 1. Hasil encoding tersebut dapat dilihat sebagai berikut.

| index | age |    sex |    bmi | smoker |     charges | region_northeast | region_northwest | region_southeast | region_southwest |
|------:|----:|-------:|-------:|-------:|------------:|-----------------:|-----------------:|-----------------:|-----------------:|
|   0   |  19 |      0 | 27.900 |      0 | 16884.92400 |                0 |                0 |                0 |                1 |
|   1   |  18 |      1 | 33.770 |      1 |  1725.55230 |                0 |                0 |                1 |                0 |
|   2   |  28 |      1 | 33.000 |      1 |  4449.46200 |                0 |                0 |                1 |                0 |
|   3   |  33 |      1 | 22.705 |      1 | 21984.47061 |                0 |                1 |                0 |                0 |
|   4   |  32 |      1 | 28.880 |      1 |  3866.85520 |                0 |                1 |                0 |                0 |

Setelah itu, splitting dataset dilakukan untuk membagi dataset menjadi data training dan data testing dengan rasio perbandingan 85% data training dan 15% data testing. Tahapan ini dilakukan untuk mempertahankan beberapa data sehingga sebagian data akan dilakukan training pada model kemudian sebagian data lainnya dapat dilakukan testing untuk evaluasi terhadap model yang telah di-training. Sehingga total jumlah data hasil splitting yaitu:
- Total sample in whole dataset: 1323
- Total sample in train dataset: 1124
- Total sample in test dataset: 199

Tahapan terakhir akan dilakukan proses standardization pada data numerikal yaitu fitur age dan bmi. Proses ini dilakukan karena model machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada tahap ini standardization pada data dilakukan dengan menggunakan fungsi StandarScaler pada library Scikit-learn. Hasil proses tersebut dapat dilihat sebagai berikut pada fitur age dan bmi.

| index |       age | sex |       bmi | smoker | region_northeast | region_northwest | region_southeast | region_southwest |
|------:|----------:|----:|----------:|-------:|-----------------:|-----------------:|-----------------:|-----------------:|
|  451  | -0.648129 |   1 | -1.053703 |      1 |                0 |                1 |                0 |                0 |
|  393  |  0.719738 |   1 |  0.168893 |      1 |                1 |                0 |                0 |                0 |
|  896  |  0.287780 |   0 | -1.745436 |      0 |                1 |                0 |                0 |                0 |
|  642  |  1.583654 |   1 |  0.603237 |      1 |                1 |                0 |                0 |                0 |
|  1113 | -0.792115 |   0 | -0.683707 |      0 |                0 |                1 |                0 |                0 |

## Modeling
Selanjutnya ketika data sudah siap untuk digunakan maka pengembangan model machine learning akan dilakukan. Proyek ini mengembangkan model machine learning sesuai dengan kasus yang dihadapi yaitu regresi. Terdapat lima model yang dikembangkan diantaranya K Nearest Neighbors, Random Forest, Ada Boost, Support Vector Machine, dan Neural Network Tensorflow. Selain mencari model yang terbaik diantara kelimanya, setiap model juga dilakukan hyperparameter tuning untuk mencari parameter terbaik pada setiap model. Hyperparameter tuning dilakukan menggunakan fungsi GridSearchCV dengan parameter scoring = 'neg_mean_squared_error'.

### K Nearest Neighbors
Pada model ini tuning dilakukan terhadap parameter : 
- 'n_neighbors': [1, 3, 6, 10, 15, 21]

Berdasarkan hasil pelatihan yang dilakukan menggunakan parameter tersebut,
- Best Estimator: KNeighborsRegressor(n_neighbors=21)
- Best Score: -126865357.842845

Kemudian berikut hasil scoring terhadap seluruh kombinasi parameter

| mean_test_score | std_test_score | params |
|----------------:|---------------:|-------:|
| -237052617.529261 | 25340162.280366 | {'n_neighbors': 1}  |
| -154781868.643013 | 17413500.637074 | {'n_neighbors': 3}  |
| -137588911.863427 | 13456091.660276 | {'n_neighbors': 6}  |
| -130663719.070448 | 10439591.356618 | {'n_neighbors': 10} |
| -129627748.447250 | 9149937.748438  | {'n_neighbors': 15} |
| -126865357.842845 | 8543500.633042  | {'n_neighbors': 21} |


### Random Forest
Pada model ini tuning dilakukan terhadap parameter : 
- 'n_estimators': [30, 50, 100, 150, 200]
- 'max_depth': [None, 16, 32, 64]

Berdasarkan hasil pelatihan yang dilakukan menggunakan parameter tersebut,
- Best Estimator: RandomForestRegressor(max_depth=16, n_estimators=150)
- Best Score: -143738339.9118495

Kemudian berikut hasil scoring terhadap seluruh kombinasi parameter

| mean_test_score | std_test_score | params |
|----------------:|---------------:|-------:|
| -147284911.322931 | (16498876.839821) | {'max_depth': None, 'n_estimators': 30}
| -146137329.401260 | (16526020.403304) | {'max_depth': None, 'n_estimators': 50}
| -144897169.896425 | (16155264.819043) | {'max_depth': None, 'n_estimators': 100}
| -144560725.047786 | (15975373.169902) | {'max_depth': None, 'n_estimators': 150}
| -144633957.382563 | (15768635.009804) | {'max_depth': None, 'n_estimators': 200}
| -147060452.578452 | (16540150.678115) | {'max_depth': 16, 'n_estimators': 30}
| -145854101.282563 | (16896667.577173) | {'max_depth': 16, 'n_estimators': 50}
| -144286494.376783 | (15993584.970384) | {'max_depth': 16, 'n_estimators': 100}
| -143738339.911849 | (16006030.498809) | {'max_depth': 16, 'n_estimators': 150}
| -143971162.677600 | (15699081.711477) | {'max_depth': 16, 'n_estimators': 200}
| -147284911.322931 | (16498876.839821) | {'max_depth': 32, 'n_estimators': 30}
| -146137329.401260 | (16526020.403304) | {'max_depth': 32, 'n_estimators': 50}
| -144897169.896425 | (16155264.819043) | {'max_depth': 32, 'n_estimators': 100}
| -144560725.047786 | (15975373.169902) | {'max_depth': 32, 'n_estimators': 150}
| -144633957.382563 | (15768635.009804) | {'max_depth': 32, 'n_estimators': 200}
| -147284911.322931 | (16498876.839821) | {'max_depth': 64, 'n_estimators': 30}
| -146137329.401260 | (16526020.403304) | {'max_depth': 64, 'n_estimators': 50}
| -144897169.896425 | (16155264.819043) | {'max_depth': 64, 'n_estimators': 100}
| -144560725.047786 | (15975373.169902) | {'max_depth': 64, 'n_estimators': 150}
| -144633957.382563 | (15768635.009804) | {'max_depth': 64, 'n_estimators': 200}

### Ada Boost
Pada model ini tuning dilakukan terhadap parameter : 
- 'n_estimators': [15, 30, 50, 100]
- 'learning_rate': [0.005, 0.05, 0.5, 1]

Berdasarkan hasil pelatihan yang dilakukan menggunakan parameter tersebut,
- Best Estimator: AdaBoostRegressor(learning_rate=0.005, n_estimators=100)
- Best Score: -122510808.63991885

Kemudian berikut hasil scoring terhadap seluruh kombinasi parameter

| mean_test_score | std_test_score | params |
|----------------:|---------------:|-------:|
| -123162352.623728 | (10443624.617998) | {'learning_rate': 0.005, 'n_estimators': 15}
| -122987596.521027 | (9642180.854195) | {'learning_rate': 0.005, 'n_estimators': 30}
| -122675531.857457 | (9610077.479041) | {'learning_rate': 0.005, 'n_estimators': 50}
| -122510808.639919 | (8667876.445030) | {'learning_rate': 0.005, 'n_estimators': 100}
| -123254425.167231 | (9074340.640534) | {'learning_rate': 0.05, 'n_estimators': 15}
| -123277150.095297 | (7994175.052963) | {'learning_rate': 0.05, 'n_estimators': 30}
| -123772481.822563 | (7177976.366041) | {'learning_rate': 0.05, 'n_estimators': 50}
| -126082715.897090 | (6808777.648890) | {'learning_rate': 0.05, 'n_estimators': 100}
| -126942064.986046 | (8554117.771821) | {'learning_rate': 0.5, 'n_estimators': 15}
| -127051948.973104 | (8768557.513287) | {'learning_rate': 0.5, 'n_estimators': 30}
| -127051948.973104 | (8768557.513287) | {'learning_rate': 0.5, 'n_estimators': 50}
| -127051948.973104 | (8768557.513287) | {'learning_rate': 0.5, 'n_estimators': 100}
| -129972351.737216 | (9777662.853013) | {'learning_rate': 1, 'n_estimators': 15}
| -129972351.737216 | (9777662.853013) | {'learning_rate': 1, 'n_estimators': 30}
| -129972351.737216 | (9777662.853013) | {'learning_rate': 1, 'n_estimators': 50}
| -129972351.737216 | (9777662.853013) | {'learning_rate': 1, 'n_estimators': 100}

### Support Vector Machine
Pada model ini tuning dilakukan terhadap parameter : 
- 'kernel': ['rbf', 'sigmoid']
- 'C': [1, 3, 6, 10, 15, 20]

Berdasarkan hasil pelatihan yang dilakukan menggunakan parameter tersebut,
- Best Estimator: SVR(C=6, kernel='sigmoid')
- Best Score: -151112531.66046908

Kemudian berikut hasil scoring terhadap seluruh kombinasi parameter

| mean_test_score | std_test_score | params |
|----------------:|---------------:|-------:|
| -152262049.680334 | (10755804.746422) | {'C': 1, 'kernel': 'rbf'}
| -152008972.983945 | (10880376.897193) | {'C': 1, 'kernel': 'sigmoid'}
| -152008547.519657 | (10973042.216585) | {'C': 3, 'kernel': 'rbf'}
| -151364337.718839 | (11159373.775342) | {'C': 3, 'kernel': 'sigmoid'}
| -151879182.653759 | (11077199.766630) | {'C': 6, 'kernel': 'rbf'}
| -151112531.660469 | (11589292.718487) | {'C': 6, 'kernel': 'sigmoid'}
| -151724175.702482 | (11088566.136385) | {'C': 10, 'kernel': 'rbf'}
| -151241487.813537 | (11988133.352514) | {'C': 10, 'kernel': 'sigmoid'}
| -151608527.102911 | (11334414.466716) | {'C': 15, 'kernel': 'rbf'}
| -151967956.252499 | (12396954.001293) | {'C': 15, 'kernel': 'sigmoid'}
| -151889752.992690 | (11607860.779026) | {'C': 20, 'kernel': 'rbf'}
| -152991528.472361 | (13349549.448385) | {'C': 20, 'kernel': 'sigmoid'}

### Neural Network Tensorflow
Pada model ini tuning dilakukan terhadap parameter : 
- 'optimizer': ['RMSprop', 'Adam']
- 'loss': ['mse', 'huber']
- 'epochs': [50, 100, 150]

Berdasarkan hasil pelatihan yang dilakukan menggunakan parameter tersebut,
- Best Estimator: KerasRegressor(optimizer=RMSprop, loss=mse, epochs=50)
- Best Score: -120666910.91840395

Kemudian berikut hasil scoring terhadap seluruh kombinasi parameter

| mean_test_score | std_test_score | params |
|----------------:|---------------:|-------:|
| -120666910.918404 | (9366668.452327) | {'epochs': 50, 'loss': 'mse', 'optimizer': 'RMSprop'}
| -120946984.129820 | (9101642.449856) | {'epochs': 50, 'loss': 'mse', 'optimizer': 'Adam'}
| -156885036.817875 | (15970558.476103) | {'epochs': 50, 'loss': 'huber', 'optimizer': 'RMSprop'}
| -157815815.488131 | (17329257.614572) | {'epochs': 50, 'loss': 'huber', 'optimizer': 'Adam'}
| -120947090.180983 | (10136695.256762) | {'epochs': 100, 'loss': 'mse', 'optimizer': 'RMSprop'}
| -120885049.276668 | (9343967.033108) | {'epochs': 100, 'loss': 'mse', 'optimizer': 'Adam'}
| -158455322.168230 | (17133772.901857) | {'epochs': 100, 'loss': 'huber', 'optimizer': 'RMSprop'}
| -157275105.477030 | (15647200.913241) | {'epochs': 100, 'loss': 'huber', 'optimizer': 'Adam'}
| -120927402.624001 | (9164873.075752) | {'epochs': 150, 'loss': 'mse', 'optimizer': 'RMSprop'}
| -120692440.185528 | (9392371.958089) | {'epochs': 150, 'loss': 'mse', 'optimizer': 'Adam'}
| -157236387.153332 | (17087097.572732) | {'epochs': 150, 'loss': 'huber', 'optimizer': 'RMSprop'}
| -156973631.565901 | (16424942.878246) | {'epochs': 150, 'loss': 'huber', 'optimizer': 'Adam'}

Selanjutnya dari kelima model machine learning tersebut akan dibandingkan bagaimana tingkat nilai mse yang dihasilkan. Berikut tabel dan visualisasi hasil perbandingannya.

|      model |         train |          test |
|-----------:|--------------:|--------------:|
|     KNN    | 113656.317417 | 115089.320615 |
|     RF     |    21458.2603 | 144220.198493 |
|  Boosting  | 114916.727735 | 117633.173742 |
|     SVM    | 150414.137986 | 142707.859543 |
| Tensorflow | 118781.874706 | 116948.339316 |

![Evaluation 2](https://raw.githubusercontent.com/RTAgung/Machine-Learning-Terapan/master/assets/submission_1/ML_EVAL_2.png)

Dari hasil perbandingan kelima model tersebut, model dari KNN, Tensorflow, dan Boosting memiliki nilai evaluasi mse yang hampir sama, sedangkan SVM memiliki nilai mse yang sangat tinggi, dan Random Forest memiliki nilai mse yang sangat rendah pada data training namun pada data testing nilai mse yang diperoleh paling tinggi diantara model yang lain. Sehingga kandidat model terbaik yang akan digunakan yaitu pada model KNN, Tensorflow, dan Boosting. Namun secara kompleksitas algoritma KNN merupakan model yang ringan dan kompleksitas rendah, sehingga perhitungan model dapat dijalankan dengan cepat dan akurat. Maka dari itu kesimpulan yang diperoleh dari perbandingan terhadap lima metode tersebut, model KNN dipilih dan digunakan pada proyek ini.

## Evaluation
Tahapan terakhir yang perlu dilakukan adalah evaluasi model machine learning. Seperti yang sudah dijelaskan pada bagian-bagian sebelumnya bahwa proyek ini akan menghitung evaluasi model menggunakan mean squared error (mse). Mean Squared Error adalah Rata-rata Kesalahan kuadrat diantara nilai aktual dan nilai peramalan. Metode Mean Squared Error secara umum digunakan untuk mengecek estimasi berapa nilai kesalahan pada peramalan. Nilai Mean Squared Error yang rendah atau nilai mean squared error mendekati nol menunjukkan bahwa hasil peramalan sesuai dengan data aktual dan bisa dijadikan untuk perhitungan peramalan di periode mendatang. Berikut ini cara menghitung nilai mse dengan n sebagai jumlah data.

$$ MSE = \frac{\sum_{t=1}^{n}\left ( actual-predict \right )^{2}}{n} $$

Dari hasil pemodelan yang telah dilakukan sebelumnya, model KNN digunakan dalam proyek dengan tingkat nilai mse sebesar 113656.32 pada data training dan 115089.32 pada data testing. Dilihat dari nilai standar deviasi dari variabel target (charges) yaitu sebesar 12110.01 nilai mse pada model KNN bukanlah tinkat error  yang buruk, namun juga bukan tingkat error yang baik. Perolehan hasil evaluasi dari model yang telah dibangun tidak hanya bergantung pada model, namun juga memiliki pengaruh besar terhadap data yang digunakan. Sehingga pada proyek ini, hasil dari proses yang telah dilakukan sudah cukup baik.
