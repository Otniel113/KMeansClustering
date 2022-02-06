# KMeansClustering
Membuat algoritma Unsupervised Learning yaitu K-Means Clustering untuk kasus Pelanggan pada Dealer Mobil. Model dibangun scratch from zero yang dikerjakan secara individu.

# Formulasi Masalah
Masalah yang akan diselesaikan adalah pada toko dealer mobil. Toko dealer tersebut memiliki data-data pelanggan yang mencakup ID, jenis kelamin, umur, sudah memiliki SIM atau belum, kode daerah, sudah pernah asuransi atau belum, umur kendaraan, kendaran pernah rusak atau tidak, premi, kanal penjualan, lama berlangganan dan tertarik atau tidak. <br>

Untuk itu, dilakukan pengklasteran (pengelompokan) pelanggan-pelanggan tersebut untuk berdasarkan data yang ada. Pengelompokan itu akan dipakai untuk analisis lebih lanjut dari pihak dealer tersebut. Untuk itu dibangun model machine learning K-Means Clustering yang merupakan algoritma Unsupervised Learning yaitu Clustering yang dapat mengelompokkan pelanngan-pelanggan tersebut tanpa memperhatikan ketertarikan atau tidak.

# Dataset dan Data Exploration
Dataset dan Data Exploration sama dengan proyek yang lain Decision Tree pada link : https://github.com/Otniel113/DecisionTree

# Data Preprocessing
1. Handling Missing Value --> Drop
2. Scaling --> Standarisasi
3. Categorical Encoding --> Label Encoding
4. Feature Selection --> Korelasi <br>
![image](https://user-images.githubusercontent.com/57952404/152667655-e79adce6-3261-4974-8f80-d00496b3cfcf.png)
5. Scaling --> Normalisasi

# Modeling
Menggunakan K-Means Clustering yang dibangun tanpa library dengan nilai K=11 (didapatkan yang optimal). <br>

## Function yang digunakan
```python
#Mencari jarak antar titik menggunakan Euclidean Distance
def EuclideanDistance(titik1, titik2):
    jumlah = 0
    for i in range (0,5):
        hasil = (titik1[i+1] - titik2[i+1]) ** 2
        jumlah += hasil
    return np.sqrt(jumlah)

#Melakukan Perubahan Centroid setelah dihitung rata2 semua titik yang masuk cluster tersebut
def geserCentroid(cluster):
  new_centroid = [0.0]
  for i in range(0,5):
    jumlah = 0
    for j in range(0, len(cluster)-1):
      jumlah += cluster[j+1][i+1]
    new_titik = jumlah / (len(cluster))
    new_centroid.append(new_titik)
  return new_centroid

#Mencari apakah suatu nilai ada pada suatu array / list
def is_found(anggota, arr):
  for i in range(0,len(arr)):
    if (arr[i] == anggota):
      return True
  return False
```

## Melakukan inisialisasi Centroid secara acak dengan mengambil data yang dimiliki, dan memasukan centroid tersebut sebagai salah satu anggota sebuah cluster
```python
#Inisiasi Centroid dengan angka random berdasarkan banyak data
calon_sentroid = []    
acak = np.random.randint(0,len(model)-1)
calon_sentroid.append(acak)

#Melakukan inisiasi angka random sebanyak K dan menjamin tidak ada angka yang sama
for i in range(0,K-1):
  while True:
    acak = np.random.randint(0,len(model)-1)
    if not is_found(acak, calon_sentroid):
      calon_sentroid.append(acak)
      break

list_centroid = []     #Berisi koordinat centroid-centroid
list_cluster = []      #Berisi cluster (sebanyak K) dan menampung titik-titik pada cluster tersebut


for i in range(0,3*K):
  if (0 <= i <= K-1):
    list_centroid.append(model[calon_sentroid[i]])                    #Mengisi titik Centroid berdasarkan angka random (yang menunjukan indeks) di atas
  elif (K <= i <= 2*K-1):
    list_cluster.append(["Cluster" + str(i-K)])                       #Menambahkan nama "Cluster ke-n" di awal list_cluster
  else:
    list_cluster[i-2*K].append(list_centroid[i-2*K])                  #Memasukkan centroid sebagai anggota pertama masing-masing cluster

list_cluster
```

## Algoritma Clustering
```python
list_jarak = []

while True:                              #Melakukan loop sampai semua Centroid tidak bergeser lagi
  for i in range(0,len(model)):          #Melakukan loop untuk menghitung titik setiap data                                                         
    for j in range(0,K):                 #Melakukan loop untuk menghitung di titik i jarak terhadap centroid yang lain
      list_jarak.append(EuclideanDistance(list_centroid[j], model[i]))
    #endfor

    jarak_terdekat = min(list_jarak)              #Memilih jarak terpendek

    for j in range(0,K):                          #Mencari jarak terdekat dari nilai di atas ada pada cluster mana, lalu memasukannya ke dalam list_cluster
      if (list_jarak[j] == jarak_terdekat):
        list_cluster[j].append(model[i])
    #endfor
    

    list_jarak.clear()                           #Melakukan clear pada list_jarak yang akan dipakai lagi pada loop berikutnya

  list_new_centroid = []                         #Tempat calon Centroid baru
  for i in range(0,K):
    list_new_centroid.append(geserCentroid(list_cluster[i]))    #Melakukan pergeseran Centroid

  if (list_new_centroid == list_centroid):                      #Kondisi loop  utama (while True) berhenti kalau tidak ada pergerakan Centroid
    break
  else:
    list_centroid = list_new_centroid                           #Kalau masih berubah, centroid diubah
  
  for i in range(0,K):                                          #Output agar tahu di setiap perulangan banyak anggota masing-masing cluster
    print("Cluster",i, " sebanyak ", len(list_cluster[i]) - 1)
  print()

  for i in range(0,K):                                          #Melakukan clear list_cluster kecuali anggota pertama (yang berisi ket. Cluster1, Cluster2, dst)
    list_cluster[i] = list_cluster[i][:1]

for i in range(0, K):                                           #Output final
  print("\nCentroid",i," ada di titik ", list_centroid[i])
  print("Cluster",i," ada sebanyak ", len(list_cluster[i])-1)
```

# Evaluasi
Dengan menggunakan Silhoutte Score <br>
![image](https://user-images.githubusercontent.com/57952404/152667894-574bb758-24fd-4f0b-bd5e-bf9672b1e8be.png)

Di mana jika nilai Silhoutte Score semakin tinggi, maka semakin baik model dalam melakukan clustering

```python
def a_i(cluster1, titik_i):
  jumlah = 0
  for i in range(1,len(cluster1)):
    if (titik_i[0] != cluster1[i][0]):
      jumlah += EuclideanDistance(cluster1[i], titik_i)
  return (jumlah/(len(cluster1)-1))

def b_i(cluster_sisa, titik_i):
  jarak_antar_cluster = []
  for i in range(1,K):
    jumlah=0
    for j in range(1,len(cluster_sisa[i])-1):
      jumlah += EuclideanDistance(cluster_sisa[i][j], titik_i)
    jarak_antar_cluster.append(jumlah/len(cluster_sisa[i]))

  return min(jarak_antar_cluster)

def s_i(list_cluster):
  i = np.random.randint(1,len(list_cluster[0])-1)
  nilai_b_i = b_i(list_cluster, list_cluster[0][i])
  nilai_a_i = a_i(list_cluster[0], list_cluster[0][i])
  return (nilai_b_i - nilai_a_i) / (max(nilai_b_i, nilai_a_i))
```

# Eksperimen
Eksperimen dilakukan dengan mengubah nilai K dan didaptkan K dengan nilai Silhoutte Score tertinggi pada K=11
![image](https://user-images.githubusercontent.com/57952404/152667940-7898630d-cc25-4459-a7c5-19c2b847b4e3.png)

# Lebih Lengkap
Untuk lebih lengkap dapat dilihat di [Laporan.pdf](https://github.com/Otniel113/KMeansClustering/files/8009440/Laporan.pdf)

# Video Presentasi
Dengan klik gambar di bawah <br>
[![video](https://img.youtube.com/vi/sj3-uHRt-Ho/0.jpg)](https://youtu.be/sj3-uHRt-Ho)
