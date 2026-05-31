# EMNIST Letters — HOG + SVM Classification

> Klasifikasi karakter tulisan tangan A–Z menggunakan HOG Feature Extraction dan Support Vector Machine

---

## Deskripsi Proyek

Program ini mengimplementasikan sistem klasifikasi karakter tulisan tangan berbasis **Histogram of Oriented Gradients (HOG)** sebagai metode ekstraksi fitur dan **Support Vector Machine (SVM)** sebagai classifier. Dataset yang digunakan adalah **EMNIST Letters** yang berisi gambar tulisan tangan huruf A hingga Z.

---

## Struktur File

```
├── emnist_letters_hog_svm.ipynb   # Program utama
├── README.md                      # Dokumentasi proyek (file ini)
└── emnist-letters-train.csv       # Dataset (letakkan di folder yang sama)
```

---

## Dataset

| Info | Detail |
|------|--------|
| **Nama** | EMNIST Letters |
| **File** | `emnist-letters-train.csv` |
| **Sumber** | https://www.kaggle.com/datasets/crawford/emnist/data |
| **Total baris** | 88.800 |
| **Total kolom** | 785 (1 label + 784 piksel) |
| **Kelas** | 26 (huruf A–Z, label 1–26) |
| **Ukuran gambar** | 28 × 28 piksel |

### Format CSV
```
Kolom 0       → Label kelas (1=A, 2=B, ..., 26=Z)
Kolom 1–784   → Nilai piksel (0–255) gambar 28×28
```

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image notebook
```

| Library | Versi Minimum | Fungsi |
|---------|:------------:|--------|
| `numpy` | 1.21 | Manipulasi array |
| `pandas` | 1.3 | Load dan proses CSV |
| `matplotlib` | 3.4 | Visualisasi grafik |
| `seaborn` | 0.11 | Heatmap confusion matrix |
| `scikit-learn` | 0.24 | SVM, GridSearch, evaluasi |
| `scikit-image` | 0.18 | HOG feature extraction |

---

## Cara Menjalankan

1. **Clone atau download** file notebook
2. **Letakkan dataset** `emnist-letters-train.csv` di folder yang sama dengan notebook
3. **Buka Jupyter Notebook**
   ```bash
   jupyter notebook emnist_letters_hog_svm.ipynb
   ```
4. **Jalankan semua cell** secara berurutan: `Kernel → Restart & Run All`

> Jika path dataset berbeda, ubah variabel `LETTERS_CSV` di Cell 1.1

---

##  Alur Program

```
emnist-letters-train.csv
        │
        ▼
┌─────────────────────────────────────────────┐
│  1. DATASET PREPARATION                     │
│     • Load CSV (88.800 baris)               │
│     • Remap label: 1–26 → 0–25             │
│     • Sample 100/kelas → 2600 total         │
│     • Shuffle (random_state=42)             │
│     • Split 80/20 stratified                │
│       Train: 2080  |  Test: 520             │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  2. HOG FEATURE EXTRACTION                  │
│     • Bandingkan 6 konfigurasi HOG          │
│     • Parameter terbaik (Config E):         │
│       orientations    = 9                   │
│       pixels_per_cell = (4, 4)              │
│       cells_per_block = (3, 3)              │
│       transform_sqrt  = True                │
│       block_norm      = L2-Hys              │
│     • Output: 1764 fitur per gambar         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  3. SVM CLASSIFICATION                      │
│     • StandardScaler (normalisasi)          │
│     • GridSearchCV 5-fold:                  │
│       kernel : rbf, linear, poly            │
│       C      : 0.1, 1, 10, 100             │
│       gamma  : scale, auto, 0.001, 0.01    │
│     • Train model final (param terbaik)     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  4. EVALUATION                              │
│     • Accuracy, Precision, Recall, F1       │
│     • Confusion Matrix (Train/Test/LOOCV)   │
│     • LOOCV: 2600 iterasi                   │
│     • Per-class analysis                    │
└─────────────────────────────────────────────┘
```

---

## Dataset Preparation

### Balanced Sampling
Dari 88.800 data asli, diambil **100 sampel per kelas** secara acak:
- Total sampel yang digunakan: **2600** (26 kelas × 100)
- Setiap kelas memiliki jumlah sampel yang **sama persis**

### Shuffle
Dataset di-shuffle sebelum diproses untuk menghindari bias urutan:
```python
shuffle_idx = np.random.permutation(2600)
X_sampled   = X_sampled[shuffle_idx]
y_sampled   = y_sampled[shuffle_idx]
```

### Train/Test Split
| Set | Jumlah | Persentase | Per Kelas |
|-----|:------:|:----------:|:---------:|
| Training | 2080 | 80% | 80 sampel |
| Testing | 520 | 20% | 20 sampel |

Split dilakukan secara **stratified** — proporsi kelas tetap seimbang di kedua set.

### Preprocessing Gambar
EMNIST menyimpan gambar dalam format transposed. Setiap gambar di-reshape dan di-transpose sebelum diproses:
```python
def reshape_emnist(pixel_row):
    return pixel_row.reshape(28, 28).T
```

---

## HOG Feature Extraction

**HOG (Histogram of Oriented Gradients)** mendeteksi tepi dan tekstur gambar berdasarkan arah gradien piksel.

### Perbandingan 6 Konfigurasi HOG

| Konfigurasi | Orientations | Pixels/Cell | Cells/Block |
|-------------|:-----------:|:-----------:|:-----------:|
| Default | 8 | (8, 8) | (3, 3) |
| Config A | 9 | (7, 7) | (2, 2) |
| Config B | 9 | (4, 4) | (2, 2) |
| Config C | 12 | (7, 7) | (2, 2) |
| Config D | 9 | (7, 7) | (3, 3) |
| **Config E ★** | **9** | **(4, 4)** | **(3, 3)** |

### Parameter HOG Terbaik (Config E)

```python
BEST_HOG = {
    'orientations'    : 9,        # bin histogram arah gradien
    'pixels_per_cell' : (4, 4),   # ukuran sel (lebih kecil = lebih detail)
    'cells_per_block' : (3, 3),   # sel per blok normalisasi
    'transform_sqrt'  : True,     # gamma correction
    'block_norm'      : 'L2-Hys'  # metode normalisasi blok
}
```

**Dimensi fitur output:** `1764` per gambar

---

##  SVM Classification

### Normalisasi Fitur
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_hog)
X_test_scaled  = scaler.transform(X_test_hog)
```
> Scaler di-fit hanya pada training set untuk mencegah **data leakage**

### GridSearchCV Parameter Grid

| Parameter | Nilai yang Dicoba |
|-----------|------------------|
| `kernel` | `rbf`, `linear`, `poly` |
| `C` | `0.1`, `1`, `10`, `100` |
| `gamma` | `scale`, `auto`, `0.001`, `0.01` |
| `degree` (poly) | `2`, `3` |

- **CV Strategy:** 5-Fold Stratified Cross Validation
- **Scoring:** Accuracy

### Penjelasan Parameter SVM

| Parameter | Fungsi |
|-----------|--------|
| `kernel='rbf'` | Pemetaan non-linear, cocok untuk data kompleks |
| `C` | Regularisasi — kecil: margin lebar, besar: fit ketat |
| `gamma` | Jangkauan pengaruh support vector |

---

## Metode Evaluasi

### Metrik yang Digunakan

| Metrik | Rumus | Keterangan |
|--------|-------|------------|
| **Accuracy** | (TP+TN) / Total | Persentase prediksi benar keseluruhan |
| **Precision** | TP / (TP+FP) | Ketepatan prediksi per kelas |
| **Recall** | TP / (TP+FN) | Kelengkapan deteksi per kelas |
| **F1-Score** | 2×(P×R)/(P+R) | Harmonic mean Precision & Recall |

### LOOCV (Leave-One-Out Cross-Validation)

LOOCV dijalankan pada seluruh **2600 sampel**:

```
Iterasi 1    : Train[2–2600]    → Test[1]
Iterasi 2    : Train[1,3–2600]  → Test[2]
...
Iterasi 2600 : Train[1–2599]    → Test[2600]
```

- **Total iterasi:** 2600
- **Training per iterasi:** 2599 sampel
- **Testing per iterasi:** 1 sampel
- **Estimasi:** paling stabil dan tidak bias dibanding k-fold biasa

### Evaluasi yang Dilakukan

| Evaluasi | Dataset | Keterangan |
|----------|---------|------------|
| Train | 2080 sampel | Mengukur kemampuan belajar model |
| Test | 520 sampel | Mengukur generalisasi ke data baru |
| LOOCV | 2600 sampel | Estimasi performa paling reliable |

---

## Output yang Dihasilkan

| File | Isi |
|------|-----|
| `sample_images.png` | Visualisasi sampel gambar per kelas |
| `class_distribution.png` | Bar chart distribusi kelas |
| `hog_comparison.png` | Perbandingan 6 konfigurasi HOG |
| `hog_visualization.png` | Visualisasi fitur HOG per kelas |
| `gridsearch_heatmap.png` | Heatmap GridSearch C vs Gamma |
| `metrics_comparison.png` | Perbandingan metrik Train vs Test |
| `confusion_matrix_train.png` | Confusion matrix training set |
| `confusion_matrix_test.png` | Confusion matrix test set |
| `confusion_matrix_loocv.png` | Confusion matrix LOOCV |
| `per_class_metrics.png` | Metrik per kelas (4 subplot) |
| `predictions_correct.png` | Contoh prediksi benar |
| `predictions_wrong.png` | Contoh prediksi salah |

---

## Poin Penting

1. **Gambar EMNIST perlu di-transpose** — format penyimpanannya berbeda dari MNIST biasa
2. **StandardScaler hanya di-fit pada train set** — mencegah data leakage ke test set
3. **Pipeline pada LOOCV** — memastikan scaling dilakukan ulang di setiap fold
4. **Stratified split** — menjaga keseimbangan kelas di train dan test set
5. **GridSearchCV otomatis** — menemukan kombinasi parameter SVM terbaik tanpa manual tuning

---

## Informasi

| | |
|--|--|
| **Mata Kuliah** | Computer Vision / Machine Learning |
| **Dataset** | EMNIST Letters — Crawford (Kaggle) |
| **Metode** | HOG + SVM |
| **Bahasa** | Python 3 |
