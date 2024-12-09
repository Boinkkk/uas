import steramlit as st
import pandas as pd
# %% [markdown]
st.title("DATASET CHRONIC KIDNEY DISEASE")
# ### print DATASET

# %% [markdown]
# ### PENJELASAN CHRONIC KIDNEY DISEASE (PENYAKIT GINJAL KRONIS)
# Penyakit Ginjal Kronis (PGK) / CHRONIC KIDNEY DISEASE (CKD) ditandai dengan adanya kerusakan ginjal atau laju filtrasi glomerulus (GFR) yang diperkirakan kurang dari 60 mL/menit/1,73 m², yang bertahan selama 3 bulan atau lebih. PGK melibatkan kehilangan fungsi ginjal secara progresif, seringkali menyebabkan kebutuhan akan terapi pengganti ginjal, seperti dialisis atau transplantasi. Klasifikasi KDIGO CKD 2012 mempertimbangkan penyebab yang mendasarinya dan mengkategorikan PGK menjadi 6 tahap perkembangan dan 3 tahap proteinuria berdasarkan laju filtrasi glomerulus dan kadar albuminuria. Meskipun penyebab PGK bervariasi, proses penyakit tertentu menunjukkan pola yang serupa.
# 
# Implikasi PGK sangat luas—muncul dari berbagai proses penyakit dan mempengaruhi kesehatan kardiovaskular, fungsi kognitif, metabolisme tulang, anemia, tekanan darah, dan banyak indikator kesehatan lainnya. Deteksi dini PGK adalah langkah pertama dalam mengobatinya, dan berbagai metode untuk mengukur eGFR telah dijelaskan. Baik faktor risiko yang dapat dimodifikasi maupun yang tidak dapat dimodifikasi mempengaruhi perkembangan PGK. Manajemen PGK melibatkan penyesuaian dosis obat sesuai dengan eGFR pasien, mempersiapkan terapi pengganti ginjal, dan mengatasi penyebab yang dapat diubah untuk memperlambat perkembangan penyakit. Kegiatan ini meninjau etiologi, evaluasi, dan manajemen PGK, dengan menekankan peran penting tim perawatan kesehatan interprofesional dalam memberikan perawatan komprehensif. Pendekatan interprofesional berfokus pada faktor risiko yang dapat dimodifikasi dan tidak dapat dimodifikasi untuk mengelola dan mengurangi perkembangan penyakit. 
# 
# **sumber : [CKD](https://www.ncbi.nlm.nih.gov/books/NBK535404/)**

# %%
import pandas as pd


df = pd.read_csv('CSV_EXCEL/kidney_disease.csv')

#Rename column
col={'age': 'age',
     'bp': 'blood_pressure',
     'sg': 'specific_gravity',
     'al': 'albumin',
     'su': 'sugar',
     'rbc': 'red_blood_cells',
     'pc': 'pus_cell',
     'pcc': 'pus_cell_clumps',
     'ba': 'bacteria',
     'bgr': 'blood_glucose_random',
     'bu': 'blood_urea',
     'sc': 'serum_creatinine',
     'sod': 'sodium',
     'pot': 'potassium',
     'hemo': 'hemoglobin',
     'pcv': 'packed_cell_volume',
     'wc': 'white_blood_cell_count',
     'rc': 'red_blood_cell_count',
     'htn': 'hypertension',
     'dm': 'diabetes_mellitus',
     'cad': 'coronary_artery_disease',
     'appet': 'appetite',
     'pe': 'pedal_edema',
     'ane': 'anemia',
     'classification': 'class'}
df.rename(columns=col, inplace=True)
print(df)

df[['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']] = df[['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']].apply(pd.to_numeric, errors='coerce')


# %% [markdown]
# DATA SET TERSEBUT MEMPUNYAI 25 FITUR DAN 400 DATA RECORDS

# %% [markdown]
# ## DEKSRIPSI SETIAP FITUR

# %%
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
chronic_kidney_disease = fetch_ucirepo(id=336) 
  
# data (as pandas dataframes) 
X = chronic_kidney_disease.data.features 
y = chronic_kidney_disease.data.targets 
  
# variable information 
var = chronic_kidney_disease.variables
var

# %%
df.describe().round(3)

# %%
df.isna().sum()

# %%
df.isna().sum().sum()

# %% [markdown]
# ## FUNGSI ANGGREGATE
# fungsi agregat berperan penting dalam meringkas dan menyederhanakan kumpulan data yang besar menjadi informasi yang lebih mudah dipahami. Fungsi-fungsi ini melakukan perhitungan pada sekumpulan nilai dan menghasilkan satu nilai tunggal sebagai representasi dari keseluruhan data.
# 
# Contoh Fungsi Agregat yang Umum:
# 
# - **Mean (Rata-rata)**: Menghitung nilai rata-rata dari sekumpulan data. Misal, rata-rata nilai ujian siswa.
# - **Median**: Menentukan nilai tengah dari sekumpulan data yang telah diurutkan.
# - **Mode**: Menentukan nilai yang paling sering muncul dalam sekumpulan data.
# - **Range**: Menentukan selisih antara nilai terbesar dan terkecil dalam sekumpulan data.
# - **Variance**: Mengukur seberapa jauh penyebaran data dari rata-ratanya.
# - **Standard Deviation**: Akar kuadrat dari varians, memberikan ukuran penyebaran data yang lebih mudah diinterpretasikan.
# - **Sum**: Menghitung jumlah total dari semua nilai dalam sekumpulan data.
# - **Count**: Menghitung jumlah data dalam suatu kumpulan data.

# %% [markdown]
# ### FUNGSI AGGREGATE PADA FITUR BERTIPE NUMERIK

# %%
import pandas as pd

numerical_features = df.select_dtypes(include=['int64', 'float64']).copy()
numerical_features[['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']] = df[['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']].apply(pd.to_numeric, errors='coerce')

numerical_features.aggregate(['mean', 'median', 'std', 'max', 'min', 'sum','var']).round(3)


# %% [markdown]
# ### FUNGSI AGGREGATE DALAM FITUR BERTIPE CATEGORICAL

# %%
categorical_features = df.select_dtypes(include=['object']).copy()
mode_df = categorical_features.mode().iloc[0].to_frame().reset_index()
mode_df.columns = ['feature', 'mode']
mode_df

# %% [markdown]
# ## UJI KORELASI MENGGUNAKAN SPEARMAN DAN PEARSON
# Korelasi adalah cara yang digunakan untuk
# menentukan keeratan hubungan antara dua
# atau lebih variabel berbeda yang
# digambarkan dengan ukuran koefisien
# korelasi. 
# 
# Koefisien korelasi merupakan koefisien yang menggambarkan kedekatan
# hubungan antara dua atau lebih variabel.
# Besar kecilnya koefisien korelasi tidak
# menggambarkan hubungan sebab akibat
# antara dua variabel atau lebih, namun hanya
# menggambarkan hubungan linier antar
# variabelnya
# 
# Apabila koefisien korelasi bernilai
# positif dikatakan korelasi searah, dan
# sebaliknya jika koefisien korelasi bernilai
# negatif maka dikatakan korelasi tidak searah.
# Nilai koefisien korelasi terletak antara -1
# hingga 1. -1 berarti terdapat hubungan negatif
# sempurna (terbalik), 0 berarti tidak memiliki
# hubungan sama sekali, dan 1 berarti memiliki
# hubungan positif sempurna.
# 
# Jenis-jenis Uji Korelasi
# - Korelasi Pearson:
# Digunakan untuk data yang berdistribusi normal.
# Mengukur hubungan linier antara dua variabel.
# - Korelasi Spearman:
# Digunakan untuk data yang tidak berdistribusi normal atau data ordinal.
# Mengukur hubungan monotonik (tidak harus linier) antara dua variabel.
# 
# Sumber : [Korelasi](https://jurnal.untidar.ac.id/index.php/thetaomega/article/view/3552/1769)

# %%
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for column in categorical_features.columns:
    le = LabelEncoder()
    categorical_features[column] = le.fit_transform(categorical_features[column].astype(str))
    label_encoders[column] = le

encoded_df = pd.concat([numerical_features, categorical_features], axis=1)


corr_spearman = encoded_df.corr(method='spearman').round(3)

print("Spearman Correlation", corr_spearman)

print("KORELASI SETIAP FITUR DENGAN TARGET")

print(corr_spearman['class'].to_frame().drop(['class']))

# %% [markdown]
# ## RANKING RANK SETIAP KORELASI DENGAN TARGET

# %%
corr_spearman['class'].drop(['id']).sort_values(key=abs, ascending=False).to_frame().reset_index().drop([0], axis=0)


# %% [markdown]
# ## ANALISIS PERSEBARAN DATA

# %% [markdown]
# ### KESEIMBANGAN DATA ANTARA CKD DAN NOT CKD

# %%
import matplotlib.pyplot as plt

# Count the occurrences of each class
class_counts = df['class'].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 6))
plt.text(-1.2, 1, f'CKD: {class_counts["ckd"]}\nNot CKD: {class_counts["notckd"]}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
class_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Distribution of CKD and Not CKD')
plt.ylabel('')
plt.show()

# %% [markdown]
# ### FITUR AGE

# %%
import matplotlib.pyplot as plt
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']

df['class'] = df['class'].str.strip()
df['age_range'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

age_range_counts = df.groupby(['age_range', 'class']).size().unstack(fill_value=0)
print(age_range_counts)

age_range_counts.plot(kind='bar')
plt.title('Distribusi Data Rentang Fitur Age terhadap Target')
plt.xlabel('Age Range')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Classification')
plt.show()


# %% [markdown]
# Pada histogram diatas pasien dengan rentang umur **61 - 70** mempunyai klasifikasi CKD terbanyak disusul dengan rentang **51-60**, pada pasien dengan rentang umur **0-10** semua pasien dengan penyakit gagal ginjal termasuk dalam gagal ginjal kronis

# %% [markdown]
# ## FITUR BLOOD PRESSURE
# 
# Blood pressure, atau tekanan darah, adalah tekanan yang dihasilkan oleh darah ketika dipompa oleh jantung ke seluruh tubuh melalui pembuluh darah. Tekanan darah penting untuk menjaga aliran darah yang membawa oksigen dan nutrisi ke organ serta jaringan tubuh, serta mengeluarkan zat-zat sisa dari sel.
# 
# Blood pressure diukur dalam dua angka, yaitu:
# 
# 1. **Tekanan Sistolik**: Angka yang lebih tinggi, menunjukkan tekanan di arteri saat jantung berkontraksi atau memompa darah ke seluruh tubuh.
# 2. **Tekanan Diastolik**: Angka yang lebih rendah, menunjukkan tekanan di arteri saat jantung beristirahat di antara detak jantung.
# 
# Sebagai contoh, tekanan darah 120/80 mmHg berarti tekanan sistolik adalah 120 mmHg dan tekanan diastolik adalah 80 mmHg.
# 
# ### Kategori Tekanan Darah
# Menurut standar kesehatan umum, tekanan darah diklasifikasikan menjadi beberapa kategori:
# - **Hipotensi (tekanan darah rendah)**: Kurang dari 90 mmHg untuk sistolik dan diastolik kurang dari 60 mmHg
# - **Normal**: Tekanan sistolik kurang dari 120 mmHg dan diastolik kurang dari 80 mmHg.
# - **Prehipertensi (Pra-Hipertensi)**: Tekanan sistolik antara 120-129 mmHg dan diastolik kurang dari 80 mmHg.
# - **Hipertensi Tingkat 1**: Tekanan sistolik antara 130-139 mmHg atau diastolik antara 80-89 mmHg.
# - **Hipertensi Tingkat 2**: Tekanan sistolik 140 mmHg atau lebih, atau diastolik 90 mmHg atau lebih.
# - **Hipertensi Krisis**: Tekanan sistolik di atas 180 mmHg atau diastolik di atas 120 mmHg, yang memerlukan perawatan medis segera.

# %%
import matplotlib.pyplot as plt

# Group the data by 'classification' and 'blood_pressure'
bp_grouped = df.groupby(['blood_pressure', 'class']).size().unstack(fill_value=0)

# Plot the grouped data
bp_grouped.plot(kind='bar', figsize=(12, 8))
print(bp_grouped)

plt.title('Distribution of Blood Pressure (bp) by Classification')
plt.xlabel('Blood Pressure (bp)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Classification')
plt.show()

# %% [markdown]
# Dalam fitur data tersebut tidak dijelaskan apakah termasuk pada diastolik atau sistolik, namun dikarenakan adanya tekanan darah dengan nilai 180 kemungkinan tekanan darah tersebut adalah sistolik. dalam histogram diatas tekanan darah tidak mempunyai pengaruh yang signifikan terhadap penentuan CKD atau bukan, Namun semua data tekanan darah diatas 90 semuanya termasuk dalam gagal ginjal chornic namun dalam jumlah yang kecil, begitupun pada kasus pada kasus hipotensi dimana semuanya termasuk dalam CKD

# %% [markdown]
# ## FITUR SPECIFIC GRAVITY
# Specific gravity atau berat jenis dalam dunia kesehatan mengacu pada berat relatif suatu zat dibandingkan dengan berat air murni pada suhu tertentu. Dalam konteks medis, specific gravity paling sering diukur pada urine.
# 
# Pengukuran specific gravity urine dapat memberikan informasi berharga tentang konsentrasi zat terlarut dalam urine dan fungsi ginjal. Nilai specific gravity yang normal biasanya berkisar antara 1.003 hingga 1.030.

# %%
import matplotlib.pyplot as plt
# Group the data by 'specific_gravity' and 'class'
sg_grouped = df.groupby(['specific_gravity', 'class']).size().unstack(fill_value=0)

# Combine CKD and Not CKD counts into one DataFrame
combined_counts = pd.DataFrame({'CKD': sg_grouped['ckd'], 'Not CKD': sg_grouped['notckd']}).fillna(0)

# Plot the combined bar chart
combined_counts.plot(kind='bar', figsize=(12, 8))
plt.title('Distribution of Specific Gravity (sg) for CKD and Not CKD')
plt.xlabel('Specific Gravity')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Classification')
plt.show()


# %% [markdown]
# Pada Renal Chronic disease test massa jenis urine dapat menurun hingga 1.008 sampai 1.012 [[sg]](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/urine-specific-gravity) hal ini sesuai dengan histrogram yang dimana data dengan nilai 1.015 kebawah semuanya berupa gagal ginjal Kronik (CKD)
# 

# %% [markdown]
# ## Fitur Albumin
# 
# Albumin adalah jenis protein utama dalam darah yang dibuat oleh hati dan memiliki beberapa fungsi penting bagi tubuh. Albumin membantu menjaga keseimbangan cairan dalam darah dan jaringan tubuh, serta memiliki peran dalam transportasi zat-zat penting. Albumin umumnya diukur melalui tes darah dan sering kali digunakan sebagai indikator kesehatan fungsi hati, ginjal, dan status nutrisi seseorang.
# 
# Dalam konteks penyakit ginjal kronis (chronic kidney disease atau CKD), albuminuria (kandungan albumin dalam urine) adalah salah satu indikator penting untuk menilai tingkat kerusakan ginjal. Albuminuria merujuk pada kondisi di mana terdapat albumin dalam urine, yang menandakan bahwa ginjal mengalami kebocoran dan tidak mampu menyaring protein dengan efektif. Pada CKD, albuminuria dibagi menjadi lima kategori, yang membantu dalam menilai tingkat keparahan kerusakan ginjal dan risiko progresi penyakit. Berikut adalah penjelasan kategori albuminuria tersebut:
# 
# ### 1. **A1 - Normal hingga Sedikit Peningkatan Albuminuria**
#    - **Kadar Albumin dalam Urine**: <30 mg/g kreatinin
#    - **Makna**: Ini adalah rentang albuminuria normal atau mendekati normal, menunjukkan fungsi ginjal yang baik atau sedikit terpengaruh.
#    - **Risiko**: Pada tingkat ini, risiko kerusakan ginjal lebih rendah, dan fungsi ginjal kemungkinan besar masih cukup baik.
# 
# ### 2. **A2 - Mikroalbuminuria atau Moderat**
#    - **Kadar Albumin dalam Urine**: 30-300 mg/g kreatinin
#    - **Makna**: Albuminuria tingkat sedang, juga disebut mikroalbuminuria, yang menunjukkan adanya gangguan fungsi ginjal yang signifikan, meskipun belum berat.
#    - **Risiko**: Ini adalah tanda awal penyakit ginjal, terutama pada pasien diabetes atau hipertensi. Risiko kerusakan ginjal semakin tinggi dibandingkan kategori A1, dan sering kali memerlukan intervensi medis untuk mencegah progresi.
# 
# ### 3. **A3 - Makroalbuminuria atau Albuminuria Berat**
#    - **Kadar Albumin dalam Urine**: >300 mg/g kreatinin
#    - **Makna**: Ini adalah albuminuria berat (makroalbuminuria) yang menandakan kerusakan ginjal serius. Pada tahap ini, ginjal mengalami penurunan fungsi yang signifikan.
#    - **Risiko**: Risiko gagal ginjal atau progresi CKD ke stadium lanjut meningkat, dan pengobatan intensif sering kali diperlukan.
# 
# ### 4. **A4 - Albuminuria Sangat Berat**
#    - **Kadar Albumin dalam Urine**: >1000 mg/g kreatinin
#    - **Makna**: Albuminuria sangat berat, menunjukkan kerusakan ginjal yang sangat parah. Ginjal tidak mampu menyaring protein dengan baik, dan lebih banyak albumin yang terbuang melalui urine.
#    - **Risiko**: Pasien pada kategori ini berada pada risiko tinggi untuk gagal ginjal total dan kemungkinan memerlukan perawatan intensif, seperti dialisis.
# 
# ### 5. **A5 - Albuminuria Ekstrem**
#    - **Kadar Albumin dalam Urine**: >2000 mg/g kreatinin
#    - **Makna**: Ini adalah tingkat tertinggi albuminuria, menunjukkan ginjal yang hampir atau benar-benar mengalami kegagalan fungsi.
#    - **Risiko**: Kategori ini sangat mengindikasikan kebutuhan untuk perawatan ginjal jangka panjang, termasuk dialisis atau transplantasi ginjal.
# 
# ### Relevansi Kategori Albuminuria dalam CKD
# Pengkategorian albuminuria ini membantu dalam menentukan:
# 
# - **Tingkat keparahan CKD**: Semakin tinggi kategori albuminuria, semakin besar kerusakan ginjal.
# - **Risiko progresi penyakit**: Albuminuria tinggi sering dikaitkan dengan risiko lebih tinggi untuk CKD stadium lanjut atau gagal ginjal.
# - **Intervensi yang diperlukan**: Semakin tinggi albuminuria, semakin intensif pengobatan yang mungkin diperlukan, baik dalam bentuk obat-obatan, perubahan gaya hidup, atau terapi ginjal jangka panjang.
# 
# Pengelolaan albuminuria pada CKD melibatkan pemantauan rutin dan penanganan faktor risiko seperti tekanan darah tinggi, kadar gula darah pada pasien diabetes, serta perubahan gaya hidup untuk mencegah progresi penyakit.

# %%
# Map albumin values to categories
albumin_mapping = {0.0: 'A1', 1.0: 'A2', 2.0: 'A3', 3.0: 'A4', 4.0: 'A5', 5.0: 'A6'}
df['albumin_category'] = df['albumin'].map(albumin_mapping)

# Group the data by 'albumin_category' and 'class'
albumin_grouped = df.groupby(['albumin_category', 'class']).size().unstack(fill_value=0)

# Plot the grouped data
albumin_grouped.plot(kind='bar', figsize=(12, 8))
plt.title('Distribution of Albumin Categories by Classification')
plt.xlabel('Albumin Category')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Classification')
plt.show()

# %% [markdown]
# Pada diagram tersebut cukup menunjukkan adanya hubungan antara kadar albumin dengan termasuknya CKD atau not ckd, selain A1 orang dengan albuminuria A2 keatas semuanya termasuk dalam gagal ginjal kronis

# %% [markdown]
# ## FITUR SUGAR (GULA DARAH)
# Gula darah (glukosa darah) adalah konsentrasi glukosa dalam darah yang sangat penting untuk dijaga pada tingkat yang normal, terutama bagi pasien dengan gagal ginjal. Dalam konteks gagal ginjal atau penyakit ginjal kronis (CKD), pengelolaan gula darah menjadi sangat penting karena gula darah yang tidak terkontrol dapat memperburuk kondisi ginjal dan mempercepat progresi penyakit. Hal ini sangat relevan, terutama pada pasien diabetes, yang merupakan salah satu penyebab utama gagal ginjal.
# 
# Berikut adalah penjelasan mengenai pentingnya gula darah dalam konteks gagal ginjal:
# 
# ### 1. **Hubungan antara Diabetes dan Penyakit Ginjal**
#    - **Diabetes** adalah penyebab utama penyakit ginjal kronis. Sekitar 30-40% pasien diabetes tipe 1 dan tipe 2 mengalami nefropati diabetik, yaitu kerusakan ginjal yang disebabkan oleh tingginya kadar gula darah.
#    - Ketika kadar gula darah tinggi dalam waktu lama, glukosa darah berlebih dapat merusak pembuluh darah kecil di ginjal, yang berperan penting dalam proses penyaringan. Kerusakan ini mengganggu kemampuan ginjal untuk menyaring darah dengan benar, sehingga protein seperti albumin dapat bocor ke urine (albuminuria), yang menjadi salah satu tanda kerusakan ginjal.
# 
# ### 2. **Pengaruh Gagal Ginjal terhadap Pengaturan Gula Darah**
#    - Ketika ginjal mengalami penurunan fungsi, kemampuan tubuh untuk mengatur kadar gula darah menjadi terganggu. Hal ini terjadi karena ginjal yang sehat turut berperan dalam mengendalikan kadar glukosa dalam tubuh, termasuk pembuangan insulin dan pengaturan glukoneogenesis (produksi glukosa baru oleh tubuh).
#    - Pada pasien gagal ginjal, **insulin tidak dibersihkan dengan cepat dari tubuh**. Akibatnya, insulin bertahan lebih lama, meningkatkan risiko **hipoglikemia** (kadar gula darah rendah). Ini adalah situasi berbahaya yang dapat menyebabkan gejala serius seperti kejang atau bahkan kehilangan kesadaran.
# 
# ### Kategori pada Fitur "Sugar" dalam Dataset CKD
# Fitur "sugar" dalam dataset CKD ini dikategorikan ke dalam lima level, yang diberi nilai sebagai berikut:
# 
# - **0** - Tidak ada gula dalam urine (nilai normal).
# - **1** - Kadar gula urine rendah (sedikit gula terdeteksi).
# - **2** - Kadar gula urine sedang.
# - **3** - Kadar gula urine tinggi.
# - **4** - Kadar gula urine sangat tinggi.

# %%
import matplotlib.pyplot as plt

# Group the data by 'sugar' and 'class'
sugar_grouped = df.groupby(['sugar', 'class']).size().unstack(fill_value=0)

# Plot the grouped data
sugar_grouped.plot(kind='bar', figsize=(12, 8))
plt.title('Distribution of Sugar Levels by Classification')
plt.xlabel('Sugar Level')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Classification')
plt.show()

# %% [markdown]
# Pada histogram diatas tidak menunjukkan hubungan yang begitu signifikan, orang dengan sugar level diatas 1 semuanya mengalami gagal ginjal namun ini dalam jumlah yang sedikit, dan jumlah CKD dalam level sugar 0.0 juga sangat banyak

# %% [markdown]
# ## FITUR RBC (RED BLOAD CELL)
# **RBC** merujuk pada keberadaan **sel darah merah** (red blood cells) dalam urine pasien. Kehadiran sel darah merah dalam urine, yang disebut **hematuria**, merupakan indikator penting dalam diagnosa dan penilaian penyakit ginjal.
# 
# Pada orang yang sehat, ginjal menyaring darah tanpa membiarkan sel darah merah bocor ke dalam urine. Jadi, keberadaan sel darah merah dalam urine menunjukkan adanya masalah pada ginjal atau saluran kemih.
# Hematuria dapat menjadi tanda adanya kerusakan pada glomerulus ginjal (bagian ginjal yang menyaring darah), infeksi saluran kemih, batu ginjal, atau bahkan tumor. Hematuria juga bisa disebabkan oleh nefritis (peradangan pada ginjal) atau penyakit autoimun tertentu.
# 
# Berikut adalah penjelasan mengenai fitur **RBC** dalam konteks dataset CKD:
# 
# ### 1. **Kategori dalam Fitur RBC**
#    - Fitur **RBC** dalam dataset CKD biasanya diisi dengan nilai **"normal"** atau **"abnormal"**.
#    - **Normal**: Tidak ada atau sangat sedikit sel darah merah dalam urine, yang dianggap kondisi normal.
#    - **Abnormal**: Terdapat jumlah sel darah merah yang signifikan dalam urine, menunjukkan hematuria atau kebocoran darah ke dalam saluran kemih.

# %%
rbc_group = df.groupby(['red_blood_cells', 'class']).size().unstack(fill_value=0)

rbc_group.plot(kind='bar', figsize=(12,8))
plt.title("Data distribution Sugar")
plt.xlabel('Sugar')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Classification')

plt.show()

# %% [markdown]
# Dalam histogram tersebut RBC cukup berkaitan dengan class dengan ditandainya seluruh orang yang menderita CKD adalah orang yang mempunyai RBC abnormal namun ada beberapa orang dengan CKD mempunyai RBC normal

# %% [markdown]
# ## FITUR PUS CELL
# 
# **PC** atau **pus cell** merujuk pada keberadaan sel nanah (pus cells) dalam urine pasien. Kehadiran sel nanah dalam urine dikenal sebagai **piuria** dan sering kali mengindikasikan adanya infeksi atau peradangan dalam saluran kemih atau ginjal.
# 
# Sel nanah dalam urine adalah akumulasi dari sel darah putih yang merespons infeksi atau peradangan. Ginjal yang sehat biasanya tidak akan memiliki sel nanah dalam urine.
# Kehadiran sel nanah dalam urine (piuria) menunjukkan adanya respons imun tubuh terhadap infeksi atau kondisi inflamasi di sepanjang saluran kemih, termasuk ginjal, kandung kemih, atau uretra.
# 
# ### 1. **Kategori dalam Fitur Pus Cell (PC)**
#    - Dalam dataset CKD, fitur **pus cell** memiliki nilai kategorikal berupa **"normal"** atau **"abnormal"**.
#    - **Normal**: Tidak ada atau sangat sedikit sel nanah dalam urine, yang menunjukkan kondisi urine yang sehat.
#    - **Abnormal**: Kehadiran sel nanah dalam jumlah signifikan, yang dapat menunjukkan adanya infeksi atau peradangan.

# %%
import matplotlib.pyplot as plt

# Group the data by 'pus_cell' and 'class'
pus_cell_grouped = df.groupby(['pus_cell', 'class']).size().unstack(fill_value=0)

# Plot the pie chart for CKD
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

pus_cell_grouped['ckd'].plot(kind='pie', ax=axes[0], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
axes[0].set_title('Distribution of Pus Cell (PC) - CKD')
axes[0].set_ylabel('')

# Plot the pie chart for Not CKD
pus_cell_grouped['notckd'].plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
axes[1].set_title('Distribution of Pus Cell (PC) - Not CKD')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

# %% [markdown]
# Pada fitur Pus cell (PC) keterkaitan dengan fitur cukup terkait, orang yang Not CKD mempunyai Pus Cell Normal sedangkan orang dengan CKD mempunyai persentasi 38,9% mempunyai PC abnormal

# %% [markdown]
# ## FITUR PUS CELL CLUMPS (PCC)
# 
# fitur PCC merujuk pada keberadaan pus cell clumps atau kelompok sel nanah dalam urine. Fitur ini memberikan informasi tambahan tentang kondisi kesehatan ginjal pasien dan berhubungan dengan kemungkinan adanya infeksi atau peradangan di saluran kemih.
# 
# Pus cell clumps (kelompok sel nanah) dalam urine adalah tanda adanya infeksi atau peradangan yang lebih serius dalam saluran kemih atau ginjal. Jika kelompok sel nanah terdeteksi, ini menunjukkan bahwa tubuh merespons terhadap kondisi yang mungkin merugikan.
# Kondisi "notpresent" menunjukkan bahwa tidak ada indikasi infeksi atau masalah inflamasi yang signifikan, sementara "present" dapat menjadi tanda peringatan untuk evaluasi lebih lanjut.
# 
# ### 1. **Kategori dalam Fitur PCC**
# Dalam dataset CKD, fitur **PCC** berisi dua kategori:
# - **"notpresent"**: Menunjukkan bahwa tidak ada kelompok sel nanah yang terdeteksi dalam urine. Ini menunjukkan kondisi urine yang normal.
# - **"present"**: Menunjukkan bahwa kelompok sel nanah terdeteksi dalam urine, yang menunjukkan kemungkinan adanya infeksi atau peradangan.
# 

# %%
import matplotlib.pyplot as plt

# Group the data by 'pus_cell_clumps' and 'class'
pcc_grouped = df.groupby(['pus_cell_clumps', 'class']).size().unstack(fill_value=0)

# Plot the pie chart for CKD
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

pcc_grouped['ckd'].plot(kind='pie', ax=axes[0], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
axes[0].set_title('Distribution of Pus Cell Clumps (PCC) - CKD')
axes[0].set_ylabel('')

# Plot the pie chart for Not CKD
pcc_grouped['notckd'].plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
axes[1].set_title('Distribution of Pus Cell Clumps (PCC) - Not CKD')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

# %% [markdown]
# Pada diagram pie tersebut PCC dengan class tidak berkaitan yang ditandai dengan pasien Notckd yang semuanya mempunya pcc notpresent namun kebanyakan pasien CKD juga tidak mempunyai PCC oleh karena itu fitur PCC tidak terlalu berkaitan

# %% [markdown]
# ## FITUR BA (BACTERIAURIA)
# fitur **BA** mengacu pada keberadaan **bacterial** atau bakteri dalam urine pasien. Kehadiran bakteri dalam urine disebut **bakteriuria** dan sering kali menjadi indikator adanya infeksi saluran kemih atau kondisi medis lain yang memerlukan perhatian lebih lanjut.
# 
# Pada individu yang sehat, urine biasanya steril, artinya tidak mengandung bakteri. Kehadiran bakteri dalam urine dapat menunjukkan adanya infeksi saluran kemih (ISK) atau masalah lain dalam sistem kemih.
# Bakteriuria penting untuk diperhatikan, terutama pada pasien dengan penyakit ginjal kronis (CKD), karena infeksi saluran kemih yang tidak diobati dapat memperparah kerusakan ginjal dan mempercepat progresi penyakit ginjal.
# 
# ### 1. **Kategori dalam Fitur BA**
#    - Fitur **BA** memiliki dua nilai kategorikal:
#      - **"notpresent"**: Tidak ada bakteri yang terdeteksi dalam urine. Ini menunjukkan kondisi urine yang bebas dari infeksi bakteri.
#      - **"present"**: Bakteri terdeteksi dalam urine, yang sering kali menjadi indikasi adanya infeksi atau kontaminasi dalam saluran kemih.

# %%

bacteria_grouped = df.groupby(['bacteria', 'class']).size().unstack(fill_value=0)

# Plot the grouped data
bacteria_grouped.plot(kind='bar', figsize=(12, 8))
plt.title('Distribution of Bacteria by Classification')
plt.xlabel('Bacteria')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Classification')
plt.show()

# %% [markdown]
# Dalam histogram tersebut adanya bacteria tidak berkaitan erat dengan klasifikasi, dikarenakan pada bacteria yang present semuanya adalah ckd namun banyak juga ba yang not present termasuk dalam ckd

# %% [markdown]
# ## BLOOD GLUCOSE RANDOM ()
# fitur BGR mengacu pada Blood Glucose Random atau kadar glukosa darah acak. Fitur ini menunjukkan kadar gula dalam darah pasien pada waktu tertentu tanpa memperhatikan kapan terakhir kali mereka makan. Kadar glukosa darah acak adalah indikator penting, terutama pada pasien dengan penyakit ginjal, karena kadar glukosa darah yang tinggi sering dikaitkan dengan kondisi seperti diabetes mellitus, yang merupakan salah satu penyebab utama penyakit ginjal kronis.
# 
# Nilai Normal: Kadar glukosa darah acak yang normal umumnya berada di bawah 140 mg/dL. Nilai ini dapat bervariasi sedikit tergantung pada pedoman medis tertentu, tetapi umumnya kurang dari 140 mg/dL dianggap normal.
# 
# Pra-Diabetes atau Risiko Tinggi: Kadar glukosa darah acak antara 140–199 mg/dL sering dianggap sebagai tanda pra-diabetes atau risiko tinggi untuk diabetes.
# 
# Diabetes: Kadar BGR yang mencapai atau melebihi 200 mg/dL secara konsisten mungkin menunjukkan diabetes mellitus, yang memerlukan pengujian lebih lanjut untuk konfirmasi.
# 

# %%

plt.figure(figsize=(10, 6))

# CKD data
ckd_data = df[df['class'] == 'ckd']
plt.scatter(ckd_data['id'], ckd_data['blood_glucose_random'], color='red', label='CKD', alpha=0.6)

# Not CKD data
notckd_data = df[df['class'] == 'notckd']
plt.scatter(notckd_data['id'], notckd_data['blood_glucose_random'], color='blue', label='Not CKD', alpha=0.6)

# Add a line to separate the dense population
plt.axhline(y=150, color='green', linestyle='--', linewidth=2, label='Dense Population Threshold')

plt.title('Scatter Plot of ID vs Blood Glucose Random (blood_glucose_random)')
plt.xlabel('ID')
plt.ylabel('Blood Glucose Random (blood_glucose_random)')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Pada dataset tersebut seseorang dengan gula darah kurang lebih diatas 150 tidak adanya pasien NOTCKD, sedangkan dibawah 150 CKD DAN NOT CKD tersebar jumlahnya secara merata

# %% [markdown]
# ## BLOOD UREA
# fitur BU mengacu pada Blood Urea atau kadar urea dalam darah. Kadar urea dalam darah adalah salah satu parameter yang sering digunakan untuk menilai fungsi ginjal, karena ginjal yang sehat bertugas membuang urea dari darah melalui urine. Urea adalah produk sampingan dari metabolisme protein dalam tubuh, yang diproduksi di hati dan dikeluarkan melalui ginjal.
# 
# BU diukur dalam satuan mg/dL (miligram per desiliter).
# Nilai BU bervariasi tergantung pada kondisi kesehatan individu dan fungsi ginjalnya.
# Pada orang dewasa, kadar normal BU biasanya berada dalam kisaran 7–20 mg/dL, meskipun ini dapat sedikit berbeda tergantung pada sumber atau pedoman medis tertentu

# %%
plt.figure(figsize=(10, 6))

# CKD data
plt.scatter(df[df['class'] == 'ckd']['id'], df[df['class'] == 'ckd']['blood_urea'], color='red', label='CKD', alpha=0.6)

# Not CKD data
plt.scatter(df[df['class'] == 'notckd']['id'], df[df['class'] == 'notckd']['blood_urea'], color='blue', label='Not CKD', alpha=0.6)

plt.title('Scatter Plot of ID vs Blood Urea')
plt.xlabel('ID')
plt.ylabel('Blood Urea')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Pada dataset tersebut orang dengan not ckd mempunyai blood urea dibawah 50, sedangkan dengan orang dengan ckd banyak berada dibawah 50 namun ada beberapa yang diatas 50 mengindikasikan blood urea tidak normal

# %% [markdown]
# ## SERUM CREATININE
# Kreatinin adalah produk limbah metabolisme otot yang secara normal dibuang oleh ginjal melalui urine. Kadar kreatinin dalam darah merupakan salah satu indikator penting untuk menilai fungsi ginjal karena ginjal yang sehat biasanya mampu membuang kreatinin secara efektif dari darah.
# 
# SC diukur dalam satuan mg/dL (miligram per desiliter).
# Nilai kreatinin serum yang normal bervariasi tergantung pada usia, jenis kelamin, dan massa otot seseorang, namun kisaran normalnya adalah sekitar:
# 
# 0.6–1.2 mg/dL untuk pria dewasa
# 
# 0.5–1.1 mg/dL untuk wanita dewasa
# 
# Pada pasien dengan penyakit ginjal kronis, kadar SC biasanya meningkat karena ginjal yang rusak tidak dapat mengeluarkan kreatinin secara efisien.

# %%
import matplotlib.pyplot as plt

# Scatter plot for serum creatinine
plt.figure(figsize=(10, 6))

# CKD data
plt.scatter(df[df['class'] == 'ckd']['id'], df[df['class'] == 'ckd']['serum_creatinine'], color='red', label='CKD', alpha=0.6)

# Not CKD data
plt.scatter(df[df['class'] == 'notckd']['id'], df[df['class'] == 'notckd']['serum_creatinine'], color='blue', label='Not CKD', alpha=0.6)

plt.axhline(y=2, color='green', linestyle='--', linewidth=2, label='Dense Population Threshold')

plt.title('Scatter Plot of ID vs Serum Creatinine')
plt.xlabel('ID')
plt.ylabel('Serum Creatinine')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# dalam scatter plot diatas serum creatinine tidak menjadi alat ukur yang tepat dalam menyatakan pasien itu menunjukkan ckd atau tidak, namun bisa dipastikan seseorang yang mempunyai serum creatinine diatas 2 mempunyai gagal ginjal termasuk kronis 

# %% [markdown]
# ## FITUR SODIUM
# fitur sodium mengacu pada kadar natrium dalam darah. Natrium adalah elektrolit esensial yang berperan penting dalam menjaga keseimbangan cairan, fungsi otot, dan saraf dalam tubuh. Ginjal yang sehat memiliki peran penting dalam mengatur kadar natrium, sehingga kadar natrium dalam darah juga dapat memberikan indikasi kesehatan ginjal.
# 
# Sodium diukur dalam satuan mEq/L (miligram ekuivalen per liter).
# 
# Kadar natrium normal dalam darah berkisar antara 135–145 mEq/L.

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# CKD data
plt.scatter(df[df['class'] == 'ckd']['id'], df[df['class'] == 'ckd']['sodium'], color='red', label='CKD', alpha=0.6)

# Not CKD data
plt.scatter(df[df['class'] == 'notckd']['id'], df[df['class'] == 'notckd']['sodium'], color='blue', label='Not CKD', alpha=0.6)
plt.axhline(y=135, color='green', linestyle='--', linewidth=2, label='Lower Normal Range')
plt.axhline(y=145, color='orange', linestyle='--', linewidth=2, label='Upper Normal Range')
plt.title('Scatter Plot of ID vs Sodium')
plt.xlabel('ID')
plt.ylabel('Sodium')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Pada pasien CKD cenderung mempunyai kadar natrium rendah namun banyak juga yang memiliki kadar natrium normal, dan ada 1 data yang mempunyai kadar natrium tinggi yaitu 160

# %% [markdown]
# ## POTASSIUM 
# potassium merujuk pada kadar kalium dalam darah. Kalium adalah elektrolit esensial yang penting untuk fungsi seluler, terutama dalam menjaga irama jantung, fungsi otot, dan impuls saraf. Ginjal yang sehat bertanggung jawab untuk menjaga kadar kalium dalam darah tetap seimbang. Ketidakseimbangan kalium sering terjadi pada pasien dengan penyakit ginjal kronis (CKD), karena ginjal yang rusak mungkin tidak mampu mengatur kadar kalium dengan baik.
# 
# Potassium diukur dalam satuan mEq/L (miligram ekuivalen per liter).
# 
# Kadar kalium normal dalam darah berkisar antara 3.5–5.0 mEq/L.
# 
# Pada pasien dengan CKD, kadar kalium dapat berfluktuasi dan cenderung mengalami peningkatan karena ginjal tidak dapat membuang kalium secara efisien.

# %%
plt.figure(figsize=(10, 6))

# CKD data
plt.scatter(df[df['class'] == 'ckd']['id'], df[df['class'] == 'ckd']['potassium'], color='red', label='CKD', alpha=0.6)

# Not CKD data
plt.scatter(df[df['class'] == 'notckd']['id'], df[df['class'] == 'notckd']['potassium'], color='blue', label='Not CKD', alpha=0.6)

plt.axhline(y=5.0, color='green', linestyle='--', linewidth=2, label='Upper Normal Range')
plt.axhline(y=3.5, color='orange', linestyle='--', linewidth=2, label='Lower Normal Range')

plt.title('Scatter Plot of ID vs Potassium')
plt.xlabel('ID')
plt.ylabel('Potassium')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# scatter plot menunjukkan kalau kadar natrium tidak bisa menjadikan patokan apakah orang tersebut termasuk dalam ckd atau tidak, bisa dilihat persebaran data dalam range normal 3,5 - 5 diisi klasifikasi keduanya, hanya sedikit data yang keluar dari batas normal

# %% [markdown]
# ## HEMOGLOBIN
# fitur hemoglobin merujuk pada kadar hemoglobin dalam darah. Hemoglobin adalah protein dalam sel darah merah yang bertanggung jawab untuk mengangkut oksigen dari paru-paru ke seluruh tubuh dan membawa kembali karbon dioksida dari tubuh ke paru-paru untuk dikeluarkan. Kadar hemoglobin yang rendah sering terjadi pada pasien dengan penyakit ginjal kronis (CKD), karena ginjal yang rusak tidak mampu memproduksi hormon yang diperlukan untuk memproduksi sel darah merah dengan optimal.
# 
# Hemoglobin diukur dalam satuan g/dL (gram per desiliter).
# 
# Kadar hemoglobin normal bervariasi, tetapi umumnya berada dalam kisaran:
# - Pria dewasa: 13.8–17.2 g/dL
# - Wanita dewasa: 12.1–15.1 g/dL
# 
# Pada pasien dengan CKD, kadar hemoglobin sering kali lebih rendah dari kisaran normal, yang dikenal sebagai anemia.
# 

# %%
plt.figure(figsize=(10, 6))

# CKD data
plt.scatter(df[df['class'] == 'ckd']['id'], df[df['class'] == 'ckd']['hemoglobin'], color='red', label='CKD', alpha=0.6)

# Not CKD data
plt.scatter(df[df['class'] == 'notckd']['id'], df[df['class'] == 'notckd']['hemoglobin'], color='blue', label='Not CKD', alpha=0.6)

plt.axhline(y=12, color='green', linestyle='--', linewidth=2, label='Lower Normal Range')
plt.axhline(y=17, color='orange', linestyle='--', linewidth=2, label='Upper Normal Range')

plt.title('Scatter Plot of ID vs Hemoglobin')
plt.xlabel('ID')
plt.ylabel('Hemoglobin')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# pada fitur hemoglobin bisa menjadi patokan apakah orang tersebut apakah termasuk ckd atau bukan dikarenakan banyaknya data dengan ckd yang mempunyai kadar hemoglobin yang rendah. dan pada uji korelasi pun uji hemoglobin menduduki peringkat pertama

# %% [markdown]
# ## PACKED CELL VOLUME
# acked Cell Volume (PCV) atau volume sel terkemas adalah pengukuran persentase volume sel darah merah (eritrosit) dalam darah dibandingkan dengan volume total darah. PCV juga dikenal sebagai hematokrit. Pemeriksaan ini adalah indikator penting dalam menilai kondisi darah pasien, khususnya untuk melihat apakah mereka mengalami anemia, yang sering terjadi pada pasien dengan penyakit ginjal kronis (CKD).
# 
# PCV dinyatakan sebagai persentase (%).
# 
# Nilai PCV normal bervariasi berdasarkan jenis kelamin, yaitu:
# - Pria dewasa: sekitar 40–50%
# - Wanita dewasa: sekitar 35–45%
# 
# Nilai PCV di bawah kisaran normal menunjukkan anemia, sementara nilai di atas kisaran normal jarang terjadi secara alami dan lebih jarang pada pasien CKD

# %%
import matplotlib.pyplot as plt

# Define bins and labels for packed cell volume
bins = [0, 10, 20, 30, 40, 50, 60]
labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60']

# Create a new column for packed cell volume range
df['pcv_range'] = pd.cut(df['packed_cell_volume'].astype(float), bins=bins, labels=labels, right=False)

# Group the data by 'pcv_range' and 'class'
pcv_grouped = df.groupby(['pcv_range', 'class']).size().unstack(fill_value=0)

# Plot the grouped data
pcv_grouped.plot(kind='bar', figsize=(12, 8))
plt.title('Distribution of Packed Cell Volume (pcv) by Classification')
plt.xlabel('Packed Cell Volume Range')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Classification')
plt.show()

# %% [markdown]
# fitur pcv dapat dijadtikan sebagai patokan apakah orang tersebut mempunyai CKD atau tidak dikarenakan banyaknya data di bawah range normal 40% dan semuanya termasuk ckd, walaupun ada beberapa diatas 40% yang termasuk ckd namun jumlah datanya sedikit

# %% [markdown]
# ## WHITE BLOOD CELL COUNT (WC)
# tur WC mengacu pada White Cell Count atau jumlah sel darah putih dalam urin. Sel darah putih, juga dikenal sebagai leukosit, berperan penting dalam sistem kekebalan tubuh, membantu tubuh melawan infeksi dan penyakit. dihitung dalam satunnya cel/ml
# 
# Nilai normal berkisar kurang dari 5000/ml
# 

# %%
import matplotlib.pyplot as plt

# Scatter plot for white blood cell count
plt.figure(figsize=(10, 6))

# CKD data
plt.scatter(df[df['class'] == 'ckd']['id'], df[df['class'] == 'ckd']['white_blood_cell_count'], color='red', label='CKD', alpha=0.6)

# Not CKD data
plt.scatter(df[df['class'] == 'notckd']['id'], df[df['class'] == 'notckd']['white_blood_cell_count'], color='blue', label='Not CKD', alpha=0.6)

# Add a line to indicate the threshold
plt.axhline(y=5000, color='green', linestyle='--', linewidth=2, label='Threshold 5000')

plt.title('Scatter Plot of ID vs White Blood Cell Count')
plt.xlabel('ID')
plt.ylabel('White Blood Cell Count')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# fitur wc tidak bisa menjadi patokan untuk klasifikasi ckd atau tidak dikarenakan data yang tersebar cukup rata

# %% [markdown]
# ## FIRUT RED BLOOD COUNT
# fitur RC merujuk pada Red Cell Count atau jumlah sel darah merah dalam urin. Sel darah merah, juga dikenal sebagai eritrosit, memiliki peran penting dalam membawa oksigen dari paru-paru ke seluruh tubuh dan mengangkut karbon dioksida kembali ke paru-paru untuk dikeluarkan.
# 
# - 0 sel/µL: Menunjukkan bahwa tidak ada sel darah merah yang terdeteksi dalam urin.
# - 1-5 sel/µL: Masih dianggap dalam batas normal, meskipun kehadiran sel darah merah ini harus diperhatikan dalam konteks kesehatan pasien.
# - Lebih dari 5 sel/µL: Dapat mengindikasikan adanya hematuria, yang mungkin disebabkan oleh berbagai kondisi, termasuk infeksi saluran kemih, batu ginjal, atau masalah lain yang lebih serius pada ginjal atau saluran kemih.
# 

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# CKD data
plt.scatter(df[df['class'] == 'ckd']['id'], df[df['class'] == 'ckd']['red_blood_cell_count'], color='red', label='CKD', alpha=0.6)

# Not CKD data
plt.scatter(df[df['class'] == 'notckd']['id'], df[df['class'] == 'notckd']['red_blood_cell_count'], color='blue', label='Not CKD', alpha=0.6)

# Add lines to indicate the thresholds
plt.axhline(y=5, color='green', linestyle='--', linewidth=2, label='Upper Normal Range')
plt.axhline(y=1, color='orange', linestyle='--', linewidth=2, label='Lower Normal Range')

plt.title('Scatter Plot of ID vs Red Blood Cell Count')
plt.xlabel('ID')
plt.ylabel('Red Blood Cell Count')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ada keanahan pada data, pada data dengan orang ckd justru kehadiran cell darah merah dalam urin lebih sedikit daripada seseorang dengan not ckd

# %% [markdown]
# ## FITUR HTN
# fitur HTN merujuk pada Hypertension atau tekanan darah tinggi. Fitur ini menunjukkan apakah seorang pasien memiliki riwayat hipertensi, yang merupakan faktor risiko signifikan untuk perkembangan dan progresi penyakit ginjal kronis.
# 
# Dalam dataset CKD, fitur HTN biasanya dikategorikan menjadi dua nilai:
# 
# - Yes: Menunjukkan bahwa pasien memiliki riwayat hipertensi.
# - No: Menunjukkan bahwa pasien tidak memiliki riwayat hipertensi

# %%
import matplotlib.pyplot as plt

# Group the data by 'hypertension' and 'class'
htn_grouped = df.groupby(['hypertension', 'class']).size().unstack(fill_value=0)

# Plot the grouped data
htn_grouped.plot(kind='bar', figsize=(12, 8), color=['#ff9999','#66b3ff'])
plt.title('Distribution of Hypertension (htn) by Classification')
plt.xlabel('Hypertension')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Classification')
plt.show()

# %% [markdown]
# Semua yang hipertensi pasti memiliki klasifikasi ckd, dan semua not ckd tidak mempunyai hipertensi, namun ada beberapa orang dengan ckd tidak memiliki hipertensi

# %% [markdown]
# ## DIABETES MELLITUS
# Diabetes Mellitus adalah suatu kondisi medis di mana tubuh tidak dapat memproduksi cukup insulin atau tidak dapat menggunakan insulin secara efektif. Ini menyebabkan peningkatan kadar glukosa (gula) dalam darah, yang dapat berakibat serius pada berbagai sistem tubuh jika tidak dikelola dengan bai
# 
# Dalam dataset CKD, fitur DM biasanya dikategorikan menjadi dua nilai:
# 
# - Yes: Menunjukkan bahwa pasien memiliki riwayat diabetes mellitus.
# - No: Menunjukkan bahwa pasien tidak memiliki riwayat diabetes mellitus

# %%
import matplotlib.pyplot as plt

# Clean up the 'diabetes_mellitus' column
df['diabetes_mellitus'] = df['diabetes_mellitus'].str.strip()

# Group the data by 'diabetes_mellitus' and 'class'
dm_grouped = df.groupby(['diabetes_mellitus', 'class']).size().unstack(fill_value=0)

# Plot the grouped data
dm_grouped.plot(kind='bar', figsize=(12, 8), color=['#ff9999','#66b3ff'])
plt.title('Distribution of Diabetes Mellitus (dm) by Classification')
plt.xlabel('Diabetes Mellitus')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Classification')
plt.show()

# %% [markdown]
# dalam histrogram tersebut bisa disimpulkan semua record yang mempunyai dm yes semuanya termasuk dalam ckd, dan begitu sebaliknya, namun ada beberapa ckd tidak mempunyai dm dan jumlahnya cukup banyak. oleh karena alasan itu dm tidak mempunyai ikatan erat

# %% [markdown]
# ## Coronary Artery Disease
# 

# %%
import matplotlib.pyplot as plt

# Count the occurrences of each category in the 'coronary_artery_disease' column
df['coronary_artery_disease'] = df['coronary_artery_disease'].str.strip()
cad_grouped = df.groupby(['coronary_artery_disease', 'class']).size().unstack(fill_value=0)

# Plot the histogram
cad_grouped.plot(kind='bar',figsize=(12,8), color=['#ff9999','#66b3ff'])
plt.title('Distribution of Coronary Artery Disease')
plt.xlabel('Coronary Artery Disease')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# %% [markdown]
# histrogram tersebut sangat tidak menentukan seseorang memiliki ckd atau tidak, namum semua orang yang positif CAD semuanya positif CKD

# %% [markdown]
# ## Appetite
# APPET merujuk pada Appetite, yang mengindikasikan kondisi nafsu makan pasien. Fitur ini penting untuk menilai status gizi dan kesehatan secara keseluruhan pasien, terutama pada mereka yang menderita penyakit ginjal kronis.
# 
# Nafsu makan adalah keinginan atau motivasi untuk makan. Perubahan dalam nafsu makan dapat menjadi indikator kesehatan yang signifikan. Pada pasien dengan penyakit ginjal, nafsu makan yang menurun sering kali terjadi akibat berbagai faktor, termasuk gejala penyakit itu sendiri, efek samping pengobatan, atau perubahan metabolisme.
# 
# Dalam dataset CKD, fitur APPET biasanya dikategorikan menjadi dua nilai:
# 
# - Good: Menunjukkan bahwa pasien memiliki nafsu makan yang baik.
# - Poor: Menunjukkan bahwa pasien memiliki nafsu makan yang buruk.

# %%
import matplotlib.pyplot as plt

# Group the data by 'appetite' and 'class'
appetite_grouped = df.groupby(['appetite', 'class']).size().unstack(fill_value=0)

# Plot the grouped data
appetite_grouped.plot(kind='bar', figsize=(12, 8), color=['#ff9999','#66b3ff'])
plt.title('Distribution of Appetite by Classification')
plt.xlabel('Appetite')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Classification')
plt.show()

# %% [markdown]
# pada fitur Appetite tidak bisa menjadi patokan pengklasifikasian, dikarenakan penyabaran data ckd di kedua value poor dan good cukup banyak, oleh karena itu fitur ini keterkaitannya lemah

# %% [markdown]
# ## PEDAL EDEMA
# fitur PE merujuk pada Pedal Edema, yang menunjukkan adanya pembengkakan di bagian kaki atau pergelangan kaki. Pedal edema adalah salah satu tanda klinis yang umum ditemukan pada pasien dengan penyakit ginjal kronis dan dapat menjadi indikator penting untuk menilai kondisi kesehatan pasien.
# 
# Pedal edema adalah akumulasi cairan di jaringan di sekitar kaki dan pergelangan kaki, yang dapat menyebabkan pembengkakan. Hal ini terjadi akibat ketidakseimbangan dalam tekanan osmotik atau volume cairan tubuh, dan dapat disebabkan oleh berbagai kondisi medis, termasuk gagal ginjal.
# 
# Dalam dataset CKD, fitur PE biasanya dikategorikan menjadi dua nilai:
# 
# - Yes: Menunjukkan bahwa pasien mengalami pedal edema (pembengkakan di kaki).
# - No: Menunjukkan bahwa pasien tidak mengalami pedal edema.

# %%
import matplotlib.pyplot as plt

# Group the data by 'pedal_edema' and 'class'
pe_grouped = df.groupby(['pedal_edema', 'class']).size().unstack(fill_value=0)

# Plot the grouped data
pe_grouped.plot(kind='bar', figsize=(12, 8), color=['#ff9999','#66b3ff'])
plt.title('Distribution of Pedal Edema by Classification')
plt.xlabel('Pedal Edema')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Classification')
plt.show()

# %% [markdown]
# pada fitur Pedal edema tidak bisa menjadi patokan pengklasifikasian, dikarenakan penyabaran data ckd di kedua value no dan yes cukup banyak, oleh karena itu fitur ini keterkaitannya lemah

# %% [markdown]
# ## Anemia
# 
# Anemia adalah kondisi medis di mana jumlah sel darah merah atau konsentrasi hemoglobin dalam darah menurun. Hemoglobin adalah protein dalam sel darah merah yang bertanggung jawab untuk mengangkut oksigen ke seluruh tubuh. Ketika kadar hemoglobin rendah, tubuh mungkin tidak mendapatkan cukup oksigen, yang dapat menyebabkan kelelahan dan berbagai gejala lainnya.
# 
# Dalam dataset CKD, fitur ANEMIA biasanya dikategorikan menjadi dua nilai:
# 
# - Yes: Menunjukkan bahwa pasien menderita anemia.
# - No: Menunjukkan bahwa pasien tidak menderita anemia.
# 

# %%
import matplotlib.pyplot as plt

# Group the data by 'anemia' and 'class'
anemia_grouped = df.groupby(['anemia', 'class']).size().unstack(fill_value=0)

# Plot the grouped data
anemia_grouped.plot(kind='bar', figsize=(12, 8), color=['#ff9999','#66b3ff'])
plt.title('Distribution of Anemia by Classification')
plt.xlabel('Anemia')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Classification')
plt.show()

# %% [markdown]
# pada fitur anemia tidak bisa menjadi patokan pengklasifikasian, dikarenakan penyabaran data ckd di kedua value no dan yes cukup banyak, oleh karena itu fitur ini keterkaitannya lemah

# %% [markdown]
# ## IDENTIFIKASI KUALITAS DATA

# %% [markdown]
# ### PEMERIKSAAN NILAI NULL

# %%
print("Total Null Disetiap Column", pd.DataFrame(df.isna().sum(), columns=['Total']))
print(f"Total Semua Missing Value Setiap Column : {df.isna().sum().sum()}" )


# %% [markdown]
# ## KESEIMBANGAN DATA

# %%
import matplotlib.pyplot as plt

class_counts = df['class'].value_counts()
print(class_counts.to_frame())

plt.figure(figsize=(8, 6))
plt.text(-1.2, 1, f'CKD: {class_counts["ckd"]}\nNot CKD: {class_counts["notckd"]}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
class_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Distribution of CKD and Not CKD')
plt.ylabel('')
plt.show()

# %% [markdown]
# ## DETEKSI OUTLIER

# %%
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

numerical_features = df.select_dtypes(include=['float64', 'int64']).drop(columns=['id'])

numerical_features.fillna(numerical_features.mean(), inplace=True)

categorical_features = df.select_dtypes(include=['object', 'category']).drop(columns=['class'])
label_encoders = {}
for column in categorical_features.columns:
    le = LabelEncoder()
    categorical_features[column] = le.fit_transform(categorical_features[column].astype(str))
    label_encoders[column] = le
    
combined_features = pd.concat([numerical_features, categorical_features], axis=1)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.075)
lof_scores = lof.fit_predict(combined_features)
lof_negative_outlier_factor = lof.negative_outlier_factor_

df['lof_score'] = lof_negative_outlier_factor

plt.figure(figsize=(10, 6))
colors = ['blue' if score == 1 else 'red' for score in lof_scores]
plt.scatter(df.index, df['lof_score'], c=colors, alpha=0.6)
plt.axhline(y=-1.9, color='green', linestyle='--', linewidth=2, label='Batasan Outlier')
plt.title('LOF Scores for CKD Dataset')
plt.xlabel('Index')
plt.ylabel('LOF Score')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # PREPROCESSING

# %% [markdown]
# ## IMPUTE MISSING VALUE WITH MEAN

# %%
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

categorical_cols = df.select_dtypes(include=['object', 'category']).columns
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0])) #(apply >= function each column) fill na then fill with mode (each column)

# df.drop(columns=['age_range', 'albumin_category', 'pcv_range', 'lof_score'], inplace=True)

print(df)
print(df.isna().sum())

# %% [markdown]
# ## CLEANING OUTLIER

# %%
# df['lof_score'] = lof_scores
# df = df[lof_scores != -1] -> pick lof_score only -1
print(df)
df.drop(columns=['lof_score'])

# %% [markdown]
# ## SCALING USING MINMAX

# %%
from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

columns_to_scale = df.select_dtypes(include=['float64', 'int64']).columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


print(df.drop(columns=['lof_score']))

# %%
numerical_features = df.select_dtypes(include=['float64', 'int64']).drop(columns=['id'])

numerical_features.fillna(numerical_features.mean(), inplace=True)

categorical_features = df.select_dtypes(include=['object', 'category']).drop(columns=['class'])
label_encoders = {}
for column in categorical_features.columns:
    le = LabelEncoder()
    categorical_features[column] = le.fit_transform(categorical_features[column].astype(str))
    label_encoders[column] = le
    
df = pd.concat([numerical_features, categorical_features, df['class']], axis=1)

# %%
df

# %% [markdown]
# # SPLITTING DATASET

# %%
from sklearn.model_selection import train_test_split

df_clean = df.drop(columns = ['age_range', 'albumin_category', 'pcv_range', 'lof_score'])



X = df_clean.drop(columns=['class'])
y = df['class']

y = y.str.strip()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42 )

print("Data Train X", X_train)
print("Data Train y", y_train.to_frame())
print("Data Test X", X_test)
print("Data test y", y_test.to_frame())

# %% [markdown]
# ### CHECK IMBALANCE IN DATA TRAINING

# %%
import matplotlib.pyplot as plt

# Count the occurrences of each class in y_train
class_counts_train = y_train.value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 6))
plt.text(-1.2, 1, f'CKD: {class_counts_train["ckd"]}\nNot CKD: {class_counts_train["notckd"]}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
class_counts_train.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Distribution of CKD and Not CKD in Training Data')
plt.ylabel('')
plt.show()

# %% [markdown]
# ## MENGATSI IMBALANCING DATA DENGAN MENGGUNAKAN METODE SMOTE

# %% [markdown]
# SMOTE (Synthetic Minority Oversampling Technique) adalah teknik statistik yang digunakan untuk meningkatkan jumlah kasus dalam kumpulan data yang tidak seimbang. SMOTE merupakan pengembangan dari metode oversampling yang membangkitkan sampel baru dari kelas minoritas. 
# SMOTE bekerja dengan cara:
# Memilih sampel secara acak dari kelas minoritas
# Menemukan K-Nearest Neighbor dari sampel yang dipilih
# Menghubungkan sampel yang dipilih ke masing-masing tetangganya menggunakan garis lurus 
# SMOTE menghasilkan instans baru dari kasus minoritas yang ada, bukan hanya salinan dari mereka. Pendekatan ini meningkatkan fitur yang tersedia untuk setiap kelas dan membuat sampel lebih umum

# %%
from imblearn.over_sampling import SMOTE

X_smote, y_smote = SMOTE().fit_resample(X_train, y_train)

class_counts_train = y_smote.value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 6))
plt.text(-1.2, 1, f'CKD: {class_counts_train["ckd"]}\nNot CKD: {class_counts_train["notckd"]}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
class_counts_train.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Distribution of CKD and Not CKD in Training Data')
plt.ylabel('')
plt.show()




# %% [markdown]
# ## MODELLING DATA MENGGUNAKAN NAIVE BAYES

# %%
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_smote, y_smote)

predict_gaussian = gaussian.predict(X_test)

result_gaussian = pd.concat([pd.DataFrame(X_test).reset_index(), pd.DataFrame(predict_gaussian)], axis=1)
print(result_gaussian)

# %% [markdown]
# ## EVALUASI

# %%
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report

precisionScore = precision_score(predict_gaussian, y_test, average='micro')
recallScore = recall_score(predict_gaussian, y_test, average='micro')
f1Score = f1_score(predict_gaussian, y_test, average='micro')
accuracyScore = accuracy_score(predict_gaussian, y_test)

sum_accuracy = pd.DataFrame.from_dict({
    'Precission Score' : [precisionScore],
    'Recall Score' : [recallScore],
    'F1 Score' : [f1Score],
    'Accuracy Score' : [accuracyScore], 
})

print(classification_report(predict_gaussian, y_test))
print(sum_accuracy)




# %%
import seaborn as sns
confussionMatrix = confusion_matrix(y_test, predict_gaussian, labels=['ckd', 'notckd'])

plt.subplots(figsize=(10, 8))
sns.heatmap(confussionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=['CKD', 'Not CKD'], yticklabels=['CKD', 'Not CKD'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# %% [markdown]
# ## MEMBANDINGKAN NAIVE BAYES DENGAN ALGORITMA LAIN

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


accuracys = []
method_list = [KNeighborsClassifier(), DecisionTreeClassifier()]

def classify(method):
    clf = method
    clf.fit(X_smote, y_smote)
    result = clf.predict(X_test)
    acc = accuracy_score(result, y_test)
    print(str(method), acc)

for method in method_list:
    classify(method)
    
    
    
    

# %% [markdown]
# ## CLASIFIKASI MENGGUNAKAN NEURAL NETWORK

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
# Input Layer = 25 Feature
# 3 HIDDEN LAYER 
# Output layer = 2 Class

class Model(nn.Module):
    def __init__(self, in_features = 24, h1 = 8, h2 = 9, h3 = 10, out_features = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2,h3)
        self.out = nn.Linear(h3, out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        
        return x
    

# %%
## RANDOMZIER

torch.manual_seed = 30
model = Model()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
nn_df = df

# %%
y_smote = y_smote.replace('ckd', 0)
y_smote = y_smote.replace('notckd', 1)

y_test = y_test.replace('ckd', 0)
y_test = y_test.replace('notckd', 1)

# %%
# to numpy arr

import numpy as np

X = X_smote.values
y = y_smote.values

X_test = X_test.values
y_test = y_test.values

# %%
# To float tensor (X)

X_train = torch.FloatTensor(X)
X_test = torch.FloatTensor(X_test)

# To float tensor (Y)

y_train = torch.LongTensor(y)
y_test = torch.LongTensor(y_test)

print(pd.DataFrame(X_train))

# %%
#Set measure error of the model, how far prediction of data using Adam Optimezer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# %%
# Train model
# Epochs -> how many turn data to fit to the model

epochs =  100
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)

    losses.append(loss.detach().numpy())
    
    if i % 10 == 0:
        print(f'Epoch : {i} and loss: {loss}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    



# %%
plt.plot(range(epochs), losses)
plt.ylabel('Loss/Error')
plt.xlabel('Epoch')
plt.show()

# %%
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)

# %%
loss

# %%
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        
        if y_test[i] == 0:
            x = 'ckd'
        else:
            x = 'notckd'
        
        print(f'{i + 1}) {str(y_val)} \t {x} \t\t {y_val.argmax().item()}')
        
        if y_val.argmax().item() == y_test[i]:
            correct += 1
    print(f'Got Correct {correct}')
         


