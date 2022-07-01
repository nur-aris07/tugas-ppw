#!/usr/bin/env python
# coding: utf-8

# # Topic Modelling mengunakan Latent Semantic Analysis(LSA) dan Latent directlet Allocation(LDA)

# ## Import Library
# 
# ### Library yang digunakan
# 
# - **Pandas**\
#     Manipulasi dan membaca data dengan bentuk tabel. 
# 
# - **matplotlib**\
#     Memvisualisasi data.
# 
# - **PySastrawi**\
#     Melakukan text processing pada data teks.
# 
# - **scikit-learn**\
#     Menghitung TF dan TF-IDF.
# 
# - **WordCloud**\
#     Memvisualisasi kata yang paling sering muncul. 

# In[1]:


# import library yang dibutuhkan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from wordcloud import WordCloud

get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')

# Proses Pre-prosesing text
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# membuat dokument DTM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation


# In[2]:


# Melakukan setting jumlah kolom maksimal pada output
pd.options.display.max_columns = 10


# ## Data set Jurnal

# Dataset : [DataSet](https://github.com/egi-190137/topic-modelling-sklearn/blob/main/contents/dataset_pta.csv)
# 
# Dataset yang digunakan adalah hasil dari proses crawling judul dan abstraksi jurnal-jurnal yang juga telah terdapat nilai kolom bidang minat dan tanpa ada duplikasi data.

# In[3]:


df = pd.read_csv('dataset_pta.csv')


# In[4]:


df.head()


# Dalam program ini hanya menggunakan data pada kolom 'judul'. Untuk mengambil kolom 'judul' saja dapat dilakukan dengan inisialisasi ulang df dengan df[['judul']] 

# In[5]:


df = df[['judul']]
df.head()


# ## Pre-processing Data

# Terdapat beberapa tahapan dalam melakukan Pre-processing data, diantaranya *case folding* (Mengubah teks menjadi *lower case*), menghapus angka dan tanda baca, menghapus white space dan *stopword removal*. Semua tahapan *pre-processing* tersebut saya masukkan ke dalam fungsi clean_text, kemudian saya aplikasikan pada data judul pada dataframe dengan method **.apply(clean_text)**. 
# 
# Untuk menghapus stopword saya menggunakan library **PySastrawi**, karena **PySastrawi** memiliki list stopword bahasa indonesia yang lebih lengkap daripada library **nltk**.
# 
# Pada Library **PySastrawi** penghapusan stopword dilakukan dengan membuata objek StopWordRemoverFactory, kemudian buat objek stopword remover dengan method create_stop_word_remover. Objek stopword remover memiliki method remove yang dapat digunakan untuk menghapus stopword dalam sebuah kalimat dengan memasukkan string ke dalam parameter method remove.  

# In[6]:


factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()


# In[7]:


def clean_text(text):
  # Mengubah ke lowercase
  cleaned_text = text.lower()
  # Menghilangkan angka
  cleaned_text = re.sub(r"\d+", "", cleaned_text)
  # menghilangkan white space
  cleaned_text = cleaned_text.strip()
  # menghilangkan tanda baca
  cleaned_text = cleaned_text.translate(str.maketrans("","",string.punctuation))
  # menghilangkan stopword
  cleaned_text = stopword.remove(cleaned_text)
  return cleaned_text
  


# In[8]:


df['cleaned_judul'] = df['judul'].apply(clean_text)


# ### Hasil pre-processing

# In[9]:


df.head()


# ### Mengganti Column __cleaned_judul__ dengan __judul__ 

# In[10]:


df.drop(['judul'],axis=1,inplace=True)


# In[11]:


df.columns = ['judul']


# In[12]:


df.head()


# ### Contoh judul yang telah di lakukan *pre-processing*

# In[13]:


df['judul'][0]


# ## Ekstraksi fitur dan membuat Document Term Matrix (DTM)
# 
# Perhitungan Latent Semantic Analysis(LSA) hanya menggunakan data TF-IDF. Maka tidak perlu menghitung nilai TF. Meghitung nilai TF-IDF dapat dilakukan dengan membuat objek dari kelas TfidfVectorizer yang disediakan library scikit-learn.
# 
# Rumus Term Frequency:
# 
# $$
# tf(t,d) = { f_{ t,d } \over \sum_{t' \in d } f_{t,d}}
# $$
# 
# $ f_{ t,d } \quad\quad\quad\quad$: Jumlah kata t muncul dalam dokumen
# 
# $ \sum_{t' \in d } f_{t,d} \quad\quad$: Jumlah seluruh kata yang ada dalam dokumen
# 
# Rumus Inverse Document Frequency:
# 
# $$
# idf( t,D ) = log { N \over { | \{ d \in D:t \in d \} | } }
# $$
# 
# $ N \quad\quad\quad\quad\quad$ : Jumlah seluruh dokumen
# 
# $ | \{ d \in D:t \in d \} | $ : Jumlah dokumen yang mengandung kata $ t $
# 
# Rumus TFIDF:
# 
# $$
# tfidf( t,d,D ) = tf( t,d ) \times idf( t,D )
# $$

# In[14]:


vect = TfidfVectorizer()


# Setelah objek **TfidfVectorizer** dibuat gunakan method **fit_transform** dengan argumen data yang akan dicari nilai **TF-IDF**-nya

# In[15]:


vect_text = vect.fit_transform(df['judul'])


# In[16]:


attr_count = vect.get_feature_names_out().shape[0]
print(f'Jumlah atribut dalam Document-Term Matrix : {attr_count}')


# #### Menyimpan hasil TF_IDF ke DataFrame

# Hasil TF-IDF perlu diubah terlebih dahulu menjadi array agar dapat digunakan sebagai data. Kemudian untuk parameter kolom-nya dapat didapatkan menggunakan method get_feature_names_out pada objek TfidfVectorizer.

# In[17]:


tfidf = pd.DataFrame(
    data=vect_text.toarray(),
    columns=vect.get_feature_names_out()
)
tfidf.head()


# Mencari nilai **idf** dengan mengakses atribut **idf_** pada objek **tfidfVectorizer**. Atribut **idf_** hanya terdefinisi apabila parameter **use_idf** saat instansiasi objekk tfidfVectorizer bernilai **True**. Namun, **use_idf** sudah bernilai **True** secara default, sehingga kita dapat perlu menentukannya secara manual. 

# In[18]:


idf = vect.idf_


# In[19]:


dd= dict(zip(vect.get_feature_names_out(), idf))

l = sorted(dd, key = dd.get)


# Kita dapat melihat kata yang paling sering dan paling jarang muncul pada judul tugas akhir berdasarkan nilai idf. Kata yang memiliki nilai lebih kecil, adalah kata yang paling sering muncul dalam judul

# In[20]:


print("5 Kata paling sering muncul:")
for i, word in enumerate(l[:5]):
    print(f"{i+1}. {word}\t(Nilai idf: {dd[word]})")


# In[21]:


print("5 Kata paling jarang muncul:")
for i, word in enumerate(l[:-5:-1]):
    print(f"{i+1}. {word}\t(Nilai idf: {dd[word]})")


# ## TOPIC MODELLING

# ### Latent Semantic Analysis (LSA)

# **Latent Semantic Analysis (LSA)** adalah metode yang memanfaatkan model statistik matematis untuk menganalisa struktur semantik suatu teks. LSA digunakan untuk menilai judul tugas akhir dengan mengkonversikan judul tugas akhir menjadi matriks-matriks yang diberi nilai pada masing-masing term untuk dicari kesamaan dengan term. Langkah-langkah LSA dalam penilaian judul tugas akhir adalah sebagai berikut:
# 
# 1. Text Processing
# 2. Document-Term Matrix
# 3. Singular Value Decomposition (SVD)
# 4. Cosine Similarity Measurement

# #### Singular Value Decomposition(SVD)

# **Singular Value Decomposition (SVD)** adalah teknik untuk mereduksi dimensi yang bermanfaat untuk memperkecil nilai kompleksitas dalam pemrosesan Document-Term Matrix. SVD merupakan teorema aljabar linier yang menyebutkan bahwa persegi panjang dari Document-Term Matrix dapat dipecah/didekomposisikan menjadi tiga matriks, yaitu Matriks ortogonal U, Matriks diagonal S, Transpose dari matriks ortogonal V.

# $$
# A_{mn} = U_{mm} \times S_{mn} \times V^{T}_{nn}
# $$
# 
# $ A_{mn} $ : matriks awal
# 
# $ U_{mm} $ : matriks ortogonal
# 
# $ S_{mn} $ : matriks diagonal
# 
# $ V^{T}_{nn} $ : Transpose matriks ortogonal V

# Setiap baris dari matriks $ U $ (Document-Term Matrix) adalah bentuk vektor dari dokumen. Panjang dari vektor-vektor tersebut adalah jumlah topik. Sedangkan matriks $ V $ (Term-Topic Matrix) berisi kata-kata dari data.
# 
# SVD akan memberikan vektor untuk setiap dokumen dan kata dalam data. Kita dapat menggunakan vektor-vektor tersebut untuk mencari kata dan dokumen serupa menggunakan metode **Cosine Similarity**.
# 
# Dalam mengimplementasikan LSA, dapat menggunakan fungsi TruncatedSVD. parameter n_components digunakan untuk menentukan jumlah topik yang akan diekstrak.
# 
# 

# In[22]:


lsa_model = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[23]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print(f"Topic {i} : {topic*100}")


# In[24]:


(count_topic, count_word) = lsa_model.components_.shape
print(f"Jumlah topik\t: {count_topic}")
print(f"Jumlah kata\t: {count_word}")


# Sekarang kita dapat mendapatkan daftar kata yang penting untuk setiap topik. Jumlah kata yang akan ditampilkan hanya 10. Untuk melakukan sorting dapat menggunakan fungsi sorted, lalu slicing dengan menambahkan \[:10\] agar data yang diambil hanya 10 data pertama. Slicing dilakukan berdasarkan nilai pada indeks 1 karena nilai dari nilai lsa.

# In[25]:


# most important words for each topic
vocab = vect.get_feature_names_out()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)

    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print(f"Topic {i}: ")
    print(" ".join([ item[0] for item in sorted_words ]))
    print("")
         


# ### Latent Dirichlet Allocation (LDA)  

# **Latent Dirichlet Allocation (LDA)** adalah model generatif statistik yang dari koleksi data diskrit seperti kumpulan dokumen (*corpus*).
# 
# Awal dibuatnya LDA yaitu bahwa dokumen terdiri dari beberapa topik.  Proses mengasumsikan bahwa dokumen berasal dari topik tertentu melalui *imaginary random process*. Setiap topik dibentuk oleh distribusi kata-kata.
# 
# Topik yang mendeskripsikan kumpulan dari suatu dokumen dapat ditentukan setalah topik LDA dibuat. Pada sisi sebelah kanan gambar diatas menunjukkan daftar topik serta 15 kata dengan distribusi tertinggi untuk masing-masing topik tersebut. 
# 
# Rumus Dirichlet Distribution:
# $$
# f\left(x_{1}, \ldots, x_{K} ; \alpha_{1}, \ldots, \alpha_{K}\right)=\frac{\Gamma\left(\sum_{i=1}^{K} \alpha_{i}\right)}{\prod_{i=1}^{K} \Gamma\left(\alpha_{i}\right)} \prod_{i=1}^{K} x_{i}^{\alpha_{i}-1}
# $$

# Untuk melakukan perhitungan LDA dengan library sklearn, dapat dilakukan dengan menggunakan kelas *LatentDirichletAllocation* yang ada pada modul *sklearn.decomposition*. Parameter yang digunakan antara lain:
# - n_components = 2\
#     Mengatur jumlah topik menjadi 2
# 
# - learning_method ='online'\
#     Mengatur agar metode pembelajaran secara online. sehingga akan lebih cepat ketika menggunakan data dalam jumlah besar.
#      
# - random_state = 42\
#     Untuk mendapatkan hasil pengacakan yang sama selama 42 kali kode dijalankan  
# 
# - max_iter = 1 \
#     Untuk mengatur jumlah iterasi training data (epoch) menjadi 1 kali saja.

# In[26]:


lda_model = LatentDirichletAllocation(n_components=2,learning_method='online',random_state=42,max_iter=1) 


# In[27]:


lda_top = lda_model.fit_transform(vect_text)


# In[1]:


(count_doc_lda, count_topic_lda) = lda_top.shape
print(f"Jumlah dokumen\t: {count_doc_lda}")
print(f"Jumlah topik\t: {count_topic_lda}")


# In[30]:


# composition of doc 0 for eg
print("Document 0: ")
for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")


# Seperti yang dapat dilihat pada program di atas bahwa Topic 1 lebih dominan daripada topik 0 pada document 0.

# In[59]:


(count_topic_lda, count_word_lda) = lda_model.components_.shape
print(f"Jumlah Topik\t: {count_topic_lda}")
print(f"Jumlah kata\t: {count_word_lda}")


# #### 10 kata paling penting untuk suatu topik

# In[49]:


vocab = vect.get_feature_names_out()

def get_important_words(comp, n):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:n]
    return " ".join([t[0] for t in sorted_words])


# In[50]:


for i, comp in enumerate(lda_model.components_):
    print("Topic "+str(i)+": ")
    print(get_important_words(comp, 10))
    print("")


# #### Visualisasi 50 kata penting menggunakan wordcloud

# In[54]:


# Generate a word cloud image for given topic
def draw_word_cloud(index):
  imp_words_topic = get_important_words(lda_model.components_[index], 50)
  
  wordcloud = WordCloud(width=600, height=400).generate(imp_words_topic)
  plt.figure( figsize=(5,5))
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout()
  plt.show()
 


# In[55]:


draw_word_cloud(0)


# In[56]:


draw_word_cloud(1)

