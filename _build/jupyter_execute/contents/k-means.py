#!/usr/bin/env python
# coding: utf-8

# # Clustering Jurnal dengan K-Means

# In[1]:


# Import library
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Clustering
from sklearn.cluster import KMeans,SpectralClustering
from yellowbrick.cluster import KElbowVisualizer

# Decomposition
from sklearn.decomposition import PCA, TruncatedSVD

# Text
import unicodedata, re, string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# In[ ]:


df=pd.read_csv('detail_manajemen.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


jurnal = df[['judul', 'abstraksi']].dropna()
jurnal.head()


# In[ ]:


factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

factory = StemmerFactory()
stemmer = factory.create_stemmer()


# In[ ]:


def clean_text(text):
    # Mengubah teks menjadi lowercase
    cleaned_text = text.lower()
    # Menghapus angka
    cleaned_text = re.sub(r"\d+", "", cleaned_text)
    # Menghapus white space
    cleaned_text = cleaned_text.strip()
    # Menghapus tanda baca
    cleaned_text = cleaned_text.translate(str.maketrans(string.punctuation, " "*len(string.punctuation), ""))
    # Hapus stopword
    cleaned_text = stopword.remove(cleaned_text)
    # Stemming
    cleaned_text = stemmer.stem(cleaned_text)

    cleaned_text = stemmer.stem(cleaned_text)

    cleaned_text = cleaned_text.replace('\n', ' ')
    cleaned_text = cleaned_text.replace('\r', ' ')
    return cleaned_text


# In[ ]:


#Tokenize text
jurnal['cleaned_abstraksi'] = jurnal['abstraksi'].apply(clean_text)
jurnal.head()


# In[ ]:


"""Apply the TF_idf vectorizer to get the sparse matrix of the TF_IDF process"""

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(jurnal['cleaned_abstraksi'])


# In[ ]:


"""A simple view of the feature names"""
print(vectorizer.get_feature_names()[:10])


# ## Dimentional reduction PCA
# **Principal component analysis (PCA)** adalah teknik analisis statistik multivariat. Teknik analisis statistik yang paling populer sekarang dapat dikatakan adalah PCA. PCA digunakan dalam bidang pengenalan pola serta pemrosesan sinyal.
# 
# PCA pada dasarnya merupakan dasar dari analisis data multivariat yang menerapkan metode proyeksi. Teknik analisis ini biasanya digunakan untuk meringkas tabel data multivariat dalam skala besar hingga bisa dijadikan kumpulan variabel yang lebih kecil atau indeks ringkasan. Dari situ, kemudian variabel dianalisis untuk mengetahui tren tertentu, klaster variabel, hingga outlier.

# In[ ]:


pca = PCA().fit(tfidf.toarray())
cmv = pca.explained_variance_ratio_.cumsum()
print(cmv)
print(cmv.shape)


# # Reduksi Dimensi
# pca = PCA(n_components=40)
# pca_df = pca.fit_transform(tfidf)
# pca_df

# ## Metode Elbow
# **Metode Elbow** adalah metode untuk menentukan jumlah cluster yang tepat melalui persentase hasil perbandingan antara jumlah cluster yang akan membentuk siku pada suatu titik. Jika nilai cluster pertama dengan nilai cluster kedua memberikan sudut dalam grafik atau nilainya mengalami penurunan paling besar maka jumlah nilai cluster tersebut yang tepat. Untuk mendapatkan perbandingannya adalah dengan menghitung Sum of Square Error (SSE) dari masing-masing nilai cluster. Karena semakin besar jumlah nilai cluster K, maka nilai SSE akan semakin kecil.
# 
# $$
# S S E= \sum_{K=1}^{K} \sum_{X_{i}}\left|x_{i}-c_{k}\right|^{2}
# $$
# 
# Keterangan:\
# ${K}$ = _cluster_ ke-c\
# $x_{i}$ = jarak data obyek ke-i\
# $c_{k}$ = pusat _cluster_ ke-i

# In[ ]:


modelKm = KMeans(random_state=12)
visualizer = KElbowVisualizer(modelKm, k=(1,12))

visualizer.fit(tfidf)
visualizer.show() 


# ## K-Means Clustering
# K-Means Clustering merupakan algoritma yang efektif untuk menentukan cluster dalam sekumpulan data, di mana pada algortima tersebut dilakukan analisis kelompok yang mengacu pada pemartisian N objek ke dalam K kelompok (Cluster) berdasarkan nilai rata-rata (means) terdekat. Adapun persamaan yang sering digunakan dalam pemecahan masalah dalam menentukan jarak terdekat adalah persamaan Euclidean berikut :
# 
# $$
# d(p, q)=\sqrt{\left(p_{1}-q_{1}\right)^{2}+\left(p_{2}-q_{2}\right)^{2}+\left(p_{3}-q_{3}\right)^{2}}
# $$
# 
# Keterangan:\
# _d_ = jarak obyek\
# _p_ = data\
# _q_ = centroid

# In[ ]:


"""Train the Kmeans with the best n of clusters"""
modelKm = KMeans(n_clusters=3, random_state=12)
modelKm.fit(tfidf)
y_kmeans = modelKm.predict(tfidf)

"""Dimensionality reduction used to plot in 2d representation"""
pc=TruncatedSVD(n_components=2)
X_new=pc.fit_transform(tfidf)
centr=pc.transform(modelKm.cluster_centers_)

print(centr)
plt.scatter(X_new[:,0],X_new[:,1],c=y_kmeans, cmap='viridis')

