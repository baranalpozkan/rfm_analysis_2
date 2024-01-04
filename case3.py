###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# İngiltere merkezli perakende şirketi müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istemektedir.
# Ortak davranışlar sergileyen müşteri segmentleri özelinde pazarlama çalışmaları yapmanın gelir artışı sağlayacağını düşünmektedir.
# Segmentlere ayırmak için RFM analizi kullanılacaktır.

# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin
# 01/12/2009 - 09/12/2011 tarihleri arasındaki online satış işlemlerini içeriyor.
# Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu müşterisinin toptancı olduğu bilgisi mevcuttur.

# Değişkenler

# Invoice: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi.
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# Price: Ürün fiyatı (Sterlin cinsinden)
# Customer ID: Eşsiz müşter numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

###############################################################
# 2. Veriyi Anlama (Data Understanding)
###############################################################

# Gerekli Kütüphaneler

import numpy as np
import pandas as pd
import datetime as dt
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: '%3f.' % x)

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

df.shape
df.describe().T
df.info()

df.isnull().sum()
df.dropna(inplace=True)

df["Description"].nunique()
df["Description"].value_counts()

df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending=False).head()

df = df[~df["Invoice"].str.contains("C", na=False)]

df["TotalPrice"] = df["Quantity"] * df["Price"]

###############################################################
# 3. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
###############################################################

df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     "Invoice": lambda Invoice: Invoice.nunique(),
                                     "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

rfm.columns = ["recency", "frequency", "monetary"]

rfm.head(20)

rfm.describe().T

rfm = rfm[rfm["monetary"] > 0]


###############################################################
# 4. RFM Skorlarının Oluşturulması ve Tek bir Değişkene Çevrilmesi
###############################################################

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)


###############################################################
# 5. RF Skorunun Segment Olarak Tanımlanması
###############################################################

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

rfm.head()


###############################################################
# 5. Kampanya
###############################################################

rfm["segment"].value_counts()

rfm.reset_index(inplace=True)
camp = rfm[rfm["segment"] == "loyal_customers"]["Customer ID"]

camp.to_excel("campaign_loyal_customers.xlsx", index=False)

###############################################################
# 6. Sürecin Fonksiyonlaştırılması
###############################################################

def create_rfm(dataframe, excel=False):

    # VERIYI HAZIRLAMA
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # RFM METRIKLERININ HESAPLANMASI
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                'TotalPrice': lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', 'monetary']
    rfm = rfm[rfm['monetary'] > 0]

    # RFM SKORLARININ HESAPLANMASI
    rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

    # cltv_df skorları kategorik değere dönüştürülüp df'e eklendi
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm["segment"] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    rfm.index = rfm.index.astype(int)

    if excel:
        rfm.to_excel("rfm.xlsx")

    return rfm







































