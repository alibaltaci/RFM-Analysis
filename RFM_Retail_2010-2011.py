# RFM Analysis - Online Retail 2010-2011


# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

# to display all columns and rows:
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# how many digits will be shown after the comma:
pd.set_option("display.float_format", lambda x: "%.2f" % x)



# General overview #

# Data for 2010 - 2011
df_2010_2011 = pd.read_excel("data/online_retail_II.xlsx", sheet_name = "Year 2010-2011")

# Let's back up dataset.
df = df_2010_2011.copy()

# First 5 lines.
df.head()

# Last 5 lines.
df.tail()

# Shape of the dataset
df.shape

# General information
df.info()

# Are there any missing values in the dataset?
df.isnull().values.any()

# How many missing values are in which variable?
df.isnull().sum()

# How many unique products are in the data set?
df["Description"].nunique()

# Products and order quantities
df["Description"].value_counts().head()

# Let's find the most ordered products
df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending=False).head()

# For RFM Analysis, let's find the total spending for each customer by multiplying the quantity and price variables.
df["TotalPrice"] = df["Quantity"] * df["Price"]

df.head()

# Let's find the canceled orders and then delete those orders from the data set.
df[df["Invoice"].str.contains("C", na = False)].head()

# Let's delete canceled products with the help of Tilda
df = df[~df["Invoice"].str.contains("C", na = False)]

#  Look at the total price for each invoice
df.groupby("Invoice").agg({"TotalPrice":"sum"}).head()

# Country order amount
df["Country"].value_counts()

# Total price for each country
df.groupby("Country").agg({"TotalPrice":"sum"}).head()

# Now let's sort the countries by total price
df.groupby("Country").agg({"TotalPrice":"sum"}).sort_values("TotalPrice", ascending = False).head()


# Let's try to find the most returned product.

# First, let's restore the initial state of the dataset.
df1 = df_2010_2011

df1.head()

df1[df1["Invoice"].str.contains("C", na = False)].head()

# Keep cancelled orders elsewhere
df_c = df1[df1["Invoice"].str.contains("C", na = False)]

df_c.head()

# 5 most returned products
df_c["Description"].value_counts().head()


# Missing value analysis
df.isnull().sum()

# Let's delete the missing values
df.dropna(inplace=True)

# Look at the missing values again
df.isnull().sum()

# Dataset after deleting missing data
df.shape


# Describe
df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95, 0.99]).T

# Querying outlier for dataset
for feature in ["Quantity","Price","TotalPrice"]:

    Q1 = df[feature].quantile(0.01)
    Q3 = df[feature].quantile(0.99)
    IQR = Q3-Q1
    upper = Q3 + 1.5*IQR
    lower = Q1 - 1.5*IQR

    if df[(df[feature] > upper) | (df[feature] < lower)].any(axis=None):
        print(feature,"yes")
        print(df[(df[feature] > upper) | (df[feature] < lower)].shape[0])
    else:
        print(feature, "no")



# RFM ANALYSIS #

"""
It consists of the initials of Recency, Frequency, Monetary.

It is a technique that helps determine marketing and sales strategies based on customers' buying habits.

* Recency: Time since the customer's last purchase

    - In other words, it is the time elapsed since the last contact of the customer.

    - Today's date - Last purchase

    - For example, if we are doing this analysis today, today's date - the last product purchase date.

    - For example, this could be 20 or 100. We know that the customer with Recency = 20 is hotter. 
      He has been in contact with us recently.

* Frequency: Total number of purchases.

* Monetary: The total expenditure made by the customer.
"""


# RECENCY

df.head()

df.info()

# first invoice date
df["InvoiceDate"].min()

# last invoice date
df["InvoiceDate"].max()

# We will consider the last transaction date as today's date.
today_date = dt.datetime(2011, 12, 9)

today_date

# Last invoice date for each customer
df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head()

# intager
df["Customer ID"] = df["Customer ID"].astype(int)

df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head()

# Time passed since the customer's last transaction?
temp_df = (today_date - (df.groupby("Customer ID")).agg({"InvoiceDate":"max"}))

temp_df.head()

# Rename
temp_df.rename(columns = {"InvoiceDate": "Recency"}, inplace = True)

temp_df.head()

# let's just pick the days
# ex:
temp_df.iloc[0,0].days

# for all rows
recency_df = temp_df["Recency"].apply(lambda x: x.days)

recency_df.head()


# FREQUENCY

df.head()

# Number of invoices for each customer

df.groupby(["Customer ID", "Invoice"]).agg({"Invoice":"nunique"}).head(50)

# We can calculate "Fequency" by simply finding number of unique values for each customer.
freq_df = df.groupby("Customer ID").agg({"InvoiceDate":"nunique"})

freq_df.head()

# Rename
freq_df.rename(columns = {"InvoiceDate":"Frequency"}, inplace=True)

freq_df.head(10)


# MONETARY

df.head()

# Let's calculate the total spend for each customer
monetary_df = df.groupby("Customer ID").agg({"TotalPrice":"sum"})

monetary_df.head()

#  Rename
monetary_df.rename(columns = {"TotalPrice":"Monetary"}, inplace = True)

monetary_df.head()


# Look at the shapes of the recency, frequency, monetary dataframe
print(recency_df.shape, freq_df.shape, monetary_df.shape)

# Let's Concatenate the above data frames.
rfm = pd.concat([recency_df, freq_df, monetary_df], axis = 1)

rfm.head()


# Recency Score

# We gave the smallest value the highest score. Because smaller value is better for innovation.
# It means that the customer has been shopping recently.
rfm["RecencyScore"] = pd.qcut(rfm["Recency"], 5, labels = [5, 4 , 3, 2, 1])

rfm.head()


# Frequency Score

# The higher the value for Frequency, the better.
# That's why we give the highest score to the highest value.
rfm["FrequencyScore"] = pd.qcut(rfm["Frequency"].rank(method = "first"), 5, labels = [1,2,3,4,5])

rfm.head()


# Monetary Score

rfm["MonetaryScore"] = pd.qcut(rfm["Monetary"].rank(method = "first"), 5, labels = [1,2,3,4,5])

rfm.head()


# Calculating RFM Score --> R + F + M = RFM --> 3 + 1 + 4 = 314
(rfm['RecencyScore'].astype(str) +
 rfm['FrequencyScore'].astype(str) +
 rfm['MonetaryScore'].astype(str)).head()

# Let's add the RFM score to the rfm data frame
rfm['RFM_SCORE'] = (rfm['RecencyScore'].astype(str) +
                    rfm['FrequencyScore'].astype(str) +
                    rfm['MonetaryScore'].astype(str))

rfm.head()

# Describe
rfm.describe().T

# Customers with a value of 555 (champions)
rfm[rfm["RFM_SCORE"]=="555"].head()

# Customers with a value of 111 (hibernating)
rfm[rfm["RFM_SCORE"]=="111"].head()


# RFM Mapping

seg_map = {
    r'[1-2][1-2]':'Hibernating',
    r'[1-2][3-4]':'At Risk',
    r'[1-2]5':'Can\'t Loose',
    r'3[1-2]':'About to Sleep',
    r'33':'Need Attention',
    r'[3-4][4-5]':'Loyal Customers',
    r'41':'Promising',
    r'51':'New Customers',
    r'[4-5][2-3]':'Potential Loyalists',
    r'5[4-5]':'Champions'
}

# For segmenting we will use Recency and Frequency scores

rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)

rfm.head()

# Let's replace 'segment' names with the seg_map names.

rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)

rfm.head()

# Summary
rfm[["Segment","Recency","Frequency", "Monetary"]].groupby("Segment").agg(["mean","median","count"])

# Number of customers
rfm.shape[0]


# Let's access the ID numbers of the customers in the loyal customers class.

rfm[rfm["Segment"] == "Loyal Customers"].index

loyal_df = pd.DataFrame()

loyal_df["LoyalCustomersID"] = rfm[rfm["Segment"] == "Loyal Customers"].index

loyal_df.head()


# Save dataframe in a .csv file.

loyal_df.to_csv("RFM_Loyal_Customers_ID_2010-2011.csv", index=False)


# References

"""
https://www.datacamp.com/community/tutorials/introduction-customer-segmentation-python

https://medium.com/@sbkaracan/rfm-analizi-ile-m%C3%BC%C5%9Fteri-segmentasyonu-proje-416e57efd0cf

https://www.wikiwand.com/en/RFM_(market_research)

https://www.veribilimiokulu.com/blog/rfm-analizi-ile-musteri-segmentasyonu/
"""
