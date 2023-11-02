#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1 Lire un fichier CSV avec pandas==========================================================================================


#import pandas as pd 
#df = pd.read_csv("data1.csv") 
#df.shape 
#df.info() 
#df.head() 
#df.tail()
#df.columns.tolist()===== THIS IS USED TO ORGANZIE THE DATA
#df.index.tolist() 

# 2 Analyser le dataframe===========================================================================================================================================================================

#df.set_index("email",inplace =True)  ==== THIS IS USED TO CHAGE AN INDEX
#df.index #df["email"].head() 
#df.email.head() 
#df.dtypes 
#type(df) 
#type(df.email) 

# Sélectionner des données=============================================================================================================================================================================

#df[10:20] ==== THIS UESD TO FIND A VALUE
#df.loc[10:20] 
#df.iloc[10:20] 
#df = df.set_index("email") 
#df.index #df.email.loc['hharridge1@gnu.org'].values
#df_email.loc['hharridge1@gnu.org'].values.tolist()
#df_email.loc[['ekilminster2@etsy.com','hharridge1@gnu.org']]
#type (df_email.loc[['ekilminster2@etsy.com','hharridge1@gnu.org']])
#df.set_index("id",inplace =True) 
#df['gender'] == "Male"


# 3  Les filtres ==========================================================================================================================================================================

#df[df['gender'] == "Male"]     ==== THIS A FILITERs (difRent ways to filiter)
#df[df['gender'] == "Female"]
#female_filter = df['gender'] == "Female"
# -->df[female_filter]
                     =============================================================
# country_filiter = df['country'].isin (["France","Canada"]) ===== THIS FLITER BY COUNTRY WHEN YOU ARE LOOKING FOR MANY VALUSES YOU CAN USE ISIN
#-->df[filiter_country ]
                     =============================================================
#df[df["price_paid"]>5] # you can not compare or calculate str to int becuse the paid price is the usd sign


# 4 Supprimer une colonne ======================================================================================================================================================================================

#=========================# HERE YOU REMAOVE THE DOLLER SIGN $ WITH NOTHING AND CHANGE THE TYPE OF THE OF THE VARINTE TO FLOAT
# BUT YOU NEED FIRST TO MAKE A COPY +++++ df_test = df.copy()+++++
# df_test = df.copy()
#df_test.price_paid = df_test.price_paid.apply(lambda x: x.replace("$", ""))
#df_test.price_paid = df_test.price_paid.astype(float)
#df_test[df_test["price_paid"]>=5.0]

# 5 Traiter les valeurs manquantes=======================================================================================================================================================

##HERE YOU CAN DELETE AND REPLACE INDEX IN THE DATA
#df1 = df.copy()\
#axis=1 is for columns 
#axis=0 is for row(line)
#df1.drop("ip_address",axis=1) =================# this will not do nothing unless YOU ADD inplace= True
#
#df1.drop("ip_address",axis=1, inplace=True)
#df1.set_index("gender", inplace=True)
#df1.drop("Male", axis=0, inplace= True)
#df1.head()

#df1.drop(['first_name', 'last_name', 'email'], axis=1, inplace =True)
#del df1["tax"]  ===============================#  ANOTHER FON=CTION TO DROP

# 6 Dealing with Missing Values=============================================================================================================================================================================

#df.isnull()   ==============================================# this is used to find the missing data 
#df.notnull()  ==============================================# this is used to find the missing data 
#df["tax"].notnull()   ======================================# this is used to find the missing data in a certian coloumn or a row 
#df[df["tax"].notnull()]  ===================================# this is used to find and filliter the data in the missing data in a certian coloumn or a row and it will show without missing value
#df[df["tax"].isnull()]  ====================================#  it will show only missing value (NaN)
#df["tax"].fillna(0) ========================================# here we want to  replace nan withb something
#df["tax"].fillna(method='bfill') ===========================# this one way to add a value autour de data take from other place and add but == is not alwasys good
#df.dropna() ================================================# this will drop any columns that has NULL value so be carefull
#df.dropna(subset=["tax"], inplace= True) 
#df 
#
#
# 7 Ajouter des colonnes=============================================================================================================================================================================

#df["tax"].fillna(0, inplace=True)
#df
#df.price_paid = df.price_paid.apply(lambda x: x.replace("$", ""))
#df.price_paid = df.price_paid.astype(float)
#df.dtypes
# df["test"] = 0 ============================create a new columns 
#del df["test"]
#df["total_price"] = df["price_paid"] * df["tax"]
#df
#df["total_price"] = df["price_paid"] * (1 - df["tax"] / 100)
#df
#countries =  {"United States":"UN", "France":"FR", "Canada": "CA", "Morocco": "MA"} # =========== making a new dict
#df['country_code'] = df['country'].map(countries)

# 8 Analyser les données=============================================================================================================================================================================

#df.describe()
#df["price_paid"].describe()
#df["tax"].describe()
#===================================================
#df.price_paid = df.price_paid.apply(lambda x: x.replace("$", ""))
#df.price_paid = df.price_paid.astype(float)
#======================================================================
#df["price_paid"].mean() # you need to to chage to type of the data before the calculation
#df["price_paid"].sum()
#df["price_paid"].max()
#df["price_paid"].min()
# =============================================================
#   =======================# FIND THE UNIQUE VALUE  AND NON UMIQUE
#df["country"].unique()
#df["country"].nunique()
#df["price_paid"].unique().tolist()
# =======================
#df["country"].value_counts() # how many numbers pery value per country
#df["country"].value_counts(normalize=True) 
#df["gender"].value_counts() # how many numbers of value per country
#df["gender"].value_counts(normalize=True) # see it in a percentage %
#==========================
#df.groupby("country")
#df.groupby("country").sum()
#df.groupby("country").mean()
#df.groupby("gender").mean()
#df.groupby("gender")["price_paid"].sum()
#df.groupby(["gender", "country"]).sum()
#df.groupby(["gender", "country"]).mean()

# 9 Installation de matplotlib + Tracer une courbe + D'autres types de graphiques =============================================================================================================================================================================

#### ========================= this part in mandetory=================
#df.price_paid = df.price_paid.apply(lambda x: x.replace("$", ""))
#df.price_paid = df.price_paid.astype(float)
####===================================================================
#df.groupby(["gender", "country"]).mean()
#df.groupby("date")["price_paid"].sum()
#df.groupby("date")["price_paid"].sum().plot(figsize=[20,10])=========================curab
#===================  all you need to make the chat is plot(figsize=[10.10])
#df.groupby("gender")["price_paid"].sum().plot.pie(legend=True, figsize=(7, 7)) =======pie
#df.groupby("country")["price_paid"].sum().plot.bar(rot=40, legend= True)==============bar


# In[1]:


import pandas as pd 


# In[2]:


pip install matplotlib


# In[10]:


df = pd.read_csv("data1.csv")


# In[ ]:





# In[ ]:




