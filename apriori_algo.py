from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks//ML/Groceries_dataset.csv')  # Change path
df.head()

transactions = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).tolist()

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)

df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
df_encoded.head()

from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)
frequent_itemsets.head(
