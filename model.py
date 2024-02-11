from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('./data/geom_1.csv')
test = pd.read_csv('./data/geom_2.csv')
# target = train['target']
# print(train.head(10))
# print(train.describe())
# print(train.shape)
labels = train.columns
# print(labels)
sns.pairplot(train[['ΔT = 2.5 K', 'ΔT = 5 K', 'ΔT = 10 K', 'ΔT = 15 K', 'ΔT = 20 K']])
plt.show()
sns.pairplot(test[['ΔT = 2.5 K', 'ΔT = 5 K', 'ΔT = 10 K', 'ΔT = 15 K', 'ΔT = 20 K']])
plt.show()
# labels = train.columns.drop(['id', 'target'])


# m = LogisticRegression(
#     penalty='11',
#     C=0.1
# )

# m.fit(train[labels], )
# m.predict_proba(test[labels])[:,1]

