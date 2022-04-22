import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('./train.csv', index_col=0)

fig = plt.figure(figsize=(11, 7))

plt.subplot2grid((2, 3), (0, 0))
df.Survived.value_counts(normalize=True).plot(kind='bar')
plt.title('Survived')

plt.subplot2grid((2, 3), (0, 1))
plt.scatter(df.Survived, df.Age, alpha=0.2)
plt.title('Age wrt Survived')

plt.subplot2grid((2, 3), (0, 2))
df.Pclass.value_counts(normalize=True).plot(kind='bar')
plt.title('Pclass')

plt.subplot2grid((2, 3), (1, 0), colspan=3)
for i in range(0, 2):
    df.Age[df.Survived == i].plot(kind='kde')
plt.title('Survived wrt Age')
plt.legend(('Deceased', 'Survived'))

plt.show()

