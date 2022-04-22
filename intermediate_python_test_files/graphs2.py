import numpy as np
import pandas as pd

from matplotlib import style
import matplotlib.pyplot as plt

male_color = '#1f77b4'
female_color = '#f6c3e5'

#style.use('seaborn-dark')
df = pd.read_csv('./train.csv', index_col=0)

fig = plt.figure(figsize=(11, 7))

plt.subplot2grid((3, 3), (0, 0))
df.Survived[df.Sex == 'male'].value_counts(normalize=True).plot(kind='bar', color=male_color)
plt.title('Males survived')

plt.subplot2grid((3, 3), (0, 1))
df.Survived[df.Sex == 'female'].value_counts(normalize=True).plot(kind='bar', color=female_color)
plt.title('Females survived')

plt.subplot2grid((3, 3), (0, 2))
df.Sex[df.Survived == 1].value_counts(normalize=True).plot(kind='bar', color=[female_color, male_color])
plt.title('Survived wrt Gender')

plt.subplot2grid((3, 3), (1, 0), colspan=2)
for x in range(1, 4):
    df.Survived[df.Pclass == x].plot(kind='kde')
plt.title('Survived wrt Pclass')
plt.legend(('1st', '2nd', '3rd'))

plt.subplot2grid((3, 3), (1, 2))
df.Survived[(df.Sex == 'male') & (df.Pclass == 1)].value_counts(normalize=True).plot(kind='bar')
plt.title('Rich males survived')

plt.subplot2grid((3, 3), (2, 0))
df.Survived[(df.Sex == 'male') & (df.Pclass == 3)].value_counts(normalize=True).plot(kind='bar')
plt.title('Poor males survived')

plt.subplot2grid((3, 3), (2, 1))
df.Survived[(df.Sex == 'female') & (df.Pclass == 1)].value_counts(normalize=True, sort=False).plot(kind='bar', color=female_color)
plt.title('Rich females survived')

plt.subplot2grid((3, 3), (2, 2))
df.Survived[(df.Sex == 'female') & (df.Pclass == 3)].value_counts(normalize=True).plot(kind='bar', color=female_color)
plt.title('Poor females survived')

plt.show()
