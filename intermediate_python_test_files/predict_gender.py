import pandas as pd

train = pd.read_csv('train.csv')

train['Hyp'] = 0

train.loc[train.Sex == 'female', 'Hyp'] = 1

train['Result'] = 0
train.loc[train.Survived == train['Hyp'], 'Result'] = 1

print(train.Result.value_counts(normalize=True))
