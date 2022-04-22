from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

import pandas as pd

def clean_data(data):
    data['Fare'].fillna(data['Fare'].dropna().median(), inplace=True)
    data['Age'].fillna(data['Age'].dropna().median(), inplace=True)

    data['Embarked'].fillna('S', inplace=True)

    encoder = LabelEncoder()
    data['Embarked'] = encoder.fit_transform(data['Embarked'])

    encoder = LabelEncoder()
    data['Sex'] = encoder.fit_transform(data['Sex'])

data = pd.read_csv('train.csv', index_col=0)
clean_data(data) 

target = data.Survived.values
feature_names = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch']
features = data[feature_names].values

#LINEAR FEATURES
logistic_classifier = LogisticRegression(solver='lbfgs')
predictions = logistic_classifier.fit(features, target)
print('The accuracy of Logistic Regression Model (out of 1):',predictions.score(features, target))

poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)

#POLYNOMIAL FEATURES
logistic_classifier = LogisticRegression(solver='lbfgs', max_iter=5000)
predictions = logistic_classifier.fit(poly_features, target)
print('The accuracy of Logistic Regression Model (out of 1):',predictions.score(poly_features, target))

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=42, max_depth=7, min_samples_split=2)
predictions = decision_tree.fit(features, target)
print('The accuracy of Decision Tree Classifier (out of 1)', decision_tree.score(features, target))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(decision_tree, features, target, scoring='accuracy', cv=50)
print(scores.mean())

