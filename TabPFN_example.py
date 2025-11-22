# New deep learning model for tabular data.
# Pre-trained, no hyper-parameter tuning.
# https://medium.com/@atabarezz/hybrid-neural-nets-now-own-the-future-of-classification-and-regression-9b90ffa96f58

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier, TabPFNRegressor

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
    )

clf = TabPFNClassifier()
#reg = TabPFNRegressor()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))