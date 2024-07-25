import pandas as pd
from sklearn import preprocessing
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

# Flag to specify if using separate CSVs for train and test
SEPARATE_CSV = True

if SEPARATE_CSV:
    train_csv = './combined_audio_features.csv'
    test_csv = './convolved_GTZAN_air_type1_air_binaural_aula_carolina_1_3_0_3.csv'
    # Read train and test data from different CSVs
    train_df = pd.read_csv(train_csv)
    train_df = train_df.iloc[0:, 1:]
    test_df = pd.read_csv(test_csv)
    test_df = test_df.iloc[0:, 1:]

    # Separate features and labels for train and test datasets
    X_train = train_df.loc[:, train_df.columns != 'label']
    y_train = train_df['label']
    X_test = test_df.loc[:, test_df.columns != 'label']
    y_test = test_df['label']
else:
    # If not using separate CSVs, read from a single CSV and split
    df = pd.read_csv('./combined_audio_features.csv')
    df = df.iloc[0:, 1:]
    y = df['label']
    X = df.loc[:, df.columns != 'label']

    # Encode class labels to numerical values
    y_encoded = preprocessing.LabelEncoder().fit_transform(y)
    
    # Calculate correlation between features and class labels
    correlations = {}
    for col in X.columns:
        correlations[col] = np.corrcoef(X[col], y_encoded)[0, 1]

    # Print correlation values
    for feature, corr in correlations.items():
        print(f"Feature: {feature}, Correlation with class: {corr}")

    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(np_scaled, columns=cols)

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Encode class labels to numerical values for separate CSV case
y_train_encoded = preprocessing.LabelEncoder().fit_transform(y_train)
y_test_encoded = preprocessing.LabelEncoder().fit_transform(y_test)
print("Unique labels in test data:", np.unique(y_test_encoded))
def testModel(model, title="Default"):
    model.fit(X_train, y_train_encoded)
    preds = model.predict(X_test)
    print(title + ' Accuracy :' + str(accuracy_score(y_test_encoded, preds)))
    print(title + ' Precision :' + str(precision_score(y_test_encoded, preds, average='macro')))
    print(title + ' Recall :' + str(recall_score(y_test_encoded, preds, average='macro')))
    print(title + ' F1 Score :' + str(f1_score(y_test_encoded, preds, average='macro')))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_encoded, preds), display_labels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])

    disp.plot()

    plt.show()

# Stochastic Gradient Descent
sgd = SGDClassifier(max_iter=5000, random_state=0)
testModel(sgd, "Stochastic Gradient Descent")

# KNN
knn = KNeighborsClassifier(n_neighbors=1)
testModel(knn, "KNN")

# Random Forest
randomForest = RandomForestClassifier(n_estimators=1000, max_depth=200, random_state=0)
testModel(randomForest, "Random Forest")

# Extra Trees
extraTrees = ExtraTreesClassifier(n_estimators=1000, max_depth=200, random_state=42)
testModel(extraTrees, "Extra Trees")

# Support Vector Machine
svm = SVC(decision_function_shape="ovo")
testModel(svm, "Support Vector Machine")

# Logistic Regression
lg = LogisticRegression(max_iter=1000, random_state=0, solver='lbfgs', multi_class='multinomial')
testModel(lg, "Logistic Regression")

# Neural Nets
nn = MLPClassifier(solver='lbfgs', max_iter=10000, alpha=1e-5, hidden_layer_sizes=(100,100,100,100), random_state=1)
testModel(nn, "Neural Nets")
