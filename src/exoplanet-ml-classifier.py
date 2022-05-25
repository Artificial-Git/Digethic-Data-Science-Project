
from scipy import stats
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import shap
from statsmodels import robust
import time

data = pd.read_csv('../data/cumulative.csv')


# Entfernung von übeflüssigen Spalten ohne Informationsgehalt:
# Spalten mit "id" oder "name" im Namen enthalten nur Bezeichnungen. Die beiden "koi_teq_err"-
# Spalten sind leer.

to_pop = ["rowid", "kepid", "kepoi_name", "kepler_name",
          "koi_teq_err1", "koi_teq_err2", "koi_tce_delivname"]

for col in to_pop:
    data.pop(col)

# Umwandelung von Ordinalwerten in numerische Werte:

koi_disposition_dict = {'FALSE POSITIVE': 0, 'CANDIDATE': 1, 'CONFIRMED': 2}
data["koi_disposition"] = data["koi_disposition"].replace(koi_disposition_dict)

koi_pdisposition_dict = {'FALSE POSITIVE': 0, 'CANDIDATE': 1}
data["koi_pdisposition"] = data["koi_pdisposition"].replace(
    koi_pdisposition_dict)


# Ersetzung fehlender Werte durch imputierte Werte

knn_imputer = KNNImputer()

for column in data:

    data[column] = knn_imputer.fit_transform(
        data[column].to_numpy().reshape(-1, 1))


# Korrelationsmatrix erstellen

fig = plt.figure(figsize=(35, 35), dpi=300)
plt.title("Correlation heatmap of Kepler Objects of Interest dataset features")
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.savefig('../data/figures/heatmap.png')


# Streudiagrammsmatrix erstellen

plt.title("Scatter plot matrix of Kepler Objects of Interest dataset features")
pd.plotting.scatter_matrix(data, figsize=(35, 35), c="#1ACC94")
plt.savefig('../data/figures/scatter_matrix.png')


# Kastengrafik erstellen
# Werte skalieren um die Kastengrafik darstellen zu können.

scaled_data = data.copy()
minmax = MinMaxScaler()

for column in scaled_data:

    if column == 'koi_disposition':
        pass
    else:
        scaled_data[column] = minmax.fit_transform(
            scaled_data[column].to_numpy().reshape(-1, 1))

fig = plt.figure(figsize=(35, 35))
plt.title("Box plot of scaled Kepler Objects of Interest dataset features")
plt.boxplot(data, labels=data.columns, medianprops=dict(color="#1ACC94"))
plt.xticks(rotation=45)
plt.show()


# Entfernung von Attributen, die Proxies des Zielwertes sind, bzw.  Flag-Werte sind.
x = data.drop(['koi_disposition', 'koi_pdisposition', 'koi_score', 'koi_fpflag_ss', 'koi_fpflag_co',
               'koi_fpflag_ec', 'koi_fpflag_nt'], axis=1)
y = data["koi_disposition"].to_numpy().reshape(-1, 1)
type(x), type(y)
x_train, x_test, y_train, y_test = train_test_split(x, y)


# Werte skalieren

minmax = MinMaxScaler()

for column in x_train:

    if column == 'koi_disposition':
        pass
    else:
        x_train[column] = minmax.fit_transform(
            x_train[column].to_numpy().reshape(-1, 1))

        x_test[column] = minmax.fit_transform(
            x_test[column].to_numpy().reshape(-1, 1))


# Gittersuchverfahren für knn-Methode mit höchster Akkuranz
grid_points_n_neighbors = np.arange(1, 250)
grid_points_weights = ("uniform", "distance")

max_z = -np.inf
best_point_knn = (5, "uniform")

for n_neighbors in grid_points_n_neighbors:
    for weights in grid_points_weights:
        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, p=1)
        knn_model_fit = knn.fit(x_train, y_train)
        knn_model_predict = knn.predict(x_test)
        z = accuracy_score(knn_model_predict, y_test)
        print(z, max_z, best_point_knn, (n_neighbors, weights))
        if z > max_z:
            max_z = z
            best_point_knn = (n_neighbors, weights)

print(f"Bester Punkt mit z={max_z:.04f} gefunden bei {best_point_knn}")


# Gittersuchverfahren für SVC-Methode mit höchster Akkuranz

grid_points_C = np.arange(2, 8, 0.1)
grid_points_degree = np.arange(5, 8)
max_z = -np.inf
best_point_svm = (0.1, 1)

for C in grid_points_C:
    for degree in grid_points_degree:
        svm = SVC(C=C, kernel="poly", degree=degree,
                  gamma="scale", probability=True)
        svm_model_fit = svm.fit(x_train, y_train)
        svm_model_predict = svm.predict(x_test)
        z = accuracy_score(svm_model_predict, y_test)
        print(z, max_z, best_point_svm, (C, degree))
        if z > max_z:
            max_z = z
            best_point_svm = (C, degree)

print(f"Bester Punkt mit z={max_z:.04f} gefunden bei {best_point_svm}")


# Gittersuchverfahren für Zufallswald-Methode mit höchster Akkuranz

grid_points_n_estimators = np.arange(50, 70)
grid_points_max_depth = np.arange(7, 14)
grid_points_min_samples_split = np.arange(2, 14)
max_z = -np.inf
best_point_rf = (35, 7, 2)
for n_estimators in grid_points_n_estimators:
    for max_depth in grid_points_max_depth:
        for min_samples_split in grid_points_min_samples_split:
            rf = RandomForestClassifier(n_estimators=n_estimators, criterion="entropy", max_depth=max_depth,
                                        min_samples_split=min_samples_split, max_features="auto")
            rf_fit = rf.fit(x_train, y_train)
            z = accuracy_score(rf.predict(x_test), y_test)
            print(z, max_z, best_point_rf,
                  (n_estimators, max_depth, min_samples_split))
            if z > max_z:
                max_z = z
                best_point_rf = (n_estimators, max_depth, min_samples_split)

print(f"Bester Punkt mit z={max_z:.04f} gefunden bei {best_point_rf}")
