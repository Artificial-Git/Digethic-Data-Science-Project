
from scipy import stats
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
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


print("Lade Modelle.")

file_to_open = open("data/models/knn_model.pickle", 'rb')
opt_knn = pickle.load(file_to_open)

file_to_open = open("data/models/svm_model.pickle", 'rb')
opt_svm = pickle.load(file_to_open)

file_to_open = open("data/models/rf_model.pickle", 'rb')
opt_rf = pickle.load(file_to_open)

print("Modelle geladen.")


print("Lade Datensatz.")

data = pd.read_csv('data/cumulative_2022.04.28_05.30.33.csv', header=53)

print("Datensatz geladen.")


# Entfernung von übeflüssigen Spalten ohne Informationsgehalt:

to_pop = ["kepid", "kepoi_name", "kepler_name",
          "koi_teq_err1", "koi_teq_err2", "koi_tce_delivname"]  # 'rowid'

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


# Entfernung von Attributen, die Proxies des Zielwertes sind, bzw.  Flag-Werte sind.

x = data.drop(['koi_disposition', 'koi_pdisposition', 'koi_score', 'koi_fpflag_ss', 'koi_fpflag_co',
               'koi_fpflag_ec', 'koi_fpflag_nt'], axis=1)
y = data["koi_disposition"].to_numpy().reshape(-1, 1)


# Triviale Hypothese

predict = np.zeros(len(y))

cm = confusion_matrix(y, predict)
# tn, fp, fn, tp = cm.ravel()


# Optimierte Modelle

fit_knn = opt_knn.fit(x, y.ravel())
predict_knn = opt_knn.predict(x)
y_proba_knn = opt_knn.predict_proba(x)
accuracy_knn = accuracy_score(y, predict_knn)
cm_knn = confusion_matrix(y, predict_knn)

fit_svm = opt_svm.fit(x, y.ravel())
predict_svm = opt_svm.predict(x)
y_proba_svm = opt_svm.predict_proba(x)
accuracy_svm = accuracy_score(y, predict_svm)
cm_svm = confusion_matrix(y, predict_svm)

fit_rf = opt_rf.fit(x, y.ravel())
predict_rf = opt_rf.predict(x)
y_proba_rf = opt_rf.predict_proba(x)
accuracy_rf = accuracy_score(y, predict_rf)
cm_rf = confusion_matrix(y, predict_rf)


print("Base Rate: Akkuranz der Annahme, dass alle Ziele den häufigsten Wert, annehmen")

print("GESAMTER DATENSATZ")
print(f"Anzahl der KOI: {len(y)}")
print(f"Anzahl falsch positiver KOI: {len(np.where(y == 0)[0])}")
print(f"Anzahl der Kandidaten-KOI: {len(np.where(y == 1)[0])}")
print(f"Anzahl der bestätigten KOI: {len(np.where(y == 2)[0])}")
print(f"Base Rate: {len(np.where(y == 0)[0])/len(y):.4f}")


# Verwirrungsmatrix der Modelle

accuracy = (accuracy_knn, accuracy_svm, accuracy_rf)
cm = [cm_knn, cm_svm, cm_rf]
method_names = ("k Nearest Neighbours Classifer",
                "Support Vector Machine Classifier", "Random Forest Classifier")

fig = plt.figure(figsize=(20, 5))

for i in range(len(method_names)):

    ax = plt.subplot(1, 3, i+1)
    sns.heatmap(cm[i], annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted dispositions')
    ax.set_ylabel('True dispositions')
    ax.set_title(f"Confusion Matrix for the {method_names[i]}")
    ax.xaxis.set_ticklabels(['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'])
    ax.yaxis.set_ticklabels(['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'])

plt.savefig(f'data/figures/CM.png')


# Funktionen für die Erstellung von ROC-Kurven bei mehr als zwei Target-Klassen -
# get_all_roc_coordinates übernommen von Vinícius Trevisan  https://tinyurl.com/2s3nbavt

def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a treshold for the predicion of the class.\n",

    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.\n",

    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


# calculate_tpr_fpr übernommen von # Vinícius Trevisan  https://tinyurl.com/2s3nbavt

def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations

    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes

    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''
    # Calculates the confusion matrix and recover each element\n",
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    # Calculates tpr and fpr
    tpr = TP/(TP + FN)  # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP)  # 1-specificity - false positive rate

    return tpr, fpr


# plot_roc_curve übernommen von  # Vinícius Trevisan  https://tinyurl.com/2s3nbavt

def plot_roc_curve(tpr, fpr, scatter=True, ax=None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).

    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''
    if ax == None:
        plt.figure(figsize=(5, 5))
        ax = plt.axes()

    if scatter:
        sns.scatterplot(x=fpr, y=tpr, ax=ax)
    sns.lineplot(x=fpr, y=tpr, ax=ax)
    sns.lineplot(x=[0, 1], y=[0, 1], color='green', ax=ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


print("Histogramme und ROC-Kurven der disposition-Werte zueinander erstellen.")
# Codeblock angepasst von Vinícius Trevisan, "Multiclass classification evaluation with ROC Curves
# and ROC AUC", Towards Data Science, 12. Februar 2022,
# https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a
# Anfang angepasster Codeblock
class_combos = []
class_list = list(koi_disposition_dict.keys())
for i in range(len(class_list)):

    for j in range(i+1, len(class_list)):
        class_combos.append((class_list[i], class_list[j]))
        class_combos.append((class_list[j], class_list[i]))

# Wandle Liste in Array um, passe die Dimensionen so an, dass eine Transponierung der ersten beiden
# Achsen dazu führt, dass erst drei unterschiedliche Dispositions-Paare im Array stehen und dann die
# umgekehrten Paare, passe dann wieder Dimensionen an und wandle das Array dann wieder in Liste um.
class_combos = (np.asarray(class_combos).reshape(3, 2, 2)
                ).transpose(1, 0, 2).reshape(6, 2).tolist()

y_proba_all = np.array([y_proba_knn, y_proba_svm, y_proba_rf])
ml_method = ["KNN", "SVM", "RF"]

# Plots the Probability Distributions and the ROC Curves One vs One
bins = [i/20 for i in range(20)] + [1]
roc_auc_ovo = {}

for i in range(len(y_proba_all[:, 0, 0])):

    fig = plt.figure(figsize=(11.25, 15))
    print(f"ROC AUC für {ml_method[i]}-Methode")

    for j in range(len(class_combos)):

        # Gets the class
        combo = class_combos[j]
        c1 = combo[0]
        c2 = combo[1]
        c1_index = class_list.index(c1)
        c2_index = class_list.index(c2)
        title = f"{c1} vs {c2}"

        # Prepares an auxiliary dataframe to help with the plots
        df_aux = x.copy()
        df_aux['class'] = y
        df_aux['prob'] = y_proba_all[i, :, c1_index]

        # Slices only the subset with both classes
        df_aux = df_aux[(df_aux['class'] == c1_index) |
                        (df_aux['class'] == c2_index)]
        df_aux['class'] = [1 if y == c1_index else 0 for y in df_aux['class']]
        df_aux = df_aux.reset_index(drop=True)

        # Plots the probability distribution for the classes
        ax = plt.subplot(4, 3, j + 1)
        sns.histplot(x="prob", data=df_aux, hue='class',
                     color='b', ax=ax, bins=bins)

        ax.set_title(title)
        ax.legend([f"{c1}", f"{c2}"])
        ax.set_xlabel(f"P({c1})")

        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(4, 3, j+7)
        tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])

        plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
        ax_bottom.set_title(f"ROC Curve  {title}")
        fig.tight_layout()

        # Calculates the ROC AUC OvO
        roc_auc_ovo[title] = roc_auc_score(df_aux['class'], df_aux['prob'])

    # Ende angepasster Codeblock

    plt.savefig(f'data/figures/hist_roc_{ml_method[i]}.png')

    print(
        f"Plots gespeichert unter (f'data/figures/hist_roc_{ml_method[i]}.png')")
    print(f"ROC AUC für {ml_method[i]}-Methode")
    for j in range(len(class_combos)):

        print(
            f"ROC AUC von {list(roc_auc_ovo)[j]}: {list(roc_auc_ovo.values())[j]:.4f}")
    print(f"Mittelwert von ROC AUC: {np.mean(list(roc_auc_ovo.values())):.4f}")
    print("---------------------------------------------------")

# plt.tight_layout()
print("Histogramme und ROC-Kurven der disposition-Werte zueinander wurden erstellt.")


# Permutation Importance und Feature Importance bestimmen und anzeigen

print("Feature Importance (nur für Zufallswald) und Permutation Importance bestimmen und plotten.")

perm_imp_knn = permutation_importance(
    opt_knn, x, y, scoring='accuracy')
importance_knn = perm_imp_knn.importances_mean

perm_imp_svm = permutation_importance(
    opt_svm, x, y, scoring='accuracy')
importance_svm = perm_imp_svm.importances_mean

perm_imp_rf = permutation_importance(
    opt_rf, x, y, scoring='accuracy')
importance_rf = perm_imp_rf.importances_mean

x_train, x_test, y_train, y_test = train_test_split(x, y)


def plot_one_method(ax, title, x):
    ax.set_title(title)
    ticks = np.arange(1, len(x) + 1)
    ax.bar(ticks, x)
    ax.set_xticks(ticks=ticks, labels=list(x_train.columns), rotation=90)


fig, ax = plt.subplots(4, 1, figsize=(10, 14.14))
plot_one_method(
    ax[0], "Permutation Importance for the k Nearest Neighbours Classifier", importance_knn)
plot_one_method(
    ax[1], "Permutation Importance for the Support Vector Machine Classifier", importance_svm)

plot_one_method(
    ax[2], "Permutation Importance for the Random Forest Classifier", importance_rf)

plot_one_method(ax[3], "Feature Importance fpr the Random Forest Classifier",
                opt_rf.feature_importances_)
plt.tight_layout()
plt.savefig(f'data/figures/importances.png')
fig.tight_layout()

print("Permutation Importances und Feature Importance geplottet: data/figures/importances.png")
