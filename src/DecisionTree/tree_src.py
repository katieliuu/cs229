import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay, precision_recall_curve
import matplotlib.pyplot as plt
import util



def decision_tree(X_train, y_train, max_depth=None, min_samples_split=2, min_samples_leaf=1, sample_weight=None):
    model = DecisionTreeClassifier(random_state=3, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    #train model
    model.fit(X_train, y_train, sample_weight=sample_weight)
    train_accuracy = model.score(X_train, y_train)
    return model, train_accuracy
    
def test_tree(model, X_test, y_test):
    #y_pred = model.predict(X_test)
    #test_accuracy = model.score(X_test, y_test)
    probs = model.predict_proba(X_test)[:, 1] #probability of being diabetic column
    return probs

def print_results(y_test, y_probs, threshold):
    print(y_probs[0:49])
    #f1 score
    f1, y_pred = util.f1_from_probs(y_test, y_probs, threshold)
    print(f'F1 Score: {f1}')
    
    #Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)#, labels=['Diabetic', 'Non-Diabetic'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Diabetic', 'Non-Diabetic'])
    disp.plot()
    plt.show()
    
    #Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    
    #Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    plt.plot(recall, precision, marker='o')
    for i, t in enumerate(thresholds):
        plt.annotate(f"{t:.2f}", (recall[i+1], precision[i+1]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve with Thresholds')
    plt.show()
 
    
# f1 score