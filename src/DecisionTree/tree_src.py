import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score



def decision_tree(X_train, y_train, max_depth=None, min_samples_split=2, min_samples_leaf=1, sample_weight=None):
    model = DecisionTreeClassifier(random_state=42, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    #train model
    model.fit(X_train, y_train, sample_weight=sample_weight)
    train_accuracy = model.score(X_train, y_train)
    return model, train_accuracy
    
def test_tree(model, X_test, y_test):
    y_pred = model.predict(X_test)
    test_accuracy = model.score(X_test, y_test)
    return y_pred, test_accuracy

def print_results(y_test, y_pred):
    # 7. Print results
    '''
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    '''
    print("F1 Score:", f1_score(y_test, y_pred))
    
# f1 score