import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

##  Function to calculate different metric scores of the model - Accuracy, Recall and Precision
def get_metrics_score(model,X_train,X_test,y_train,y_test,flag=True,return_df=True):
    '''
    model : classifier to predict values of X

    '''
    # defining an empty list to store train and test results
    score_list=[] 
    
    pred_train = model.predict(X_train)
    pred_train_proba = model.predict_proba(X_train)[:,1]
    pred_test = model.predict(X_test)
    pred_test_proba = model.predict_proba(X_test)[:,1]
    
    train_acc = metrics.accuracy_score(y_train,pred_train)
    test_acc = metrics.accuracy_score(y_test,pred_test)
    
    train_recall = metrics.recall_score(y_train,pred_train)
    test_recall = metrics.recall_score(y_test,pred_test)
    
    train_precision = metrics.precision_score(y_train,pred_train)
    test_precision = metrics.precision_score(y_test,pred_test)

    train_f1_score = metrics.f1_score(y_train,pred_train)
    test_f1_score = metrics.f1_score(y_test,pred_test)

    train_auc_score = metrics.roc_auc_score(y_train,pred_train_proba)
    test_auc_score = metrics.roc_auc_score(y_test,pred_test_proba)
    
    train_log_loss = metrics.log_loss(y_train,pred_train_proba,eps=1e-7)
    test_log_loss = metrics.log_loss(y_test,pred_test_proba,eps=1e-7)
                                              
    score_list.extend((train_acc,test_acc,train_recall,test_recall,train_precision,test_precision,
                       train_f1_score,test_f1_score,train_auc_score,test_auc_score,train_log_loss,test_log_loss))
    
    scores_df = pd.DataFrame(data=np.array(score_list).reshape((1, 12)),columns=['Accuracy Training', 'Accuracy Test',
                                           'Recall Training', 'Recall Test','Precision Training', 'Precision Test',
                                           'F1 Training', 'F1 Test', 'AUC Training', 'AUC Test', 'Logloss Training', 'Logloss Test'])
    
    # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if flag == True: 
        print("Accuracy on training set : ",train_acc)
        print("Accuracy on test set : ",test_acc)
        print("Recall on training set : ",train_recall)
        print("Recall on test set : ",test_recall)
        print("Precision on training set : ",train_precision)
        print("Precision on test set : ",test_precision)
        print("F1 score on training set : ",train_f1_score)
        print("F1 score on test set : ",test_f1_score)
        print("AUC score on training set : ",train_auc_score)
        print("AUC score on test set : ",test_auc_score)
        print("Log loss on training set : ",train_log_loss)
        print("Log loss on test set : ",test_log_loss)
    if return_df == True:
        return scores_df
    else:
        return score_list
    
## Function to create confusion matrix
def make_confusion_matrix(model,X_test,y_actual,labels=[1, 0]):
    '''
    model : classifier to predict values of X
    y_actual : ground truth  
    
    '''
    y_predict = model.predict(X_test)
    cm=metrics.confusion_matrix( y_actual, y_predict, labels=[0, 1])
    df_cm = pd.DataFrame(cm, index = [i for i in ["Actual - No","Actual - Yes"]],
                  columns = [i for i in ['Predicted - No','Predicted - Yes']])
    group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=labels,fmt='')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')