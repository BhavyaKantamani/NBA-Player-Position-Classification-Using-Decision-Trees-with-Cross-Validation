import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def main():
    
    data=pd.read_csv("nba_stats.csv")
    
    data=data[data['MP']>=5] # as stats of players with limited minutes played are less indicative of their true characteristics
    
    X = data.drop(columns=['Pos','Age'])  # Features 
    y = data['Pos']  # Target
    
    train_feature, val_feature, train_class, val_class = train_test_split(X,y,test_size=0.2,stratify=y, random_state=0)
    
    
    
    tree = DecisionTreeClassifier(max_depth=9, random_state=0)
    tree.fit(train_feature, train_class)
    
    
    print("\nTask 1:")
    print("\nTraining set Accuracy: {:.3f}".format(tree.score(train_feature, train_class)))
    train_prediction=tree.predict(train_feature)
    print("Training set Confusion matrix:")
    print(pd.crosstab(train_class, train_prediction, rownames=['True'], colnames=['Predicted'], margins=True))
    
    
    
    print("\nValidation set Accuracy: {:.3f}".format(tree.score(val_feature, val_class)))

    prediction = tree.predict(val_feature)
    print("Validation set Confusion matrix:")
    print(pd.crosstab(val_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))
    
    
    
    test_data=pd.read_csv("dummy_test.csv")
    
    
    
    X_test=test_data.drop(columns=['Pos','Predicted Pos','Age'])
    y_test=test_data['Pos']

    
    print("\nTask 2:")
    print("\ndummy_test set Accuracy: {:.3f}".format(tree.score(X_test, y_test)))

    prediction = tree.predict(X_test)
    print("dummy_test set Confusion matrix:")
    print(pd.crosstab(y_test, prediction, rownames=['True'], colnames=['Predicted'], margins=True))
    
    
    
    scores = cross_val_score(DecisionTreeClassifier(max_depth=9, random_state=0), X, y, cv=10)
    print("\nTask 3:")
    print("\nCross-validation scores: {}".format(scores))
    print("Average cross-validation score: {:.2f}".format(scores.mean()))
    
    
if __name__=="__main__":
    
    main()
    
    