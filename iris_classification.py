'''Iris flower has three species; setosa, versicolor and verginicaa which
differs according to their measurements. Now assume that you have the measurements
of the Iris flower according to their species, and here your task is to train a machine 
training model that can learn from the measurement of the iris species and classify them. '''

# OASIS INFOBYTE TASK LEVEL --1
# SO LETS GET STARTED


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

# Load the Iris dataset
data = pd.read_csv('Iris.csv')

# Seperate the features and target variable
X = data.drop('Species', axis=1)
y = data['Species']

# Split the data into traning and testing sets 
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#train the model
model = LogisticRegression()
model.fit(X_train,y_train)

# Predict the target variable for the test set 
y_pred = model.predict(X_test)

#Evaluate the model
print(classification_report(y_test,y_pred))
confusion = confusion_matrix(y_test,y_pred)

# Create a graph
plt.figure(figsize=(8,6))
plt.imshow(confusion,cmap=plt.cm.Blues)
plt.title('confusion Matrix')
plt.colorbar()
classes = ['Setosa', 'Versicolor', 'Virginica']
tick_marks = [0,1,2]
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks,classes)

# Add labels to the graph 
thresh = confusion.max()/2
for i in range (confusion.shape[0]):
    for j in range (confusion.shape[1]):
        plt.text(j,i, format(confusion[i,j],'d'),
                horizontalalignment = 'center',
                color="white" if confusion[i,j] > thresh else "black")
        
plt.xlabel('predicted label')
plt.ylabel('True label')
plt.tight_layout()

#Show the graph
plt.show()


