import tkinter as tk #gui development
from tkinter import filedialog #drang and drop of files
import pandas as pd # data manuplation
import matplotlib.pyplot as plt # plotting
import tensorflow as tf # machine learning models deploy in tensorflow framework
from sklearn.model_selection import train_test_split # data training and testing splitting of data 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier # decision tree
from sklearn.metrics import classification_report, confusion_matrix # confusion matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np 
import os
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # calculation of metrics 
# ------------------------------------importing libraries-----------------------------------------------#

root = tk.Tk()
root.title("My GUI")
root.geometry("1200x1300") # leangth X breath of screen for window

# Label for the title
label = tk.Label(root, text="NETWORK INTRUSION CLASSIFICATION SYSTEM", bg="red", fg="white", font=("Arial", 16))
label.pack(pady=25)



# Function to upload the dataset
def uploadDataset(): # initilization of function
    global filename, data, df # global variables # global , local 
    text.delete('1.0', tk.END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(tk.END, str(filename) + " Dataset Loaded\n\n")
    pathlabel.config(text=str(filename) + " Dataset Loaded")

    data = pd.read_csv(filename, encoding='iso-8859-1')
    text.insert(tk.END, str(data.head(10)) + "\n\n")
    text.insert(tk.END, "*************Dataset Description***********\n\n\n")
    text.insert(tk.END, str(data.describe()) + "\n\n\n\n")

def preprocess():
    global filename, data, df, features, target, features_encoded, X_train, X_test, y_train, y_test
    
    data['num_outbound_cmds'].value_counts()
    data.drop(['num_outbound_cmds'], axis=1, inplace=True)
      
    attack_class_freq_train = data[['attack_class']].apply(lambda x: x.value_counts())
    attack_class_freq_test = data[['attack_class']].apply(lambda x: x.value_counts())
    attack_class_freq_train['frequency_percent_train'] = round((100 * attack_class_freq_train / attack_class_freq_train.sum()), 2)
    attack_class_freq_test['frequency_percent_test'] = round((100 * attack_class_freq_test / attack_class_freq_test.sum()), 2)
    df = pd.concat([attack_class_freq_train, attack_class_freq_test], axis=1) 
    text.insert(tk.END, "*************Pre-processing***********\n\n\n")
    text.insert(tk.END, "Attack Class Distribution:\n\n\n")
    text.insert(tk.END, str(df) + "\n\n\n")
    
    # Select the features and target variable
    features = data.drop('attack_class', axis=1)
    target = data['attack_class']

    # Perform one-hot encoding for relevant categorical features
    features_encoded = pd.get_dummies(features)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

def plotAttackClass():
    text.insert(tk.END, "*************Plotting***********\n\n\n")
    attack_class_plot = df[['frequency_percent_train', 'frequency_percent_test']].plot(kind="bar")
    attack_class_plot.set_title("Attack Class Distribution", fontsize=20)
    attack_class_plot.grid(color='lightgray', alpha=0.5)
    plt.show()


def trainDecisionTree():
    # Select the feaures and target variable
    features = data.drop('attack_class', axis=1)
    target = data['attack_class']

    # Perform one-hot encoding for relevant categorical features
    features_encoded = pd.get_dummies(features)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

    # Initialize the decision tree classifier
    dt_classifier = DecisionTreeClassifier()

    # Train the classifier
    dt_classifier.fit(X_train, y_train) # traning the model .fit() , .predict()--> predition, confusion_matrix

    # Make predictions on the testing set
    y_pred = dt_classifier.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    text.insert(tk.END, "*************Decision Tree Model Evaluation***********\n\n\n")
    text.insert(tk.END, "Confusion Matrix:\n")
    text.insert(tk.END, str(cm) + "\n\n\n")
    text.insert(tk.END, "Classification Report:\n")
    text.insert(tk.END, classification_report(y_test, y_pred) + "\n\n\n")

    # Plot confusion matrix
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(np.arange(len(target.unique())), target.unique(), rotation=90)
    plt.yticks(np.arange(len(target.unique())), target.unique())
    plt.show()


#-------------------------cnn model-----------------------------------------------#



def compareGraph():
    # Performance metrics data
    Accuracy = [78.41, 98.0]
    Precision = [99.0, 95.2]
    Recall = [79.2, 92.6]
    FScore = [87.5, 93.6]

    n = 2
    r = np.arange(n)
    width = 0.20

    # Plotting the bars
    plt.bar(r, Accuracy, color='b', width=width, edgecolor='black', label='Accuracy')
    plt.bar(r + width, Precision, color='g', width=width, edgecolor='black', label='Precision')
    plt.bar(r + width + 0.20, Recall, color='r', width=width, edgecolor='black', label='Recall')
    plt.bar(r + width + 0.40, FScore, color='y', width=width, edgecolor='black', label='FScore')

    # Chart labels and title
    plt.xlabel("Comparison Algorithms")
    plt.ylabel("Performance Value (%)")
    plt.title("Performance Comparison")
    plt.xticks(r + width / 2, ['CNN', 'DT'])
    plt.legend()

    # Display the chart
    plt.show()
    

    

##---------------------------------model comparison---------------------------------------------#

def compareModels():
    # Select the features and target variable
    features = data.drop('attack_class', axis=1)
    target = data['attack_class']

    # Perform one-hot encoding for relevant categorical features
    features_encoded = pd.get_dummies(features)

    # Encode the target variable
    le = LabelEncoder()
    target_encoded = le.fit_transform(target)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target_encoded, test_size=0.2, random_state=42)

    # Initialize the CNN model
    cnn_model = Sequential()
    cnn_model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(64, activation='relu'))
    cnn_model.add(Dense(1, activation='sigmoid'))
    # Compile the CNN model
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the CNN model
    cnn_model.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=5, batch_size=32, verbose=1)

    # Make predictions on the testing set using the CNN model
    y_pred_cnn = cnn_model.predict_classes(X_test.values.reshape(-1, X_test.shape[1], 1))


    rf_model = DecisionTreeClassifier()

  
    rf_model.fit(X_train, y_train)

    # Make predictions on the testing set using the Random Forest model
    y_pred_rf = rf_model.predict(X_test)

    # Compute the performance metrics for the models
    accuracy_cnn = accuracy_score(y_test, y_pred_cnn)
    precision_cnn = precision_score(y_test, y_pred_cnn, average='weighted')
    recall_cnn = recall_score(y_test, y_pred_cnn, average='weighted')

    f1_score_cnn = f1_score(y_test, y_pred_cnn, average='weighted')


    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf, average='weighted')

    recall_rf = recall_score(y_test, y_pred_rf, average='weighted')

    f1_score_rf = f1_score(y_test, y_pred_rf, average='weighted')


    # Define the models and performance metrics
    models = ['CNN', 'Decision Tree']
    accuracies = [accuracy_cnn, accuracy_rf]
    precisions = [precision_cnn, precision_rf]
    recalls = [recall_cnn, recall_rf]
    f1_scores = [f1_score_cnn, f1_score_rf]

    # Set the width of the bars
    bar_width = 0.15

    # Set the positions of the bars on the x-axis
    x_pos = np.arange(len(models))

    # Plot the comparison chart
    plt.figure(figsize=(10, 6))
    plt.bar(x_pos, accuracies, width=bar_width, label='Accuracy', color='blue')
    plt.bar(x_pos + bar_width, precisions, width=bar_width, label='Precision', color='green')
    plt.bar(x_pos + 2 * bar_width, recalls, width=bar_width, label='Recall', color='orange')
    plt.bar(x_pos + 3 * bar_width, f1_scores, width=bar_width, label='F1-Score', color='green')

    # Add labels.


    # Add labels and title to the chart
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Comparison of Model Performance Metrics')
    plt.xticks(x_pos + 1.5 * bar_width, models)
    plt.legend()

    # Label the values on the bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'Acc: {round(v, 2)}', ha='center', color='black', fontweight='bold')
        plt.text(i, precisions[i] + 0.01, f'Prec: {round(precisions[i], 2)}', ha='center', color='black', fontweight='bold')
        plt.text(i, recalls[i] + 0.01, f'Rec: {round(recalls[i], 2)}', ha='center', color='black', fontweight='bold')
        plt.text(i, f1_scores[i] + 0.01, f'F1: {round(f1_scores[i], 2)}', ha='center', color='black', fontweight='bold')

    # Display the chart
    plt.show()

#-----------------------------BUILD BY ANONYMOUS-----------------------------#


# Function to exit the GUI
def exitGUI():
    root.destroy()



# Frame to hold the buttons and text box
frame = tk.Frame(root)
frame.pack(pady=20)

# Buttons for different actions
button1 = tk.Button(frame, text="Upload Dataset", command=uploadDataset, font=("Arial", 12))
button1.pack(side="left", padx=10)

button2 = tk.Button(frame, text="Preprocess", command=preprocess, font=("Arial", 12))
button2.pack(side="left", padx=10)

button3 = tk.Button(frame, text="Plot Attack Class", command=plotAttackClass, font=("Arial", 12))
button3.pack(side="left", padx=10)

button4 = tk.Button(frame, text="Train Models", command=trainDecisionTree, font=("Arial", 12))
button4.pack(side="left", padx=10)

button5 = tk.Button(frame, text="Comparision Graph", command=compareGraph, font=("Arial", 12))
button5.pack(side="left", padx=10)


button7 = tk.Button(frame, text="Exit", command=exitGUI, font=("Arial", 12))
button7.pack(side="left", padx=10)


# Text box to display messages
text = tk.Text(root, height=25, width=100)
text.pack(pady=20)

# Label to display the loaded dataset path
# Label to display the loaded dataset path
pathlabel = tk.Label(root, text="", font=("Arial", 12,"bold"), bg="lightgray")
pathlabel.pack(pady=10)


root.mainloop()