import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import time

def load_data(choice):
    if choice.lower() == 'iris':
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
    else:
        print("You chose to use random dataset.")
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100)

    return X, y

def main():
    print("Choose dataset:\n1. Iris\n2. Random dataset")
    choice = input("Enter your choice (1/2): ")

    if choice == '1':
        dataset_choice = 'iris'
    elif choice == '2':
        dataset_choice = 'your_own'
    else:
        print("Invalid choice. Using Iris dataset by default.")
        dataset_choice = 'iris'

    X, y = load_data(dataset_choice)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    nb_classifier = GaussianNB()
    dt_classifier = DecisionTreeClassifier()

    # Training time measurement for Naive Bayes
    start_time = time.time()
    nb_classifier.fit(X_train, y_train)
    nb_train_time = time.time() - start_time

    # Training time measurement for Decision Tree
    start_time = time.time()
    dt_classifier.fit(X_train, y_train)
    dt_train_time = time.time() - start_time

    # Prediction time measurement for Naive Bayes
    start_time = time.time()
    nb_predictions = nb_classifier.predict(X_test)
    nb_prediction_time = time.time() - start_time

    # Prediction time measurement for Decision Tree
    start_time = time.time()
    dt_predictions = dt_classifier.predict(X_test)
    dt_prediction_time = time.time() - start_time

    nb_accuracy = accuracy_score(y_test, nb_predictions)
    dt_accuracy = accuracy_score(y_test, dt_predictions)

    print("Naive Bayes Accuracy:", nb_accuracy)
    print("Decision Tree Accuracy:", dt_accuracy)
    print("Naive Bayes Training Time:", nb_train_time, "seconds")
    print("Decision Tree Training Time:", dt_train_time, "seconds")
    print("Naive Bayes Prediction Time:", nb_prediction_time, "seconds")
    print("Decision Tree Prediction Time:", dt_prediction_time, "seconds")

    labels = ['Naive Bayes', 'Decision Tree']
    accuracies = [nb_accuracy, dt_accuracy]
    plt.bar(labels, accuracies)
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Classification Algorithms')
    plt.show()

if __name__ == "__main__":
    main()
