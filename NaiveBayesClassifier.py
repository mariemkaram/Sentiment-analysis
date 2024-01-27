from sklearn.datasets import load_iris
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt

class Naive_Bayes:
    # from matplotlib import pyplot
    # def train_model(self,classifier,X_train,X_test,Y_train,Y_test):
    #     # fit the training dataset on the classifier
    #     classifier.fit(X_train, Y_train)
    #     # predict the labels on validation dataset
    #     predictionTRAIN=classifier.predict(X_train)
    #     predictions = classifier.predict(X_test)
    #     return metrics.accuracy_score(predictions, Y_test),metrics.accuracy_score(predictionTRAIN, Y_train)

    def naivebayes(self,X_train,X_test,Y_train,Y_test):
        gnb = MultinomialNB(alpha=0.5)
        naive_bayes =gnb.fit (X_train,Y_train)


        # making predictions on the testing set
        y_pred = naive_bayes.predict(X_test)

        # comparing actual response values (y_test) with predicted response values (y_pred)
        print("Gaussian Naive Bayes model accuracy:", metrics.accuracy_score(Y_test, y_pred))
        plt.bar(['Naive Bayes'], [metrics.accuracy_score(Y_test, y_pred)*100])
        print("Naive Bayes Confuison matrix : ",metrics.confusion_matrix(Y_test,y_pred))
        print("Naive Bayes Confuison Report : \n", metrics.classification_report(Y_test, y_pred))
        print("Naive Bayes Confuison matrix : ", metrics.confusion_matrix(Y_test, y_pred))
        # alphes = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # train_scores = list()
        # test_scores = list()
        # for i in alphes:
        #     accuracy,acc=self.train_model(MultinomialNB(alpha=i),X_train,X_test,Y_train,Y_test)
        #     print("Accuracy: ",i," " , accuracy ," ", acc)
        #     train_scores.append(acc)
        #     test_scores.append(accuracy)

        # plt.plot(alphes, train_scores, '-o', label='Train')
        # plt.plot(alphes, test_scores, '-o', label='Test')
        # plt.legend()
        # plt.show()
        #


