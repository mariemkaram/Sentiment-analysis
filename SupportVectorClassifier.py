import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics

import sklearn.metrics
# % matplotlib
# inline
# import some data to play with
import time

class SVClassifier:
    def S_V_C(self,X_train,X_test,Y_train,Y_test):
# we create an instance of SVM and fit out data.
        C = 1  # SVM regularization parameter
        train_time=list()
        stat_time=time.time()
        svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
        end_time=time.time()
        training_time = end_time - stat_time
        train_time.append(training_time)
        print("trainig time for linear SVC",training_time)
        stat_time = time.time()
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.2, C=C).fit(X_train, Y_train)
        end_time=time.time()
        training_time = end_time - stat_time
        train_time.append(training_time)
        print("trainig time for rbf SVC",training_time)
        stat_time = time.time()
        poly_svc = svm.SVC(kernel='poly', degree=2, C=C).fit(X_train, Y_train)
        end_time=time.time()
        training_time = end_time - stat_time
        train_time.append(training_time)
        print("trainig time for poly SVC",training_time)
        # create a mesh to plot in


        # title for the plots
        titles = ['SVC with linear kernel',
                  'SVC with RBF kernel',
                  'SVC with polynomial (degree 3) kernel']

        accuracy=list()
        test_time=list()
       # # create a mesh to plot in
       #  x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
       #  y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
       #  h = .02  # step size in the mesh
       #  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
       #               np.arange(y_min, y_max, h))
        for i, clf in enumerate((svc, rbf_svc, poly_svc)):
            print(clf)
            stat_time = time.time()
            predictions = clf.predict(X_test)
            end_time=time.time()
            testing_time=end_time-stat_time
            test_time.append(testing_time)
            print("testing time for "+clf.kernel, end_time - stat_time)
            acc = accuracy_score(Y_test, predictions)
            if clf.kernel == 'linear':
                accuracy=acc
            if clf==svc:
                plt.bar(['SVM Linear'], [acc * 100])
            elif clf==rbf_svc:
                plt.bar(['SVM rbf'], [acc * 100])
            else:
                plt.bar(['SVM poly'], [acc * 100])


            print("accuracy: ", acc)











        logreg = LogisticRegression()
        logreg.fit(X_train, Y_train)
        logreg_pred = logreg.predict(X_test)
        accuracy = list()
        test_time = list()
        acc1 = accuracy_score(Y_test, logreg_pred)
        print("logistic accuracy : ",acc1)


        print("logistic Confuison matrix : ", metrics.confusion_matrix(Y_test, logreg_pred))
        print("logistic Confuison Report : \n", metrics.classification_report(Y_test, logreg_pred))
        #plotting a bar chart
        plt.bar(['logistic Regression'], [acc1*100])


        # Add labels and title
        plt.xlabel('Classifier')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of News Article Classification')

        # Show theplot

        # labels = np.array(['log','svc'])
        # perc = np.array([acc, 100 - acc * 100])
        #
        # plt.figure(figsize=(7, 7))
        # plt.pie(perc, labels=labels, autopct='%1.1f%%', startangle=90)
        # plt.show()

        return accuracy,train_time,test_time


# title = ('Decision surface of linear SVC ')
# print(X_test.shape)
# x_min, x_max = X_test.iloc[:, 0].min() - 1, X_test.iloc[:, 0].max() + 1
# y_min, y_max = X_test.iloc[:, 1].min() - 1, X_test.iloc[:, 1].max() + 1
# h = .02  # step size in the mesh
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
#
# fig, ax = plt.subplots()
# plt.subplots_adjust(wspace=0.4, hspace=0.4)
# predictions = svc.predict(np.c_[xx.ravel(), yy.ravel()])
# # Put the result into a color plot
# Z = predictions.reshape(xx.shape)
# plt.contourf(ax, svc, xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
#
# plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=plt.cm.coolwarm)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
# plt.title(title)
#
# plt.show()
#
# # stat_time = time.time()
# # rbf_svc = svm.SVC(kernel='rbf', gamma=0.1, C=C).fit(X_train, Y_train)
# # end_time=time.time()
# # training_time = end_time - stat_time
# # train_time.append(training_time)
# # print("trainig time for rbf SVC",training_time)
# # stat_time = time.time()
# # poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, Y_train)
# # end_time=time.time()
# # training_time = end_time - stat_time
# # train_time.append(training_time)
# # print("trainig time for poly SVC",training_time)
# # create a mesh to plot in
#
#################################33
#   logreg=LogisticRegression()
#         logreg.fit(X_train,Y_train)
#         logreg_pred=logreg.predict(X_test)
#         accuracy=list()
#         test_time=list()
#         acc= accuracy_score(Y_test, logreg_pred)
# train_time = list()
# stat_time = time.time()
# # svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
# # rbf_svc = svm.SVC(kernel='rbf', gamma=.8, C=C).fit(X_train, Y_train)
#
# print("accuracy: ", acc)
# # print("accuracy: ", acc)
#
# end_time = time.time()
# training_time = end_time - stat_time
# train_time.append(training_time)
# print("trainig time for linear SVC", training_time)