# importing the necessary dependencies and libraries
from tkinter import E
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

#imports for flask and database
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import sqlite3


app = Flask(__name__)


@app.route('/')  # route to display the home page
@cross_origin()
def index():
    return render_template("home.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/index')
def getaprediction():
    return render_template("index.html")

@app.route('/accuracy')
def checkaccuracy():
    return render_template("accuracy.html")

@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            #_______________________ inputs from user will be stored here_____________________________#

            my_age = float(request.form['age'])
            my_Patient_RestingBP = int(request.form['Patient_RestingBP'])
            my_Patient_Cholesterol = int(request.form['Patient_Cholesterol'])
            my_Patient_MaxHR = int(request.form['Patient_MaxHR'])
            my_Patient_Oldpeak = float(request.form['Patient_Oldpeak'])
            ###############################################
            gender = request.form['gender']
            if (gender == 'male'):
                my_gender = 1
            else:
                my_gender = 0
            ##############################################
            Patient_ChestPainType = request.form['Patient_ChestPainType']
            if Patient_ChestPainType == "Asymptomatic":
                my_Patient_ChestPainType = 3
            elif Patient_ChestPainType == "Atypical Angina":
                my_Patient_ChestPainType = 2
            elif Patient_ChestPainType == "Non-aginal Pain":
                my_Patient_ChestPainType = 1
            else:
                my_Patient_ChestPainType = 0
            ##############################################
            Patient_ST_Slope = request.form['Patient_ST_Slope']
            if Patient_ST_Slope == 'Flat':
                my_Patient_ST_Slope = 0
            elif Patient_ST_Slope == 'Upsloping':
                my_Patient_ST_Slope = 1
            else:
                my_Patient_ST_Slope = 2
            ##############################################
            Patient_ExerciseAngina = request.form['Patient_ExerciseAngina']
            if Patient_ExerciseAngina == "Yes":
                my_Patient_ExerciseAngina = 1
            else:
                my_Patient_ExerciseAngina = 0
            ##############################################
            Patient_RestingECG = request.form['Patient_RestingECG']
            if Patient_RestingECG == "Normal":
                my_Patient_RestingECG = 0
            elif Patient_RestingECG == "ST":
                my_Patient_RestingECG = 1
            else:
                my_Patient_RestingECG = 2
            ##############################################
            Patient_FastingBS = request.form['Patient_FastingBS']
            if Patient_FastingBS == "<=120 mg/dl":
                my_Patient_FastingBS = 0
            else:
                my_Patient_FastingBS = 1
            ##############################################
            print(my_Patient_FastingBS,
                  my_Patient_RestingECG,
                  my_Patient_ExerciseAngina,
                  my_Patient_ST_Slope,
                  my_Patient_ChestPainType,
                  my_gender,
                  my_Patient_Oldpeak,
                  my_Patient_MaxHR,
                  my_Patient_Cholesterol,
                  my_Patient_RestingBP,
                  my_age
                  )
            
            #--------------------------------------backend---------------------------------------------------#

            data = pd.read_csv('dataset_new.csv')
            data1 = data.copy()
            data2 = pd.DataFrame(data1)
            df = data2.drop('Target', axis=1)
            df_labels = data2['Target'].copy()

            X_train, X_test, y_train, y_test = train_test_split(df, df_labels, train_size=0.8, random_state=42)

            scale_col = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

            num_pipeline = Pipeline([
                ('std_scaler', StandardScaler())
            ])

            full_pipeline = ColumnTransformer([
                ('num', num_pipeline, scale_col)
            ], remainder='passthrough')

            heart_prepared = full_pipeline.fit_transform(X_train)
            heart_test_prepared = full_pipeline.transform(X_test)

            # After performing hyperparameter tuning using GridSearchCV
            knn_best = KNeighborsClassifier(algorithm='auto',
                                            leaf_size=1,
                                            metric='minkowski',
                                            metric_params=None,
                                            n_jobs=None,
                                            n_neighbors=11,
                                            p=1,
                                            weights='uniform')

            # KNeighborsClassifier model building
            knn_best.fit(heart_prepared, y_train)

            # Performing cross validation to check how accuracy changes on random splitting w.r.t decided splits
            # print('Cross Validation Results: ', cross_val_score(knn_best, heart_prepared, y_train, cv = 3, scoring = 'accuracy'))
            # ------------------------------------------------------------------------------------------

            # def train_metrics():
            _pred = knn_best.predict(heart_prepared)
            _cm = confusion_matrix(y_train, _pred)
            # print('Confusion matrix of train data: ', _cm)
            # print('F1_score of train data: ', f1_score(y_train, _pred))
            print(' ')
            print('KNN accuracy on train data: ', ((_cm[0, 0] + _cm[1, 1]) / y_train.shape[0] * 100))

            # ------------------------------------------------------------------------------------------

            _pred1 = knn_best.predict(heart_test_prepared)
            _cm1 = confusion_matrix(y_test, _pred1)
            # print('Confusion matrix of test data: ', _cm1)
            # print('F1_score of test data: ',f1_score(y_test, _pred1))
            print(' ')
            print('KNN accuracy on test data: ', ((_cm1[0, 0] + _cm1[1, 1]) / y_test.shape[0]) * 100)

            # ------------------------------------------------------------------------------------------

            # LOGISTIC REGRESSION CLASSIFIER
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression()
            lr.fit(heart_prepared, y_train)
            lr_pred = lr.predict(heart_prepared)
            lr_cm = confusion_matrix(y_train, lr_pred)
            # print('Confusion matrix of train data: ', lr_cm)
            # print('F1_score of train data: ', f1_score(y_train, lr_pred))
            print(' ')
            print('Logistic Regression accuracy on the train data: ',
                  ((lr_cm[0, 0] + lr_cm[1, 1]) / y_train.shape[0]) * 100)

            lr_pred_test = lr.predict(heart_test_prepared)
            lr_cm1 = confusion_matrix(y_test, lr_pred_test)
            # print('Confusion matrix of train data: ', lr_cm1)
            # print('F1_score of train data: ', f1_score(y_test, lr_pred_test))
            print(' ')
            print('Logistic Regression accuracy on the test data: ',
                  ((lr_cm1[0, 0] + lr_cm1[1, 1]) / y_test.shape[0]) * 100)

            # ------------------------------------------------------------------------------------------

            # SVM
            from sklearn.svm import SVC
            from sklearn.metrics import accuracy_score
            svm_clf = SVC()
            svm_clf.fit(heart_prepared, y_train)
            svm_pred = svm_clf.predict(heart_prepared)
            svm_cm = confusion_matrix(y_train, svm_pred)
            # print('Confusion matrix of train data: ', svm_cm)
            # print('F1_score of train data: ', f1_score(y_train, svm_pred))
            print(' ')
            print('SVM accuracy on the train data: ', ((svm_cm[0, 0] + svm_cm[1, 1]) / y_train.shape[0]) * 100)

            svm_pred_test = svm_clf.predict(heart_test_prepared)
            svm_cm1 = confusion_matrix(y_test, svm_pred_test)
            # print('Confusion matrix of test data: ', svm_cm1)
            # print('F1_score of test data: ', f1_score(y_test, svm_pred_test))
            print(' ')
            print('SVM accuracy on the test data: ', ((svm_cm1[0, 0] + svm_cm1[1, 1]) / y_test.shape[0]) * 100)
            
            print(' ')
            print('Best Accuracy is of KNN on our dataset so we have proceeded with KNN')
            print(' ')


            ###########################################################
            data = [my_age, my_gender, my_Patient_ChestPainType, my_Patient_RestingBP,
                    my_Patient_Cholesterol, my_Patient_FastingBS, my_Patient_RestingECG,
                    my_Patient_MaxHR, my_Patient_ExerciseAngina, my_Patient_Oldpeak, my_Patient_ST_Slope]

            column = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                      'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

            data_df = pd.DataFrame([data], columns = column)

            data_prepared = full_pipeline.transform(data_df)

            result_final = knn_best.predict(data_prepared)
            # --output
            

            if result_final[0] == 0:
                target = 'Negative'
                to_print = 'The Patient seems normal, No need to worry '
            else:
                target = "Positive"
                to_print = 'The Patient have been diagnosed with heart problems. Please have a checkup. '
                
            
            try:
                connection_obj = sqlite3.connect('HDP.sqlite3')
                cur = connection_obj.cursor()
                
                # Creating table
                table = """ CREATE TABLE IF NOT EXISTS HDP_Patient_Data
                            (
                            Age CHAR(50) ,
                            Sex CHAR(50) ,
                            ChestPainType CHAR(50) ,
                            RestingBP CHAR(50) ,
                            Cholesterol CHAR(50) ,
                            FastingBS CHAR(50) ,
                            RestingECG CHAR(50) ,
                            MaxHR CHAR(50) ,
                            ExerciseAngina CHAR(50) ,
                            Oldpeak CHAR(50) ,
                            ST_Slope CHAR(50) ,
                            Target CHAR(50)
                            ); 
                        """
                cur.execute(table)
                
                to_insert = '''
                                INSERT INTO HDP_Patient_Data VALUES (?,?,?,?,?,?,?,?,?,?,?,?);
                            '''
                val_to_insert = (str(my_age), str(gender), str(Patient_ChestPainType), str(my_Patient_RestingBP),
                                str(my_Patient_Cholesterol), str(Patient_FastingBS), str(Patient_RestingECG), 
                                str(my_Patient_MaxHR), str(Patient_ExerciseAngina),
                                str(my_Patient_Oldpeak), str(Patient_ST_Slope), str(target))
                cur.execute(to_insert, val_to_insert)
                connection_obj.commit()
                print("Database Entry Successful")
            except Exception as e:
                print("database error"+"\n"+"*"*50)
                print(e)
                

            return render_template("results.html", to_print = to_print)

        except Exception as e:
            print(e)




if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8001, debug=True)
    #app.run(debug=True)
