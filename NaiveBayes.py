import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df = pd.read_csv("C:/Users/Admin/OneDrive/Desktop/naive byes/TrainingDataset.csv")
df.Age = df.Age.astype(str)
df.height = df.height.astype(str)
df.weight = df.weight.astype(str)
df.QRS_duration = df.QRS_duration.astype(str)
df.T_interval = df.T_interval.astype(str)
df.P_interval = df.P_interval.astype(str)
df.QRS = df.QRS.astype(str)
df.T = df.T.astype(str)
df.P = df.P.astype(str)
df.QRST = df.QRST.astype(str)
df.J= df.J.astype(str)
df.HR = df.HR.astype(str)
df.DI_Q_wave = df.DI_Q_wave.astype(str)
df.DI_R_wave = df.DI_R_wave.astype(str)
df.DI_S_wave = df.DI_S_wave.astype(str)


for column in df.columns:
    temp_new = le.fit_transform(df[column].astype('category'))
    df.drop(labels=[column], axis="columns", inplace=True)
    df[column] = temp_new
feature_col_names = ['Age','sex','height','weight','QRS_duration','P-R_interval','Q-T_interval','T_interval','P_interval','QRS','T','P','QRST','J','HR','DI_Q_wave','DI_R_wave','DI_S_wave','DI_R`_wave','DI_S`_wave']
predicted_class_names = ['Result']

X = df[feature_col_names].values # these are factors for the prediction
y = df[predicted_class_names].values # this is what we want to predict

#splitting the dataset into train and test data

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.40)

print (xtrain)
print ('\n the total number of Test Data :',ytest.shape)


# Training Naive Bayes (NB) classifier on training data.

clf = GaussianNB().fit(xtrain,ytrain.ravel())
predicted = clf.predict(xtest)
predictTestData= clf.predict([[13,0,169,51,100,167,321,174,91,107,66,52,88,0,84,0,36,48,0,0]])

#printing Confusion matrix, accuracy, Precision and Recall

print('\n Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))

print('\n Accuracy of the classifier is',metrics.accuracy_score(ytest,predicted))
from sklearn.ensemble import RandomForestClassifier
clf4 = RandomForestClassifier()
clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
from sklearn.metrics import accuracy_score
ypred=clf4.predict(xtest)
print(accuracy_score(ytest, ypred))
print(accuracy_score(ytest, ypred,normalize=False))


from sklearn import tree

clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
from sklearn.metrics import accuracy_score
y_pred=clf3.predict(xtest)
print(accuracy_score(ytest, y_pred))
print()
print(accuracy_score(ytest, y_pred,normalize=False))


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
from sklearn.metrics import accuracy_score
y_ppred=gnb.predict(xtest)
print(accuracy_score(ytest, y_ppred))
print(accuracy_score(ytest, y_ppred,normalize=False))



def predict_note_authentication(Age,sex,height,weight,QRS_duration,P-R_interval,Q-T_interval,T_interval,P_interval,QRS,T,P,QRST,J,HR,DI_Q_wave,DI_R_wave,DI_S_wave,DI_R`_wave,DI_S`_wave):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Age
        in: query
        type: number
        required: true
      - name: sex
        in: query
        type: number
        required: true
      - name: height
        in: query
        type: number
        required: true
      - name: weight
        in: query
        type: number
        required: true
      - name: variance
        in: QRS_duration
        type: number
        required: true
      - name: P-R_interval
        in: query
        type: number
        required: true
      - name: Q-T_interval
        in: query
        type: number
        required: true
      - name: T_interval
        in: query
        type: number
        required: true
      - name: P_interval
        in: query
        type: number
        required: true
      - name: QRS
        in: query
        type: number
        required: true
      - name: T
        in: query
        type: number
        required: true
      - name: P
        in: query
        type: number
        required: true
      - name: QRST
        in: query
        type: number
        required: true
      - name: HR
        in: query
        type: number
        required: true
      - name: DI_Q_wave
        in: query
        type: number
        required: true
      - name: DI_R_wave
        in: query
        type: number
        required: true
      - name: DI_S_wave
        in: query
        type: number
        required: true
      - name: DI_R`_wave
        in: query
        type: number
        required: true
      - name: DI_S`_wave
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=clf4.predict([[Age,sex,height,weight,QRS_duration,P_R_interval,Q_T_interval,T_interval,P_interval,QRS,T,P,QRST,J,HR,DI_Q_wave,DI_R_wave,DI_S_wave,DI_R`_wave,DI_S`_wave]])
    print(prediction)
    return prediction



def main():
    st.title("Cardiac disease prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Age = st.text_input("Age","Type Here")
    sex = st.text_input("sex","Type Here")
    height = st.text_input("height","Type Here")
    weight = st.text_input("weight","Type Here")
    QRS_duration = st.text_input("QRS_duration","Type Here")
    P_R_interval = st.text_input("P_R_interval","Type Here")
    Q_T_interval = st.text_input("Q_T_interval","Type Here")
    T_interval = st.text_input("T_interval","Type Here")
    P_interval = st.text_input("P_interval","Type Here")
    QRS = st.text_input("QRS","Type Here")
    T = st.text_input("T","Type Here")
    P = st.text_input("P","Type Here")
    QRST = st.text_input("QRST","Type Here")
    J = st.text_input("J","Type Here")
    HR = st.text_input("HR","Type Here")
    DI_Q_wave = st.text_input("DI_Q_wave","Type Here")
    DI_R_wave = st.text_input("DI_R_wave","Type Here")
    DI_S_wave = st.text_input("DI_S_wave","Type Here")
    DI_R`_wave = st.text_input("DI_R`_wave","Type Here")
    DI_S`_wave = st.text_input("DI_S`_wave","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(Age,sex,height,weight,QRS_duration,P-R_interval,Q-T_interval,T_interval,P_interval,QRS,T,P,QRST,J,HR,DI_Q_wave,DI_R_wave,DI_S_wave,DI_R`_wave,DI_S`_wave)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()