import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score 



def main():
    st.title("Binary Classification wejjb APP")    
    st.sidebar.title("Binary Classification Web app")
    st.markdown("Are your Mashrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your Mashrooms edible or poisonous? 🍄")

    
    # we dont want our app to reload the data every time we run the app, 
    # so we will use a cache: see the code line below
    @st.cache_data(persist=True)
    def load_data():
        url = ".\mushrooms.csv"
        data = pd.read_csv(url)
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    

    @st.cache_data(persist= True)
    def split(df):
        y = df.type
        x = df.drop(columns =['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        # return x_train, x_test, y_train, y_test
        return x_train, x_test, y_train.to_numpy().ravel().copy(), y_test.to_numpy().ravel().copy()


    def plot_metrics(metric_list):
        if 'confusion_matrix' in metric_list:
            st.subheader("Confusion Matrix")
            cm = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names)
            st.pyplot(cm.figure_)
        if 'roc_curve' in metric_list:
            st.subheader("ROC Curve")
            roc = RocCurveDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(roc.figure_)
        if 'precision_recall_curve' in metric_list:
            st.subheader("Precision-Recall Curve")
            prc = PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(prc.figure_)
        

        

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine", "Logistic Regression", "Random Forest"))
    
    if classifier == "Support Vector Machine":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, value=1.0)
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"),key = 'kernel')
        gamma = st.sidebar.radio("Gamma", ("scale", "auto"), key = "gamma")
        metrics = st.sidebar.multiselect("What metrics to plot?", ("confusion_matrix", "roc_curve", "precision_recall_curve"), key = 'metrics')  
        if st.sidebar.button("Classify", key = "cllassify"):
            st.subheader("Support Vector Machine Classification")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            st.write("Model trained successfully!")
            st.write("Accuracy:", model.score(x_test, y_test))
            st.write("Precision:", precision_score(y_test, y_pred, average='binary'))
            st.write("Recall:", recall_score(y_test, y_pred, average='binary'))
            plot_metrics(metrics)


    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, step=0.01, value=1.0)
        st.write("Logistic Regression Model trained successfully!")
        metrics = st.sidebar.multiselect("What metrics to plot?", ("confusion_matrix", "roc_curve", "precision_recall_curve"), key = 'metrics')
        if st.sidebar.button("Classify", key = "classify"):
            st.subheader("Logistic Regression Classification")
            model = LogisticRegression(C=C, random_state=0)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            st.write("Model trained successfully!")
            st.write("Accuracy:", model.score(x_test, y_test))
            st.write("Precision:", precision_score(y_test, y_pred, average='binary'))
            st.write("Recall:", recall_score(y_test, y_pred, average='binary'))
            plot_metrics(metrics)



    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, step=10, value=100)
        max_depth = st.sidebar.slider("Max Depth", 1, 20, step=1, value=10)
    
        st.write("Random Forest Model trained successfully!")
        metrics = st.sidebar.multiselect("What metrics to plot?", ("confusion_matrix", "roc_curve", "precision_recall_curve"), key = 'metrics')
        if st.sidebar.button("Classify", key = "classify"):
            st.subheader("Random Forest Classification")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            st.write("Model trained successfully!")
            st.write("Accuracy:", model.score(x_test, y_test))
            st.write("Precision:", precision_score(y_test, y_pred, average='binary'))
            st.write("Recall:", recall_score(y_test, y_pred, average='binary'))
            plot_metrics(metrics)
            


    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mashroom Data Set (Classification) ")
        st.write(df)

if __name__ == '__main__':
    main()


