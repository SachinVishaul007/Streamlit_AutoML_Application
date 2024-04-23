import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

# Configuring main settings for the Streamlit page
st.set_page_config(page_title='AutoML App', layout='wide')
st.title("AutoML App")
st.subheader('Comparing Performance of Models for a Given Dataset')

# Function to build and evaluate models
def build_model(df):
    df = df.loc[:100]  # Limit data for testing; remove for full dataset usage
    X = df.iloc[:, :-1]  # Features: all columns except the last one
    Y = df.iloc[:, -1]   # Target: the last column

    # Displaying data dimensions
    st.markdown('**1.2. Dataset dimension**')
    st.write('X')
    st.info(X.shape)
    st.write('Y')
    st.info(Y.shape)

    # Displaying variables involved
    st.markdown('**1.3. Variable details**:')
    st.write('X variable (first 20 are shown)')
    st.info(list(X.columns[:20]))
    st.write('Y variable')
    st.info(Y.name)

    # Splitting data and applying LazyRegressor
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, random_state=seed_number)
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models_train, predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test, predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)

    # Display model performance
    st.subheader('2. Table of Model Performance')
    st.write('Training set')
    st.write(predictions_train)
    st.write('Test set')
    st.write(predictions_test)

    # Visualizing R-squared values
    st.subheader('3. Plot of Model Performance (Test set)')
    with st.markdown('**R-squared**'):
        predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"]]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test)
        ax1.set(xlim=(0, 1))
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)

    with st.markdown('**RMSE (capped at 50)**'):
        
        predictions_test["RMSE"] = [50 if i > 50 else i for i in predictions_test["RMSE"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
    
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)


    with st.markdown('**Calculation time**'):
        # Tall
        predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)

        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)


# Rendering Sidebar for user input
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)

# Area for dataset display and model building
st.subheader('1. Dataset')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Use Sample Dataset(Boston Housing)'):
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names).loc[:100]
        Y = pd.Series(boston.target, name='response').loc[:100]
        df = pd.concat([X, Y], axis=1)
        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df.head(5))
        build_model(df)
