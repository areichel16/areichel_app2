# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set up the app title and image
header_html = """
    <div style="text-align: center; font-weight: bold; font-size: 48px;">
        <span style="background: linear-gradient(to right, red, yellow, green); 
                    -webkit-background-clip: text; color: transparent;">
            Traffic Volume Predictor
        </span>
    </div>
"""
st.markdown(header_html, unsafe_allow_html=True)
subheader_html = """
    <div style="text-align: center; font-size: 19px;">
        Utilize our advanced Machine Learning application to predict traffic volume.
    </div>
"""
st.markdown(subheader_html, unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,12,1])
with col2:
    st.image('traffic_image.gif', width = 600)
    

# Reading the pickle file that we created before 
model_pickle = open('traffic.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()

       
df = pd.read_csv('Traffic_Volume.csv')
df = df.fillna('None')

df['date_time'] = pd.to_datetime(df['date_time'])
df['month'] = df['date_time'].dt.month_name()
df['weekday'] = df['date_time'].dt.day_name()
df['hour'] = df['date_time'].dt.hour
df.drop(columns=['date_time'], inplace=True)

with st.sidebar:
    st.image('traffic_sidebar.jpg', use_column_width=True,
         caption = "Traffic Volume Predictor")
    st.write('### Input Features')
    st.write('Either upload your data file or manually enter input features.')
    with st.expander('Option 1: Upload CSV File', expanded = False):
        st.write('Upload a CSV file containing traffic details.')
        uploaded_file = st.file_uploader('''Choose a CSV file''', type=['''csv'''])
        st.write('''### Sample Data Format for Upload''')
        st.dataframe(df.head(5).drop(columns=['traffic_volume']))
        st.warning('''⚠️ Ensure your uploaded file has the same column names and data types as shown above.''')

with st.sidebar:
    with st.expander('Option 2: Fill Out Form', expanded = False):
        with st.form("Enter the traffic details manually using the form below."):
            holiday = st.selectbox('Choose whether today is a disignated holiday or not', options=df['holiday'].unique())
            temp = st.number_input("Average temperature in Kelvin", min_value=df['temp'].min(), max_value=df['temp'].max(), step=0.01)
            rain_1h = st.number_input("Amount in mm of rain that occurred in the hour", min_value=df['rain_1h'].min(), max_value=df['rain_1h'].max(), step=0.01)
            snow_1h = st.number_input("Amount in mm of snow that occurred in the hour", min_value=df['snow_1h'].min(), max_value=df['snow_1h'].max(), step=0.01)
            clouds_all = st.number_input("Percent of cloud cover", min_value=df['clouds_all'].min(), max_value=df['clouds_all'].max(), step=1)
            weather_main = st.selectbox("Choose the current weather", options=df['weather_main'].unique())
            month = st.selectbox("Choose month", options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
            weekday = st.selectbox("Choose day of the week", options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            hour = st.selectbox("Choose hour", options=list(range(24)))
            submit_button = st.form_submit_button('Submit Form Data')

if uploaded_file is not None:
    st.success('✅ CSV File Uploaded Successfully')
elif submit_button == True:
    st.success('✅ Form Data Submitted Successfully')
else:
    st.info('Please choose a data input method to proceed')

# Sidebar input fields for numerical variables using sliders
alpha = st.slider('Select alpha value for prediction intervals', min_value =0.01, max_value=0.50, value = 0.10, step=0.01)

# Encode the inputs for model prediction
encode_df = df.copy()
encode_df = encode_df.drop(columns=['traffic_volume'])

# Combine the list of user data as a row to default_df
encode_df.loc[len(encode_df)] = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, month, weekday, hour]
cat_var1 = ['holiday', 'weather_main', 'month', 'weekday', 'hour']

# Create dummies for encode_df
encode_dummy_df = pd.get_dummies(encode_df, columns=cat_var1)

# Extract encoded user data
user_encoded_df = encode_dummy_df.tail(1)

# Get the prediction with its intervals
alpha_value = alpha 
prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha_value)
pred_value = prediction[0]
lower_limit = intervals[:, 0]
upper_limit = intervals[:, 1]

# Ensure limits are within [0, 1]  
lower_limit = max(0, lower_limit[0][0])
upper_limit = upper_limit[0][0]


# Predictions
if uploaded_file is not None:
    input = pd.read_csv(uploaded_file)
    merge = pd.concat([input, encode_df], join='outer')
    df = pd.get_dummies(merge, columns = ['holiday', 'weather_main', 'month', 'weekday', 'hour'])
    user_df = df.head()
    user_df = user_df.reset_index(drop=True)

    usercopy = user_df.copy()
    prediction2, intervals2 = reg_model.predict(usercopy, alpha = alpha_value)
    lower_limits = [item[0][0] for item in intervals2]
    upper_limits = [item[1][0] for item in intervals2]
   
    non_negative = [max(value, 0) for value in lower_limits]

    input['Predicted Traffic'] = [f"{x:.1f}" for x in prediction2]
    input['Lower Limit'] = [f"{x:.1f}" for x in non_negative]
    input['Upper Limit'] = [f"{x:.1f}" for x in upper_limits]
    st.subheader(f"**Prediction Results with** {(1-alpha)*100:.0f}% Prediction Interval")
    st.write(input)
else:
    # Show the prediction on the app
    st.write("## Predicting Traffic Volume...")
    # Display results using metric card
    st.metric(label = "Predicted Traffic Volume", value = f"{pred_value:.0f}")
    st.write(f"**Prediction Interval** ({(1-alpha) * 100:.0f}%): [{lower_limit:.0f}, {upper_limit:.0f}]")


# Additional tabs for DT model performance
st.subheader("Model Performance and Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")