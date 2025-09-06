import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
from tensorflow.keras.models import load_model


st.title("Churn Model For XYZ Bank")

credit = st.slider("CreditScore :", min_value=0, max_value=1000)
st.write(f"Creditscore is: {credit}")

geog = st.selectbox("Geography: ", ["France","Spain","Germany"])
gender = st.radio("Gender: ",["Male","Female"])
age = st.slider("Age:",min_value=18,max_value=75)
tenure = st.slider("Tenure:",min_value=1,max_value=10)


bal = st.text_input("Enter your balance:",50000.00,placeholder="balance")
if bal:
    try:
        bal = float(bal)
        if (bal >= 0):
            st.success(f"Valid positive number: {bal}")
        else:
            st.error("Number must be non-negative")
    except ValueError:
        st.error("Enter a number")

no_of_prod = st.slider("Number of products:",min_value=1,max_value=5)


has_cred_card = st.radio("Has Credit Card: ",["Yes","No"])
if has_cred_card is "Yes":
    has_cred_card = 1
else:
    has_cred_card = 0


is_active_mem = st.radio("Is active member: ",["Yes","No"])
if is_active_mem is "Yes":
    is_active_mem = 1
else:
    is_active_mem = 0


estimated_sal = st.text_input("Estimated Salary:",100000.00,placeholder="salary")
if estimated_sal:
    try:
        estimated_sal = float(estimated_sal)
        if (estimated_sal >= 0):
            st.success(f"Valid positive number: {estimated_sal}")
        else:
            st.error("Number must be non-negative")
    except ValueError:
        st.error("Enter a number")


input_data = {
   'CreditScore' : [credit],
   'Geography' : [geog],
   'Gender' : [gender],
   'Age' : [age],
   'Tenure' : [tenure],
   'Balance' : [bal],
   'NumOfProducts' : [no_of_prod],
   'HasCrCard' : [has_cred_card],
   'IsActiveMember' : [is_active_mem],
   'EstimatedSalary' : [estimated_sal]
}

input_data = pd.DataFrame(input_data)

with open("oneHotenc.pkl","rb") as file:
    oneHotenc = pickle.load(file)

with open("labelenc.pkl","rb") as file:
    labelenc = pickle.load(file)

with open("scale.pkl","rb") as file:
    scale = pickle.load(file)

#OneHotEncoding
encoded_geo = oneHotenc.transform(input_data[["Geography"]]).toarray()
encoded_geo_df = pd.DataFrame(encoded_geo, columns=oneHotenc.get_feature_names_out())

input_data = pd.concat([input_data.drop(["Geography"], axis=1),encoded_geo_df],axis=1)

#LabelEncoding
input_data["Gender"] = labelenc.transform(input_data["Gender"])

#StandardScalar
scaled_data = scale.transform(input_data)

model = load_model("model.h5")
prediction = model.predict(scaled_data)
st.write(f"Prediction Value: {prediction[0][0]:.3f}")
if prediction[0][0] > 0.5:
    st.write("Customer is likely to churn!!")
else:
    st.write("Customer is not likely to churn!!")