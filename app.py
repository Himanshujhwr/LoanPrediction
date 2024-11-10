import streamlit as st
import pandas as pd
import pickle as pk

# Load model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# Set page config
st.set_page_config(page_title="Loan Prediction App", layout="centered")

# Main title
st.title('  🏦 Loan Approval App')
st.header('Predict Your Loan Approval Status')

# Sidebar for inputs
st.sidebar.header('Input Parameters')
no_of_dep = st.sidebar.slider('Choose Number of Dependents', 0, 5)
grad = st.sidebar.selectbox('Education Level', ['Graduated', 'Not Graduated'])
self_emp = st.sidebar.selectbox('Self Employed?', ['Yes', 'No'])
Annual_Income = st.sidebar.slider('Annual Income', 0, 10000000, step=10000)
Loan_Amount = st.sidebar.slider('Loan Amount', 0, 5000000, step=10000)
Loan_Dur = st.sidebar.slider('Loan Duration (Years)', 0, 20)
Cibil = st.sidebar.slider('Cibil Score', 0, 1000)
Assets = st.sidebar.slider('Assets Value', 0, 10000000, step=10000)

# Convert inputs for model
grad_s = 0 if grad == 'Graduated' else 1
emp_s = 0 if self_emp == 'No' else 1

# Centering the selected values display with better layout
st.markdown("<h2 style='text-align: left;'>Selected Values:</h2>", unsafe_allow_html=True)

# Create two columns for better alignment
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Number of Dependents:**")
    st.write(no_of_dep)
    st.markdown("**Education Level:**")
    st.write(grad)
    st.markdown("**Self Employed:**")
    st.write(self_emp)
    st.markdown("**Loan Duration:**")
    st.write(f"{Loan_Dur} years")

with col2:
    st.markdown("**Cibil Score:**")
    st.write(Cibil)
    st.markdown("**Annual Income:**")
    st.write(f"₹{Annual_Income:,.2f}")
    st.markdown("**Loan Amount:**")
    st.write(f"₹{Loan_Amount:,.2f}")
    st.markdown("**Assets Value:**")
    st.write(f"₹{Assets:,.2f}")

# Add some vertical space
st.write("<br>", unsafe_allow_html=True)

# Centering the Predict button using markdown
st.markdown("<div style='text-align: center;'><br>", unsafe_allow_html=True)
if st.button("Predict", key="predict_button"):
    with st.spinner('Making Prediction...'):
        pred_data = pd.DataFrame([[no_of_dep, grad_s, emp_s, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]],
                                  columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets'])
        pred_data = scaler.transform(pred_data)
        predict = model.predict(pred_data)

        # Display result
        st.markdown("<h2 style='text-align: center;'>Prediction Result:</h2>", unsafe_allow_html=True)
        if predict[0] == 1:
            st.success(' Congrats,Your Loan Application has been Approved', icon="✅")
        else:
            st.error(' Sorry, Your Loan Application is Rejected', icon="❌")
st.markdown("</div>", unsafe_allow_html=True)



# Add footer or additional information if needed
st.markdown("---")
st.markdown("<div style='text-align: center;'>This app is designed to help you predict loan approval based on various parameters.</div>", unsafe_allow_html=True)
