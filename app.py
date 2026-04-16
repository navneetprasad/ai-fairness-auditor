import streamlit as st
import pandas as pd
from google import genai

# --- Web App UI Setup ---
st.set_page_config(page_title="AI Fairness Auditor", page_icon="⚖️")
st.title("⚖️ AI Fairness Auditor")
st.markdown("Evaluate credit scoring datasets for historical bias and generate automated compliance reports.")

# --- Core Math Engine & Data ---
# Simulated dataset with names for export functionality
data = {
    'First_Name': ['Aarav', 'Zoya', 'Rahul', 'Priya', 'Kabir', 'Ananya', 'Vikram', 'Neha', 'Rohan', 'Aditi'] * 100,
    'Surname': ['Sharma', 'Khan', 'Patel', 'Singh', 'Gupta', 'Verma', 'Kumar', 'Reddy', 'Das', 'Jain'] * 100,
    'Age_Group': ['Under 25'] * 300 + ['Over 25'] * 700,
    'AI_Loan_Decision': [1] * 90 + [0] * 210 + [1] * 560 + [0] * 140 
}
df = pd.DataFrame(data)

# Ensure the roster is cleanly sorted alphabetically by first name for administrative export
df = df.sort_values(by='First_Name').reset_index(drop=True)

st.subheader("1. Ingested Dataset Roster")
st.dataframe(df.head(10)) # Show preview

# Export Button
st.download_button(
    label="Download Roster (CSV)",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='audited_roster.csv',
    mime='text/csv'
)

st.markdown("---")
st.subheader("2. Run Statistical Audit")

# --- Run the Audit ---
if st.button("Calculate Bias & Generate Report"):
    
    # Calculate Math
    approved_over_25 = df[(df['Age_Group'] == 'Over 25') & (df['AI_Loan_Decision'] == 1)].shape[0]
    total_over_25 = df[df['Age_Group'] == 'Over 25'].shape[0]
    rate_over_25 = approved_over_25 / total_over_25

    approved_under_25 = df[(df['Age_Group'] == 'Under 25') & (df['AI_Loan_Decision'] == 1)].shape[0]
    total_under_25 = df[df['Age_Group'] == 'Under 25'].shape[0]
    rate_under_25 = approved_under_25 / total_under_25

    disparate_impact = rate_under_25 / rate_over_25
        # Display Math visually
        col1, col2, col3 = st.columns(3)
        col1.metric("Over 25 Approval", f"{rate_over_25 * 100:.1f}%")
        col2.metric("Under 25 Approval", f"{rate_under_25 * 100:.1f}%")
        
        if disparate_impact < 0.80:
            col3.error(f"Disparate Impact: {disparate_impact:.2f} (FAIL)")
        else:
            col3.success(f"Disparate Impact: {disparate_impact:.2f} (PASS)")

        st.markdown("---")
        st.subheader("3. AI Compliance Report")
        
        # Connect to Gemini
        with st.spinner("Generating regulatory analysis..."):
            try:
               client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                prompt = f"""
                You are an expert financial compliance officer. I am providing you with the results of an algorithmic fairness audit on an AI credit scoring model.

                Audit Results:
                - Privilege Group (Over 25) Approval Rate: {rate_over_25 * 100}%
                - Unprivileged Group (Under 25) Approval Rate: {rate_under_25 * 100}%
                - Disparate Impact Ratio: {disparate_impact:.2f}

                The model failed the 80% rule (Disparate Impact < 0.80). 
                Write a brief, professional, 2-paragraph compliance report explaining this bias and suggesting one technical way a data scientist could fix the training data to be more fair.
                """
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                )
                st.info(response.text)
            except Exception as e:
                st.error(f"Failed to connect. Error: {e}")
