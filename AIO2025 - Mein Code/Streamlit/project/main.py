import streamlit as st
import pandas as pd
import math

# set title
st.write("Mortgage Repayment Calculator")
#? set number_input for Home Value, Deposit, Interest Rate, Loan Tearm
st.write("Input Data")
col1, col2 = st.columns(2)
home_value = col1.number_input("Home Value", min_value=0, max_value=500000)
deposit = col1.number_input("Deposit", min_value=0, max_value=200000)
interest_rate = col2.number_input("Interest Rate (in %)", min_value=0.1, max_value=5.5) # min and max same datatype
# how long you going to loan in year
loan_term = col2.number_input("Loan Term", min_value=1, max_value=30)

loan_amount = home_value - deposit
monthly_interest_rate = (interest_rate/100)/12
number_of_payments = loan_term * 12
monthly_payment = (
    loan_amount
    * (monthly_interest_rate * (1 + monthly_interest_rate) ** number_of_payments)
    / ((1 + monthly_interest_rate) ** number_of_payments - 1)
)

# Display the repayments.
total_payments = monthly_payment * number_of_payments
total_interest = total_payments - loan_amount

#? Display the repayment using .metric
# create 3 columns
col1, col2, col3 = st.columns(3, gap="small", vertical_alignment="top")
# for each column add .metric with label (monthly, total_payments, interest)
col1.metric(label="Monthly Interest Rate", value=f'${monthly_interest_rate:.10f}')
col2.metric(label="Total Payment", value=f'${total_payments:,.0f}')
col3.metric(label="Total Interest Rate", value=f'${total_interest:,.0f}')

#? Calculate the remaining un-pay loan each month
# Create a data-frame with the payment schedule
schedule = []
remaining_balance = loan_amount # how much money left to pay

for i in range(1, int(number_of_payments) + 1):
    interest_payment = remaining_balance * monthly_interest_rate
    principal_payment = monthly_payment - interest_payment
    remaining_balance -= principal_payment
    year = math.ceil(i / 12)  # Calculate the year into the loan
    schedule.append(
        [
            i,
            monthly_payment,
            principal_payment,
            interest_payment,
            remaining_balance,
            year,
        ]
    )

# display using pandas DataFram
# columns = data in schedule
df = pd.DataFrame(
    schedule,
    columns=['Month', 'Payment', 'Principle', 'Interest', 'Remaining Balance', 'Year'],
)

st.dataframe(df)

# # Display the data-frame as a chart
# # groupby Year, Remaining Balance take the minimum using pandas
st.write("Payment Schedule")
chart_df = df[['Year', 'Remaining Balance']].groupby("Year").min()
# display linechart_df
st.line_chart(chart_df)