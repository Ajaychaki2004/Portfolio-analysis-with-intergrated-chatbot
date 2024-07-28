import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from openai import OpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import altair as alt

@st.cache_data
def chat(ques):
    API_ENDPOINT = "https://stock.name.co/stock" #enter your host url
    headers = {
        "name": ques,
        "Authorization": "API KEY" #enter your API KEY
        }
    response = requests.request("GET", API_ENDPOINT,  headers=headers)
    return response.json()

# Page configuration
st.set_page_config(page_icon="📊", layout="wide", page_title="Investment Portfolio & Chat Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Portfolio Analysis", "Q&A Chat", "Risk Prediction", "Current Prices"])

# Function to load each page
def load_page(page):
    if page == "Home":
        home()
    elif page == "Portfolio Analysis":
        portfolio_analysis()
    elif page == "Q&A Chat":
        qa_chat()
    elif page == "Risk Prediction":
        risk_prediction()
    elif page == "Current Prices":
        current_prices()

# Home Page
def home():
    st.title("Welcome to the Investment Portfolio & Chat Dashboard")
    st.write("Use the sidebar to use the following features.")
    st.write("""
             ***Features***:

**Portfolio Analysis**: Input your assets and starting date to get a detailed analysis of your portfolio's performance. Compare it against benchmark indices like the S&P 500 and visualize your portfolio composition.

**Q&A Chat**: Leverage AI to answer your investment-related questions. Whether you need market insights, stock-specific information, or general financial advice, our AI chat is here to help.

**Risk Prediction**: Enter the name of any stock to get a comprehensive risk analysis. Understand the volatility and potential risks associated with your investments.

**Current Prices**: Check the latest prices of your favorite stocks. Enter multiple stock names to get real-time prices and visualize the data with interactive charts
             """)
# Portfolio Analysis Page
def portfolio_analysis():
    st.header("Investment Portfolio Analysis")

    # Input for assets and starting date
    assets = st.text_input("Provide your assets (comma-separated)", "AAPL,GOOGL")
    start = st.date_input("Pick a starting date for your analysis", value=pd.to_datetime('2024-06-01'))

    # Downloading stock data
    data = yf.download(assets.split(','), start=start)['Adj Close']

    # Relative change for the provided assets
    ret_df = data.pct_change()
    cumul_ret = (ret_df + 1).cumprod() - 1
    pf_cumul_ret = cumul_ret.mean(axis=1)

    # Download benchmark data (S&P 500)
    benchmark = yf.download('^GSPC', start=start)['Adj Close']
    bench_ret = benchmark.pct_change()
    bench_dev = (bench_ret + 1).cumprod() - 1

    # Portfolio standard deviation
    W = (np.ones(len(ret_df.cov())) / len(ret_df.cov()))
    pf_std = (W.dot(ret_df.cov()).dot(W)) ** (1/2)

    # Portfolio and benchmark performance
    st.subheader("Portfolio vs. Index Development")
    tog = pd.concat([bench_dev, pf_cumul_ret], axis=1)
    tog.columns = ['S&P500 Performance', 'Portfolio Performance']
    st.line_chart(data=tog)

    # Portfolio and benchmark risk
    st.subheader("Portfolio Risk:")
    st.write(pf_std)

    st.subheader("Benchmark Risk:")
    bench_risk = bench_ret.std()
    st.write(bench_risk)

    # Portfolio composition pie chart
    st.subheader("Portfolio Composition:")
    fig, ax = plt.subplots(facecolor='#121212')
    ax.pie(W, labels=data.columns, autopct='%1.1f%%', textprops={'color':'white'})
    st.pyplot(fig)

# Q&A Chat Page
def qa_chat():
    st.header("Q&A Chat")
    ques = st.text_input("Ask with AI")
# Initialize the LLM
    def chat(ques):
        API_ENDPOINT = "https://stock.name.co/stock" # Enter your host url
        headers = {
            "name": ques,
            "Authorization": "API KEY" # Enter your API KEY
            }
        response = requests.request("GET", API_ENDPOINT,  headers=headers)
        return response.json() 
    def sprice(ques):
        return  chat(ques)['currentPrice']["NSE"]
    def his(ques):
        return chat(ques)['stockTechnicalData'][0]
    model = ChatGroq(
        temperature = 0.5,
        base_url= 'http://api.name.co/', # Enter your host url
        api_key='API KEY',# Enter your API KEY
        model="gpt-4o",
        )
    template = """Answer the question based on the context below.
 
    Context: The Indian stock market includes a diverse range of stocks spanning various sectors such as Information Technology, Finance, Energy, Healthcare, Automobiles, Consumer Goods, Telecommunications, Metals and Mining, and Real Estate.
    Notable companies include Tata Consultancy Services, Infosys, HDFC Bank, Reliance Industries, Sun Pharmaceutical, Maruti Suzuki, Hindustan Unilever, Bharti Airtel, Tata Steel, and DLF Limited.
    These stocks are traded on major exchanges like the Bombay Stock Exchange (BSE) and the National Stock Exchange (NSE), reflecting the multifaceted nature of India's economy.
    The market is regulated by the Securities and Exchange Board of India (SEBI) to ensure transparency and protect investors, making it a key component of the country's financial system and economic development.
    """+"""

    {ques}
 
    Answer: """
 
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=template)

    if "price" in ques:
        st.write(sprice("ques"))
    elif "avg" in ques:
        st.write(his("ques"))
    else:
      res = model.invoke(prompt_template.format(ques=chat(ques)))
      st.write(res)


# Risk Prediction Page
def risk_prediction():
    st.header("Risk Prediction")

    ques = st.text_input("Enter Stock Name for prediction")

    if ques:
        data = chat(ques)
        st.subheader("Stock Information")
        st.write(f"Company Name: {data['companyName']}")
        st.write(f"Industry: {data['industry']}")
        st.write(f"Current Price: ${data['currentPrice']}")
        st.write(f"Risk Meter: {data['riskMeter']}")
        if 'companyProfile' in data and 'companyDescription' in data['companyProfile']:
            st.subheader("Company Description")
            st.write(data['companyProfile']['companyDescription'])
        else:
            st.info("Company description not available.")

def current_price(ques):
    API_ENDPOINT = "https://stock.coralflow.co/stock"
    headers = {
        "name": ques,
        "Authorization": "Bearer cf-ajay-5758dbd785b43a31aac4b7e23a5f990cfe028e40c28ae0e45006049b" 
        }
    response = requests.request("GET", API_ENDPOINT,  headers=headers)
    return response.json()

# Current Prices Page
def current_prices():
    st.header("Current Price of Stocks")
    name = st.text_input("Enter stock name 1")
    name1 = st.text_input("Enter stock name 2")
    name2 = st.text_input("Enter stock name 3")

    if name and name1 and name2:
        news_results1 = current_price(name).get('currentPrice', {})
        news_results2 = current_price(name1).get('currentPrice', {})
        news_results3 = current_price(name2).get('currentPrice', {})

    price1 = news_results1.get('NSE', 'N/A')
    price2 = news_results2.get('NSE', 'N/A')
    price3 = news_results3.get('NSE', 'N/A')

    st.write(f"Current price of {name}: {price1}")
    st.write(f"Current price of {name1}: {price2}")
    st.write(f"Current price of {name2}: {price3}")

    data = {
        'Stock': [name, name1, name2],
        'Price': [price1, price2, price3]
    }

    df = pd.DataFrame(data)

    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna()

    # Plot histogram
    histogram = alt.Chart(df).mark_bar().encode(
            x=alt.X('Price:Q', bin=alt.Bin(maxbins=10), title='Price'),
            y='count():Q',
            color='Stock:N'
        ).properties(
            width=600,
            height=400
        ).configure_axis(
            grid=True
        ).configure_view(
            strokeWidth=0
        )
    
    st.altair_chart(histogram, use_container_width=True)
    
# Run the selected page
load_page(page)