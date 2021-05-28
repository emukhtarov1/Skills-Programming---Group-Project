#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This program lets the user choose a number of listed stocks taken directly from Yahoo Finance. It utilises the data from Yahoo Finance to compare the portofolio of stocks chosen based on a number of financial and operational KPIs


# In[2]:


pip install yfinance


# In[3]:


pip install pandas_datareader


# In[4]:


pip install plotly


# In[1]:


import pandas as pd
from pandas_datareader import data as pdr
import seaborn as sns
from copy import copy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import datetime as dt
import yfinance as yf
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# define all the functions necessary to perform the CAPM analysis 

# normalize function to compare S&P 500 and selected stock(s) from base = 1
def normalize(df):
    x = df.copy()
    for i in x:
        x[i] = x[i] / x[i][0]
    return x


# line chart showing adjusted close
# we consider the adjusted closing price rather the closing price to account 
# for stock splits, dividend and stock issuances
# the adjusted closing price amends a stock's closing price to reflect 
# that stock's value after accounting for any corporate actions.

def interactive_plot(df):
    line_chart = px.line(df.copy(), title = 'Adjusted Close Price' 
    + ' of Selected Stocks and S&P500')

    line_chart.update_xaxes(title_text = 'Date', dtick="M1")
    line_chart.update_yaxes(title_text = 'Adjusted Close Price')
    line_chart.update_layout(showlegend = True)

    line_chart.show()
    
# develop a function to calculate the shares' daily and index's returns 
#which are later used in the CAPM mode

def daily_return(df):
    df_daily_return = df.copy()
    
    for i in df:
        for j in range(1, len(df)):
            df_daily_return[i][j] = ((df[i][j] - df[i][j-1]) / df[i][j-1]) * 100
    
        df_daily_return[i][0] = 0
    return df_daily_return


# function plot a scatter plot between the selected stock 
# and the S&P500 (Market)

def capm_scatter(df):
    for i in df:
        if i != 'Date' and i != '^GSPC':

            fig = px.scatter(portfolio_dailyreturn, x = '^GSPC',
                             y = i, title = i)

            b, a = np.polyfit(portfolio_dailyreturn['^GSPC'], 
                              portfolio_dailyreturn[i], 1)

            fig.add_scatter(x = portfolio_dailyreturn['^GSPC'], 
                            y = b*portfolio_dailyreturn['^GSPC'] + a)
                
            fig.show()

        
# function to calculate beta and alpha for all stocks in chosen portfolio

def beta_in_portfolio(df):
    for i in portfolio_dailyreturn:
    
        if i != 'Date' and i != '^GSPC':

            b, a = np.polyfit(portfolio_dailyreturn['^GSPC'], 
                              portfolio_dailyreturn[i], 1)

            beta[i] = b

            alpha[i] = a


# In[3]:


#enter timefrime which has to retrieved out from yahoo finance 
restart = ('yes')

while restart == ('yes'):

    start = input('Do you want to start the program ? (yes/no) ')
    
    if 'yes' in start:
        print("Now please define the timeframe which you want to analyse.") 
        start_date = input("What’s the start date? (YYYY-MM-DD) ")
        end_date = input("What’s the end date? (YYYY-MM-DD) ")
        break


# In[4]:


#the tickers entered are directly retrieved from yahoo finance 
#enter the tickers all caps without commas between them, and a space e.g: "JNJ MRK"

portfolio = [str(i) for i in input('Please enter the tickers of the stocks you' 
+' want to analyze: ').split() ]


# In[5]:


infos = []

for i in portfolio:
    infos.append(yf.Ticker(i).info)

df = pd.DataFrame(infos)
df = df.set_index('symbol')


# In[6]:


fundamentals = ['trailingPE', 'pegRatio', 'priceToSalesTrailing12Months', 
                'priceToBook', 'enterpriseToEbitda']


# In[7]:


df[df.columns[df.columns.isin(fundamentals)]]


# In[8]:


df['enterpriseToEbitda'].nlargest()


# In[9]:


df['priceToBook'].nlargest()


# In[10]:


plt.figure(figsize = (10, 5))
plt.bar(df.index, df.enterpriseToEbitda)
plt.title('Comparison of EV/EBITDA ratios')
plt.show()


# In[11]:


plt.figure(figsize = (10,5))
plt.bar(df.index, df.priceToBook)
plt.title('Comparison of Price to Book Value ratios')
plt.show()


# In[12]:


# use market indexes to compare the individual stock to: S&P 500

portfolio.append('^GSPC')


# In[15]:


# next step is to import the data the user entered from yahoo finance, 
# taking into account dividends and stocks splits --> adj close 

user_stocks = pd.DataFrame(pdr.DataReader(portfolio, 'yahoo', start_date, 
                                          end_date)['Adj Close'])


# In[16]:


# interactive plot using normalized data from base year

interactive_plot(normalize(user_stocks))


# In[18]:


portfolio_dailyreturn = daily_return(user_stocks)


# In[19]:



portfolio_dailyreturn['^GSPC']

#rm represent the market return: it averages out the daily return per the number of tradings days per year (252)
rm = portfolio_dailyreturn.mean()['^GSPC'] * 252


# In[20]:


# capm scatter using px 

capm_scatter(portfolio_dailyreturn)


# In[21]:


print("""This part of the program lets you perform a Capital Asset Pricing Model
analysis of your selected stocks from Yahoo Finance, and evaluates the relative
 performance and volatility of your selected stocks comapred to the 
performance of the S&P500 market index. You can select which stocks you would 
like to include as part of the CAPM analysis.""")


# In[22]:


# getting beta and alpha for portfolio of selected stocks 
# empty dictionaries to hold values, need to be cleared after first 
# run through entire program 

beta = {}
alpha = {}

for i in portfolio_dailyreturn:
    
    if i != 'Date' and i != '^GSPC':
        
        b, a = np.polyfit(portfolio_dailyreturn['^GSPC'],
                          portfolio_dailyreturn[i], 1)

        beta[i] = b

        alpha[i] = a


# In[23]:


# dictionary for user to see entire portfolio beta 
beta


# In[24]:


# the risk free rate is taken from statista 
# The 10 year U.S treasury bond yield is considered as the risk free rate 
# for the CAPM
rf = {
    "2021": 1.61,
    "2020": 1.01,
    "2019": 1.92,
    "2018": 2.69,
    "2017": 2.41,
    "2016": 2.45,
    "2015": 2.25,
    "2014": 2.17,
    "2013": 3.03,
    "2012": 1.76,
    "2011": 1.88,
    "2010": 3.29,
    "2009": 3.84,
    "2008": 2.22,
    "2007": 4.04,
    "2006": 4.70,
    "2005": 4.40,
    "2004": 4.22,
    "2003": 4.25,
    "2002": 3.82,
    "2001": 5.03,
    "2000": 5.11,
}


# In[ ]:


# asks user for ticker of selected portfolio and calculates CAPM one ticker
# at a time 

rf_inputted = rf.get(input("Enter the ending year: "))

for i in portfolio:

  ER_user = print("the company's cost of equity is: ", rf_inputted +
                (beta.get(input('Enter the ticker of the stock : ')) *
                 (rm - rf_inputted)))
ER_user


# In[ ]:


restart = input('Do you want to restart the program ? (yes/no) ')

if restart == ('yes'):
    print('Let ́s start over !')
elif restart == ('no'):
    print('Thank you for your time ! Bye !')
    
    


# In[ ]:




