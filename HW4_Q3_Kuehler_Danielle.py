# Danielle Kuehler
# ITP 449 Summer 2020
# HW4
# Question 3

import pandas as pd
import matplotlib.pyplot as plt

#Read in CSV Files
covidData = pd.read_csv("06-18-2020.csv")
covidConfirmed = pd.read_csv("time_series_covid19_confirmed_US.csv")
covidDeaths = pd.read_csv("time_series_covid19_deaths_US.csv")
pd.set_option("display.max_columns", None)

'''
1.	What state in the US currently has the highest number of active cases?
'''
#Drop NaN Values
covidData.dropna(axis=0, inplace=True)
#Match active column with maximum value of active, check if value has max value of active
print("Question 1:\nHighest number of active cases: ", covidData.loc[covidData.Active == covidData.Active.max(), 'Province_State']) #loc fetches province state

'''
2.	What state in the US has the highest fatality rate (deaths as a ratio of infection)?
'''
#Highest mortality rate, display:
print("\nQuestion 2:\nHighest fatility rate: ", covidData.loc[covidData.Mortality_Rate == covidData.Mortality_Rate.max(), 'Province_State'])

'''
3.	What is the difference in the testing rate between the state that tests the most and the state that tests the least?
'''
#State that tests most
print("\nQuestion 3:\nState that tests the most: ", covidData.loc[covidData.Testing_Rate == covidData.Testing_Rate.max(), 'Province_State'])
#State that tests the least
print("State that tests the least: ", covidData.loc[covidData.Testing_Rate == covidData.Testing_Rate.min(), 'Province_State'])
#Display difference in testing
print("Difference in testing rate:", round(covidData.Testing_Rate.max() - covidData.Testing_Rate.min()), "cases")

'''
4.	Plot the number of daily new cases in the US for the top 5 states with the highest confirmed cases (as of today). From March 1 – today. Use Subplot 1.
'''
#Group data by state
sumCases = covidConfirmed.groupby(["Province_State"]).sum().loc[:,"3/1/20":"6/18/20"].reset_index()

#Sort to most cases
high5 = sumCases.sort_values(by=["6/18/20"], ascending=False).head(5)

#Get index of states with highest cases
high5Indexes = high5.set_index("Province_State")

#Count of cases from march 1 through now
x = high5.transpose() #Swap rows and columns
x = x.iloc[1:,:] #All row's except 0th row
x.columns = ['New York','New Jersey','California','Illinois','Massachusetts']

#Plotting
fig = plt.figure()
fig.suptitle('Kuehler_Danielle_HW4\nCOVID19 Data')
ax1 = fig.add_subplot(1,2,1, title="COVID19 Cases") #number of rows, number of columns, first subplot
ax1.plot(x)

#Formatting
ax1.set_xticks(['3/1/20','3/18/20','4/1/20', '4/18/20', '5/1/20', '5/18/20', '6/1/20', '6/18/20'])
ax1.set_xlabel("Date")
ax1.set_ylabel("Number of Cases (Daily)")
ax1.legend(x, loc="upper left")

'''
5.	Plot the number of daily deaths in the US for the top 5 states with the highest confirmed cases (as of today). From March 1 – today. Use Subplot 2.
'''
#Group data by state
sumDeaths = covidDeaths.groupby(["Province_State"]).sum().loc[:,"3/1/20":"6/18/20"].reset_index()

#Sort to most cases
top5 = sumDeaths.sort_values(by=["6/18/20"], ascending=False).head(5)

#Get index of states with highest cases
top5Indexes = top5.set_index("Province_State")

#Count of deaths from march 1 through now
a = top5.transpose() #Swap rows and columns
a = a.iloc[1:,:] #All row's except 0th row
a.columns = ['New York','New Jersey','Massachusetts','Illinois','Pennsylvania']

#Plotting
ax2 = fig.add_subplot(1,2,2, title="COVID19 Deaths") #number of rows, number of columns, first subplot
ax2.plot(a)

#Formatting
ax2.set_xticks(['3/1/20','3/18/20','4/1/20', '4/18/20', '5/1/20', '5/18/20', '6/1/20', '6/18/20'])
ax2.set_xlabel("Date")
ax2.set_ylabel("Number of Deaths (Daily)")
ax2.legend(a, loc="upper left")

#Display
plt.show()





