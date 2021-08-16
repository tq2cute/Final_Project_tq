# Housing Price analysis in CA (Final_Group_Project) 

# Predicting CA Housing Prices  
In this project, we will Predict the average housing prices per SQF for each county in CA. Then we will Visualize the housing prices per county on a Map. 
We will build Machine Learning Models to help investors and homeowners assess the housing prices in California based on Housing transactions. We will build a  supervised neural network machine learning model using the following independent parameters, Zip Code, House Age, SQFT, and Days on Market.

## Data Source

1) MLS Data, https://pro.mlslistings.com/, is deposited at "amazonaws.com" with the name 'big_main.csv". The data set includes 4344 housing sale transactions in California for the period from 6/2020 till 6/2021. The data set includes Street_Address, City, Zip_Code, SqFtTotal, Lot_Size, Age, BathsTotal, BedsTotal, BathsFull, BathsHalf, DOM, Year_Sold, Year_List, List_Price, Sale_Price, Listing_Date, Sale_Date, Year_Built. 

2) Additional Data set includes "county_zipcode.csv" which was deposited at "amazonaws.com". This files lists all the Zip Codes to County Name. It is from <a href="https://data.chhs.ca.gov/dataset/ead44d40-fd63-4f9f-950a-3b0111074de8/resource/ec32eece-7474-4488-87f0-6e91cb577458/download/covid19vaccinesbyzipcode_test.csv" target="_blank">California Health and Human Services Open Data Portal</a>

## Questions we Hope to answer:

Motivation, Housing prices are a hot topic, especially During the COVID-19 Pandemic (Leading to increased demand for housing). We strive to build a Machine learning model to Guide investors, Potential Buyers, or Real estate professionals on housing prices per county.<br>
1. We will build a supervised Neural Machine learning model using  House Age, SQF, Lot size & Days on the Market to predict the house Sale Price per SQF for each test data point.<br>
2. We will build a regression model to do the same<br>
3. We will compare the result of the two and discuss any pros and cons.<br>

## Communications Protocols:
* Members: Trong-Quyen Nguyen, Srividhya Thirumalairajan, Dawit Alaro, Angelica Villanueva & Mikhail Zaatra

##    Segment 1:

* Roles and Responsibilities:<br>
    o	Presentation : Mikhail Zaatra<br>
    o	GitHub: Srividhya Thirumalairajan<br>
    o	Machine Learning Model : Trong-Quyen Nguyen, Angelica Villanueva,  Srividhya Thirumalairajan & Mikhail Zaatra<br>
    o	DataBase: Dawit Alaro  & Angelica Villanueva<br>

* Project Tools: Collaboration:<br>
* Communication Methods:<br>
    *GitHub will be the Main tool for Documents and Code Development.<br>
        - Sharing resources via slack<br>
        - using zoom meeting every Thursday as well from 7-9 to work on the project
    
##    Segment 2:

*Final_data_processing.ipynb:
*   Use google Colab and pyspark, and process this online
*   Combine two data sources mentioned above "big_main.csv" and "county_zipcode.csv"
*   Use "StringIndex" from 'pyspark.ml.feature import' to assign a county name a number
*   Joined two data sets together.  Cleaned up and deleted null values.<br> ![Data Processing Yeah Data is cleaned](Final_Project/images/data_processing_1.png)
*   Export the data to postgresql.  There are three sets of data exported: 'house_data.csv', 'sale_data.csv' and 'final_data.csv'.<br> ![Data Processing data export](Final_Project/images/data_processing_2.png)


*Regression_Basic.ipynb
* 	Read the 'final_data.csv' from AWS
*   The data has 4225 records. That amount is split 95% for training and 5% for testing.
*   That split means 4013 records are for training and 212 records for testing 
*   The result is impressive, with R squared is 95%.  The model can explain 95% of the price variation.<br> ![Regression Basic MSE and R squared](Final_Project/images/Regression_Basic_1.png)
*   The coefficients for the columns "County_Index|SqFtTotal|Lot_Size|Age|BathsTotal|BedsTotal|BathsFull|BathsHalf|DOM|Year_Sold|List_Price" is below:<br> ![Regression Basic Model Coefficients](Final_Project/images/Regression_Basic_2.png)
*   With this, one can build an estimate calculator.
*   With the coefficients above, we applied the coefficient to all the sales, and created a new column "Predicted Value".
*   "final_data.csv" with an extra column of "Predicted Value" is saved as "final_prediction_all.csv". This "final_prediction_all" will be used for mapping and visualization.

##    Segment 3:

*Final_Regression_NN.ipynb
*   This file is considered "final" because this file is a merge of Regression and Neural Network learnings together
*   The purpose is to force two types of learning to accept the same number of X_train, y_train, X_test, and y_test data points
*   This final file is even further broken down into 4 models: Regression with List_Price, Regression without List_Price, Neural Network with List_Price, and Neural_Network without List_Price.
<br>![Final Descriptive](Final_Project/images/1_Final_descriptive.png)

#### 1.  Model 1: Regression with List_Price
*   We have a list of coefficients. The List_Price is highly correlated with the Sale_Price, the model dependent variable.
<br>![model 1 coefficients](Final_Project/images/2_model1_coefficient.png)
*   We have some metrics for model 1.  The R2_squared is 99%
<br>![model 1 metrics](Final_Project/images/3_model1_metrics.png)
*   Here is an example how model 1 predicts:
<br>![model 1 Sale Price examples](Final_Project/images/4_model1_saleprice_examples.png)
*   Here is the residuals plot of the model 1. The residual plot centers around the mean of zero.
<br>![model 1 residuals plot](Final_Project/images/5_model1_residuals_plot.png)
*   The prediction price plot can tell how closely model 1 predicts its Sale_Price. Without seeing the blue line (prediction value), the two Sale_Price and Prediction 1 values are overlapping most of the time.
<br>![model 1 Sale Price prediction plot](Final_Project/images/6_model1_prediction_plot.png)

#### 2.  Model 2: Regression without List_Price
*   We have a list of coefficients for model 2.  There is no coefficient for List_Price in this model
<br>![model 2 coefficients](Final_Project/images/7_model2_coefficient.png)
*   Notice how the r2_square drops significantly to 65.9%. Model 2 can only explain 65.9% of the sale_price accurately
<br>![model 2 metrics](Final_Project/images/8_model2_metrics.png)
*   Look at how apart model 2 predicts Sale_Price
<br>![model 2 Sale Price examples](Final_Project/images/9_model2_saleprice_examples.png)
*   Take a peak at the model 2 residuals plot. Its mean is centered around zero, but there are examples at the positive tail.
<br>![model 2 residual plot](Final_Project/images/10_model2_residuals_plot.png)
*   As expected, model 2 prediction is not as accurate as model 1, as seen by the blue line.
<br>![model 2 sale price prediction plot](Final_Project/images/11_model2_prediction_plot.png)

#### 3.  Model 3: Neural Network with List_Price
*   Model 3 neural network deploys three hidden layers of "relu" with 15, 20, 10 neurons respectively.  The final layer activation is "linear". This model uses optimizer='Adam', loss='mean_absolute_error', metrics=['MSE'].
<br>![NN Model 3 summary](Final_Project/images/12_model3_summary.png)
*   Notice how the value of R2_squared is pretty good at 98.6%
<br>![model 3 metrics](Final_Project/images/12_model3_metrics.png)
*  Here is the sale price examples of model 3
<br>![model 3 sale price examples](Final_Project/images/13_model3_saleprice_examples.png)
*   As expected, model 3 is pretty good at predicting the Sale_Price as the two lines are overlapping
<br>![model 3 sale price prediction plot](Final_Project/images/15_model3_prediction_plot.png)

#### 4.  Model 4: Neural Network without List_Price
*   Model 4 is exactly the same as model 3; it deploys three hidden layers of "relu" with 15, 20, 10 neurons respectively.  The final layer activation is "linear". This model uses optimizer='Adam', loss='mean_absolute_error', metrics=['MSE'].
<br>![NN Model 4 summary](Final_Project/images/16_model4_summary.png)
*   One big difference is that it has no input of "List_Price"
*   Notice how the value of R2_squared is bad at 34.04%
<br>![model 3 metrics](Final_Project/images/16_model4_metrics.png)
*  Column "Prediction 4" lists the prediction of model 4.  Amazingly, it is far from the true Sale_Price.
<br>![model 4 sale price examples](Final_Project/images/17_model4_saleprice_examples.png)
*   As expected, model 4 is amazingly bad.
<br>![model 3 sale price prediction plot](Final_Project/images/18_model4_prediction_plot.png)

#### 5.  Conclusion
<br>![conclusion rq2_squared](Final_Project/images/20_conclusion_r2_squared.png)

*   With "List_Price", both Regression and Neural Network have high r2_squared scores.  Without the "List_Price", Regression (prediction 2) does a better job at prediction the "Sale_Price" than the Neural Network (prediction 4).
<br>![Conclusion Prediction all](Final_Project/images/20_conclusion_prediction_all.png)
*   So, what is the role of "List Price"? 
*   First, take a look at the original data
<br>![Final Descriptive](Final_Project/images/1_Final_descriptive.png)
*   The data has lot of **QUANTITATIVE** parameters: SqFtTotal, Lot_Size, Age, BathsTotal, BedsTotal, BathsFull, BathsHalf, DOM, Year_Sold.  But it lacks the *"qualitative"* parameter.  Two houses of the same size, built in the same year within the same neighborhood can have them constructed with different *quality"; for example, vinyl vs tile.
*   While "List_Price" is highly correlated with the "Sale_Price", in this analysis, List_Price serves as the **ONLY** quality parameter.  For example, sellers tend to ask for a higher price if the house has been remodeled recently.

#### 6.  Submission Details
This week, you should be submitting a link to your GitHub repository, which should contain all the deliverables for:

1. the presentation:<br>

*<a href="Final_Project/CA%20Housing%20Prices%20Analysis.pptx" target="_blank">CA_Housing_Prices_Analysis.pptx</a>: this will be edited to be our presentation powerpoint.<br>

*<a href="Final_Project/Final_Project-CA%20Housing%20Price%20Analysis.twbx" target="_blank">Final_Project-CA%20Housing_Price_Analysis.twbx</a>: This is an inter-active dashboard where you can pick and choose county, city or zip on various measure.<br>

2. GitHub:<br>

3. machine learning model: <a href="Final_Project/Final_Regression_NN.ipynb" target="_blank">Final_Regression_NN.ipynb</a><br>

4. and database sections: these two sets of data are used for presentation

* <a href="Final_Project/Data/final_data.csv" target="_blank">final_data.csv</a><br>

* <a href="Final_Project/Data/final_prediction_all_2.csv" target="_blank">final_prediction_all_2.csv</a><br>

= = = = = = = = = = = = = = = = = = = = = = =<br>
•	Roles and Responsibilities:
*   Presentation : Mikhail Zaatra
*   GitHub: Srividhya Thirumalairajan
*   Machine Learning Model : Trong-Quyen Nguyen, Srividhya Thirumalairajan & Mikhail Zaatra
*   DataBase: Dawit Alaro & Trong-Quyen Nguyen
*   DashBoard: Mikhail Zaatra


