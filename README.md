## UC Berkeley AI/ML Certification Program / 2025
### Used Car Pricing Evaluation - Practical Application 2
#### Github location: https://github.com/watermj/used-car-evaluation
#### *Jason Waterman*

## Overview
The purpose of this project to is analyze Kaggle dataset on the sales of used cars called 'vehicles.csv'. This data set contains information on 426k vehicles sold in the US. The goals of the project are two-fold:
- Understand what factors make a car more or less expensive
- Provide clear recommendatons to the used car delearship (client) to what customers value in a used car
We will utilize the <a href="data/crisp.png" target="_blank">CRISP-DM</a> process framework for AI/ML projects.

## Business Problem
We want to determine what drives the sales prices of our used cars. Fortunately, we have access to a comprehensive database of 426k vehicles that gives us actual customer and vehicle data on vehicles that have been sold. We can utilize this nationwide information to help us more clearly understand what are the main drivers of vehicle price, both positive drivers and negative drivers. For this project we will examine the data using modern data mining techniques and data modeling analysis to access the price drivers. We will then provide a list of recommendations for our chain of used car dealerships and their salespersons to improve our overall total sales volumn and profit margins. 

## Data Understanding
We will begin by taking a look at the provided used car database to understand the detail and breath of information we have to analyze. The basic process is as follows:
1. Load in dataset (vehicles.csv)
2. Examine the overall data to see what data is available to be used as features and how clean the dataset is. 
3. Visualize the data to get an overall understanding. We can do this by plotting data.

### Initial Conclusions
- The feature set includes 18 features which is a very reasonable size. Therefore, we can consider all the features in our analysis.
- Price is the defined target. Id probably isn't relavent. So that leaves us with 16 features of primary interest. 
- Location is captured in both region and state. Initially, we will keep both, but expect to be able to drop the state in our final analysis as the region should be the focus for our used car dealerships.

### Data Plot Insights
- ~80% of the sold vehicles <$30,000. Maximum price is around $100k. Neither are surprising. (Figure 1)
- Gas vehicles are by far the most sold vehicles (84%) followed by other (7.2%), diesel (7.1%), hybrid (1.2%), and electric (0.4%). The natural followup question is  what does “other” mean? (Figure 2)
- There is a very clear linear correlation between price and mileage. We can also see that gas vehicles are the most affordable and hybrid are slightly more expensive than gas. (Figure 3)
- For the value conscious consumer (price <$20k) excellent condition has a significant impact on price. (Figure 4) 
- There is an exponential non-linear relationship between year and price. Value drops off by age quickly (Figure 5.1)
- The price for top brands is almost independent with model year. They retain their value (Figure 5.1)
- Tesla and RAM are the top 2 domestic brands for retaining value (Figure 5.1)
- Furthermore, RAM vehicles not only retain their value but have the 4th highest sales volume (Figure 5.2)
- Ford & Chevrolet are the top selling vehicles and account for > 50% of used car market (volume * price) (Figure 5.2)
- For value conscious consumer (sales price <$20k)— sedans and mini-vans account for > 50% sales volume (Figure 6)
- Between $20-40k price point, pickup trucks account for top overall sales (volume * price) (Figure 6)
- Median used car prices vary significantly by state. Most rural states (WV, AK, MT) have a significantly higher average sales price. Need to dive deeper to understand why. (Figure 7)
- The states with the highest number of high sales volume are: CA, FL, NY, TX. Basically the states with the highest populations as you would expect (Figure 8)
- Regions with the highest median sales prices are: Anchorage (AK), Ft.Myers/SW Florida, Hawaii, Las Vegas, Austin (TX), San Antonio (TX). (Figure 9)
- Top sales volume regions: Columbus (OH), Jacksonville (FL), Omaha (NE) (Figure 10)
- Top median sales price regions and also among the top 10 sales volume are: Ft.Meyers/SW Florida, Las Vegas (NV) (Figure 11)
- Overall there is a strong correlation between median sales price and average vehicle mileage (Figure 11)

## Data Preparation
Perform some cleaning up of the data set so that we can run some models. Data cleaning includes:
- Remove rows that have mostly incomplete information
- Remove vehicles where that have no price.
- Remove vehicle price outliers where price <$100 or >$100,0000

Looking back at the relationship between price and a few of the main features that we plotted provides some insights into linear and non-linear relationships with the features. Let's examine a few of these before we beginning modeling to help guide us in our modeling approach.
Let's look at the relationship between price and some of the main features by graphing them out individually. This will let us know

## Modeling
We utilize some basic modeling on the preprocessed vehicles dataset 'vehicles_encoded' using 'price' as the target. Here we perform the follow modeling using cross validation and hyperparameters to optimize our predictions and receive data insights.
- Linear Regression 
- Ridge 
- Lasso
- Random Forest

## Model Evaluation
### Model Performance
Among the four models evaluated — Linear Regression, Ridge, Lasso, and Random Forest — the ensemble model delivered a clear performance advantage.
The Random Forest explains roughly 79% of the variance in the target variable and generalizes well between CV (−7037 RMSE) and test (6710 RMSE), indicating a balanced bias–variance tradeoff.

### Diagnostic Visuals
#### Actual vs Predicted Scatter
The scatter plot shows a dense diagonal pattern, demonstrating that the Random Forest model tracks well with actual outcomes across most of the value range.
However, mild compression appears in the upper range (> 60,000), where the model slightly underpredicts high-value observations—a common behavior when few training examples exist in that region.

#### Residual Distribution
Residuals are centered around zero with no severe skew, but show heteroscedasticity: larger variance at higher predicted values.
This suggests that prediction uncertainty grows for higher-valued vehicles, again likely due to data imbalance rather than model bias.

### Feature Performance
The Random Forest feature importances align strongly with domain expectations:

| Rank  | Feature                            | Importance | Interpretation                                               |
| ----- | ---------------------------------- | ---------- | ------------------------------------------------------------ |
| 1     | **odometer**                       | 0.24       | Usage/mileage is the single most influential factor in value |
| 2     | **year**                           | 0.20       | Newer vehicles retain more value                             |
| 3     | **model**                          | 0.14       | Vehicle model contributes distinct value signals             |
| 4     | **region**                         | 0.09       | Geographic pricing differences affect valuations             |
| 5     | **manufacturer**                   | 0.07       | Brand reputation and reliability premium                     |
| 6-10  | fuel type, transmission, body type | 0.02–0.03  | Reflect drivetrain and configuration effects                 |
| 11-15 | condition, type, missing flags     | < 0.02     | Marginal but helpful refinements                             |

The dominance of odometer and year confirms the expected relationship between vehicle age/usage and market value.
The strong contribution from model and region suggests the model successfully captures localized and categorical pricing variations.

### Overall Assessment
The Random Forest captures complex, non-linear dependencies that linear models miss, providing a 40-point R² improvement.
Its residuals show no systematic bias, only natural variance growth with value scale.
Feature importance analysis enhances interpretability and validates model behavior against domain intuition.
Further performance gains may be achievable through gradient boosting (e.g., HistGradientBoostingRegressor or XGBoost) or data balancing for high-value ranges.

### Conclusion
The Random Forest model delivers robust predictive accuracy and interpretable results for estimating vehicle value.
The model’s insights — particularly the prominence of odometer, year, and model — mirror real-world market dynamics, indicating that the learned patterns are meaningful and reliable.

### Client Recommendations
The data clearly show that cross market the used car vehicle mileage and vehicle age are the strongest drivers of vehicle pricing. Keep this in mind, my recommendations to the nation-wide change of used car delearshps are as follows:
- Focus on keeping a used car fleet of newer vehicles. It is better to move older vehicle inventory at discounts than to keep them in inventory.
- When bidding on used cars either at auction or trade in make strong offers for vehicles with low mileage and try to avoid high-mileage vehicles. If accept a trade-in vehicle with high mileage make sure that factor in the risk in the customer offer. Clearly communicate this to your customers.
- Highest individual sales regions are maily in the midwest (Columbus, Jacksonville, Omaha, Grand Rapids). Support these locations as turnover will be a major driver of overall nationwide sales.
- California has the most number of high sales regions. You should have a regional sales support specifically for the state as it is such a large driver of overall nationwide sales. 
- Fort Meyers/SW Florida and Las Vegas are your brand prestige locations. These locations sell by far higher value vehicles and still manage to be a high savles volume driver for your nationwide sales. Therefore, marketing and branding for these two delearships should cater to these customer base.
- Ford & Chevrolet are you best selling vehilces. However, RAM, Porsche, and (more recently) Tesla have the strongest brand loyalty based on retainment of value over vehicle age.
- Your customers want to purchase vehicles in 'excellent' or 'good' condition. Avoid acquiring used cars that are below these expectations. Make sure you have a uniform checklist across you nationwide dealership to assess vehicle condition consistantly.
- We have a clear numerical feature rankings related to sale price. Develop a uniform pricing calculator to account for top features. Adjust as necessary by specific regions. 

## GitHub Project Repository Structure
<pre>
used-car-evaluation/
├── README.md                     # README - start here for project overview
├── used_car_price_drivers.ipynb  # main project notebook
│
├── data/                         
│   └── vehicles.csv/             # Kaggle used car database
│
├── images/
│   ├── crisp.png/                 # CRISP-DM framework
│   └── kurt.jpeg/                 # run logs, training logs, etc.
│
└── requirements.txt
</pre>


