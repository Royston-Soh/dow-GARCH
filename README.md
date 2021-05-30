# Time series forecasting for Dow Jones Industrial Average (using GARCH)
In this post, we will try to model and predict the Dow Jones Industrial Average using Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models, which is an extension of the ARCH(q) model.

In the ARCH(q) model, the time-dependant volatility depends on q lag squared values of the error terms.  

The standard GARCH model assumes none constant variance of the error terms over time, hence we extend the ARCH(q) model by including the p number of lags for error variance, that is also assumed to follow an ARMA process.

```bash
library(xts)
library(quantmod)
library(ggplot2)
library(forecast)
library(rugarch)
library(PerformanceAnalytics)
library(lubridate)
```
## Read historical data for Dow Jones Industrial Average (adjusted closing prices)
```bash
setwd("F:/My Files/R Studio/DOW S&P500")
dow_data=read.csv('dow_data_v3.csv',header = T,sep = ',')
head(dow_data)
tail(dow_data)
```
![](https://github.com/Royston-Soh/dow-GARCH/blob/main/pic/1%20head_tail.jpg)

## Standardize the date format and convert the data to time series xts file type
Let's use the data from year 2008 onward
```bash
dow_data$Date=dmy(dow_data$X)

df=data.frame(dow_data$DJI)
rownames(df)=dow_data$Date
colnames(df)=c('DJI')

df_xts=as.xts(df)
df_xts=df_xts['2008/']
```
![](https://github.com/Royston-Soh/dow-GARCH/blob/main/pic/2%20time%20series.jpg)

## Plot the chart series
We observe an upward multiplicative (exponential) trend, with high volatility. The index enters several peaks before pulling back strongly.
```bash
chartSeries(df_xts,
            theme=chartTheme('white'))
```
![](https://github.com/Royston-Soh/dow-GARCH/blob/main/pic/3%20dow%20plot.jpg)

## Split the data to training and test set
Let's predict and validate for 252 trading days
```bash
length(df_xts$DJI)-252 #Calculate number of training days

training_xts=first(df_xts,'3121 days')
test_xts=last(df_xts, '252 days')
```

## Calculate daily return data for adjusted closing price
```bash
return=CalculateReturns(training_xts)
return=na.omit(return)
View(return)
```
![](https://github.com/Royston-Soh/dow-GARCH/blob/main/pic/4%20returns.jpg)

## Visualize the daily returns data
We observe that the daily returns are close to zero for most days, there are some days with really high returns (5% to 10% etc), similarly for some days with excessive negative returns. 
```bash
hist(return)
```
![](https://github.com/Royston-Soh/dow-GARCH/blob/main/pic/5%20returns%20plot.jpg)

## Add density curve and normal distribution curve
As compared against the normal distribution, curve for return is taller, also implies that there are more extreme values (thicker tails). Distribution for returns looks more like Student t-distribution, rather than the normal distribution.
```bash
chart.Histogram(return,
                methods = c('add.density','add.normal'),
                colorset = c('blue','green','red'))
```
![](https://github.com/Royston-Soh/dow-GARCH/blob/main/pic/6%20returns%20bell%20curve.jpg)

## Plot series
We observe a time series for returns with no sign of seasonality or trend components. There is volatility clustering on certain months
```bash
chartSeries(return,theme='white')
```
![](https://github.com/Royston-Soh/dow-GARCH/blob/main/pic/7%20chart%20series%20for%20returns.jpg)

## Plot for annualized volatility
Once again, we notice very high volatility on certain months as compared to the  rest of the months
```bash
chart.RollingPerformance(R=return['2008::2020'],
                         width = 22,
                         FUN = 'sd.annualized',
                         scale=252,
                         main = 'DJI monthly volatility')
```
![](https://github.com/Royston-Soh/dow-GARCH/blob/main/pic/8%20annual%20volatility.jpg)

## Build and fit the GARCH model
Build a few variations of GARCH model and find the most accurate one by ranking them. 
Create specification for model using `ugarchspec` and store as `s`, by specifying the:

- mean model
- variance model 
- distribution of error terms

### Model 1: We begin with the simplest standard GARCH model with constant mean
```bash
s=ugarchspec(mean.model = list(armaOrder=c(0,0)),
             variance.model = list(model='sGARCH'),
             distribution.model = 'norm')
m=ugarchfit(data = return,spec = s)
m
```

### Interpretation of model output
Optimal Parameters:  
Check if coefficients of the model are statistically significant (ie. P-values <0.05)  
P-values for all 4 coefficients are less than 0.05, and hence statistically significant

Weighted Ljung-Box Test:  
Checks for autocorrelation of residuals  
Ho: Residuals are not autocorrelated  
P-value <0.05 denotes autocorrelation for the respective lags  

Adjusted Pearson Goodness-of-Fit Test:  
Helps verify whether the distribution of the error terms fits the distribution that we have chosen  
Ho: The distribution of the error terms follow the distribution chosen by us  
In our case, with P-values <0.05, we reject the null hypothesis and conclude that the normal distribution chosen by us is not a good fit for the distribution of residuals.  

Information Criteria:  
Statistical measures of fit, generally smaller values means a better model, this can be used to rank various models

### Plot and interprete the 12 charts
plot(m,which='all')

Plot no. 8 and 9 also measures whether the distribution of the error terms fit the distribution that we have chosen.  
Referring to plot no.8 'Empirical Density of Standardized Residuals', we observe that the histogram of standardized residuals is taller than normal distribution. For plot no.9, we observe that the tails deviate from the straight line at extreme values. This further corroborates the above findings from the Adjusted Pearson Goodness-of-Fit Test that the normal distribution that we have chosen is not a good fit for modeling the distribution of error terms.

### ACF plots for evaluating autocorrelation
Referring to ACF plots for observations: plots 4, 5, 6, 7  
We observe that the columns cross the red line, which suggests the presence of significant autocorrelation for the observations  

Next we refer to ACF plots for residuals: plots 10 and 11  
There's much improvement as compared to ACP plots for Observations, most columns are within red line 

## Continue to build and evaluate variations of GARCH models
### Model 2: GARCH with skewed student t-distribution (sstd)
Adopt student t-distribution for distribution of error terms, which is taller with fatter tails
```bash
s=ugarchspec(mean.model = list(armaOrder=c(0,0)),
             variance.model = list(model='sGARCH'),
             distribution.model = 'sstd')
```

### Model 3: gjrGARCH model
Developed by Glosten-Jagannathan-Runkle takes into account the asymmetry of returns that is affected by good or bad news. In the financial markets, bad news typically results in more pronounced impact on volatility, as compared to good news. This is graphically depicted in plot 12
```bash
s=ugarchspec(mean.model = list(armaOrder=c(0,0)),
             variance.model = list(model='gjrGARCH'),
             distribution.model = 'sstd')
```

### Model 4: AR(1) gjrGARCH
Adjusted Pearson Goodness-of-Fit Test: P-values are all lesser than 0.05, hence reject the null hypothesis that the model is a good fit
```bash
s=ugarchspec(mean.model = list(armaOrder=c(1,0)),
             variance.model = list(model='gjrGARCH'),
             distribution.model = 'sstd')
```

### Model 5: gjrGARCH in mean
```bash
s=ugarchspec(mean.model = list(armaOrder=c(0,0),
             archm=T,
             archpow=2),
             variance.model = list(model='gjrGARCH'),
             distribution.model = 'sstd')
```

We observe under 'Optimal Parameters' that coefficient archm is not statistically significant, hence, the addition of this variable does not improve the model.   Disregard model 5  

For models 1 and 4, results from the Adjusted Pearson Goodness-of-Fit Test, indicates that the distribution chosen is not a good fit for the distribution of residuals.  

Model 3 seems to be a better more model than model 2, as it has a lower Information Criteria values  

## Model 3 is the best model
gjrGARCH model
```bash
s_final=ugarchspec(mean.model = list(armaOrder=c(0,0)),
                   variance.model = list(model='gjrGARCH'),
                   distribution.model = 'sstd')

m=ugarchfit(data = return,spec = s_final)
setfixed(s_final)=as.list(coef(m))
```

## Run simulation and store results of simulation in sim
```bash
sim=ugarchpath(spec = s_final,
               m.sim = 1,
               n.sim = 1*252,
               rseed = 123)
```

## Calculate forecasted DOW index
```bash
#Firstly extract the final data point for training set
tail(training_xts)
x=as.numeric(last(training_xts$DJI))
x

p=x*apply(fitted(sim), 2, 'cumsum')+x

df_p=data.frame(p)
predicted=df_p$p
```
## Extract actual index values for test data
```bash
actuals_df=data.frame(test_xts)
actuals=actuals_df$DJI
```

## Visualization of actual vs predicted Dow Jones Industrial Index
```bash
to_plot=data.frame(predicted,actuals)
matplot(to_plot,
        type = 'l',
        lwd = 3,
        xlab = 'Predicted trading day',
        ylab = 'Dow Jones Industrial Average')

legend("bottomright", inset=0.01, legend=colnames(to_plot), col=c(1:2),pch=15:19,
       bg= ("white"), horiz=F)
```

## Model accuracy
```bash
round(accuracy(predicted,actuals),2)
```
