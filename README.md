# Bass-Diffusion-Model
The Bass diffusion model is a tool that uses mathematics to estimate how quickly new products, technologies, or ideas will be adopted by a population. This model was first introduced by Frank Bass in 1969 and has since become popular in marketing research. It is based on two main factors: how appealing the innovation is and how much influence marketing efforts have on the adoption rate. The model predicts that the adoption of a new product will start slowly, then speed up as more people become aware of it, and eventually slow down as the market becomes saturated. It is helpful for marketers to understand and forecast how new products will spread and to develop effective marketing strategies for their success.


## How to run the code
```python
from bass_model import BassDiffusionModel
import os

current_directory = os.getcwd()

data_file = os.path.join(current_directory, "salesmonthly.csv")  # Appending the file name "salesmonthly.csv" to the current directory

bass_model = BassDiffusionModel(data_file)

future_sales = bass_model.problemsolver()  # Solving the marketing problem

bass_model.visualize_regression()  # Generating the visualization

summary_output = bass_model.summary()  # Obtaining the summary output
```

## Testing Prediction Results
The future sales predictions show an increasing trend over time, starting from 156.96 and reaching a peak at 159.18.
The summary output shows that the mean sales value is 149.99, which is close to the median value of 154.64. 
The standard deviation value of 31.49 suggests that there is some variability in the sales data, which may be due to external factors such as changes in the market, competition, or other factors. 
The minimum sales value is 0, which may indicate a lack of sales or missing data points.
The quartile values show that 50% of sales fall between 137.49 and 169, with a maximum value of 211.13. 
Overall, these results suggest that the Bass model is predicting a successful marketing campaign, with the potential for further growth in sales. Marketers should continue to monitor changes in the market environment and adjust their strategies accordingly to maintain success.

## Different Classes 
- **curv_fit:** a function that is used to perform non-linear regression. Given a set of data points, curve_fit fits a curve to the data by minimizing the sum of the squared residuals between the observed data and the model's predictions. It can be used to fit various types of functions, not just linear regression models.
- **scikit-learn:** a LinearRegression model to fit a linear regression line to the data. 

## Data 
The user is expected to have the sales data in the following order: 
- **Dates:**  the daily/monthly/quarterly sales data
- **Sales:**  the sales throughout the timeframe 

## References 
Bass diffusion model: https://www.immagic.com/eLibrary/ARCHIVES/GENERAL/WIKIPEDI/W101203B.pdf [Accessed May 8th 2022]
By u/alejandropuerto https://github.com/alejandropuerto/product-market-forecasting-bass-model/blob/master/Bass%20Model.ipynb
By https://www.kaggle.com/milanzdravkovic 









