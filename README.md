# Bass-Diffusion-Model
The Bass diffusion model is a tool that uses mathematics to estimate how quickly new products, technologies, or ideas will be adopted by a population. This model was first introduced by Frank Bass in 1969 and has since become popular in marketing research. It is based on two main factors: how appealing the innovation is and how much influence marketing efforts have on the adoption rate. The model predicts that the adoption of a new product will start slowly, then speed up as more people become aware of it, and eventually slow down as the market becomes saturated. It is helpful for marketers to understand and forecast how new products will spread and to develop effective marketing strategies for their success.


## How to run the code
```python
from bass import BassDiffusionModel

# Create a BassDiffusionModel object with your data file
model = BassDiffusionModel('data.csv')

# Solve a specific marketing problem using the Bass Diffusion Model
future_sales = model.problemsolver()

# Generate a summary output of your data
summary = model.summary()

# Visualize the regression model fit
model.visualize_regression()
```




