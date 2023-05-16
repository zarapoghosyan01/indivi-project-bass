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

##




