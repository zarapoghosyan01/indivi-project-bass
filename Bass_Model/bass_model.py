import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class BassDiffusionModel:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
    
    def problemsolver(self):
        
        
       
        model = self.build_model()
        future_sales = self.predict_sales(model)
        
        return future_sales
    
    def build_model(self):
        # Prepare the data for regression
        dates = pd.to_datetime(self.data['date'])
        sales = self.data['Sales']
        
        # Convert dates to numeric representation (e.g., days since the first date)
        dates_numeric = (dates - dates.min()).dt.days
        
        # Fit a linear regression model
        model = LinearRegression()
        model.fit(dates_numeric.values.reshape(-1, 1), sales)
        
        return model
    
    def predict_sales(self, model):
        # Generate future sales predictions based on the regression model
        future_dates = pd.date_range(start=self.data['date'].iloc[-1], periods=12, freq='M')
        future_dates_numeric = (future_dates - pd.to_datetime(self.data['date']).min()).days
        
        future_sales = model.predict(future_dates_numeric.values.reshape(-1, 1))
        
        return future_sales
    
    def visualize_regression(self):
        # Visualize the regression model fit
        dates = pd.to_datetime(self.data['date'])
        sales = self.data['Sales']
        
        # Convert dates to numeric representation (e.g., days since the first date)
        dates_numeric = (dates - dates.min()).dt.days
        
        # Fit a linear regression model
        model = LinearRegression()
        model.fit(dates_numeric.values.reshape(-1, 1), sales)
        
        # Plot the actual sales data and the regression line
        plt.figure(figsize=(8, 6))
        plt.scatter(dates, sales, color='b', label='Actual Sales')
        plt.plot(dates, model.predict(dates_numeric.values.reshape(-1, 1)), color='r', label='Regression Line')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title('Regression Model Fit')
        plt.legend()
        plt.show()
    
    def summary(self):
        # Generate a summary output
        summary = self.data.describe()
        return summary