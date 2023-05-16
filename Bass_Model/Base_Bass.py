import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class BassDiffusionModel:
    def __init__(self, sales_data_file):
        # Initializing the class with sales data from a CSV file
        self.sales_data = pd.read_csv(sales_data_file)
        # Creating an array of time values (in weeks)
        self.t = np.array(range(len(self.sales_data)))
        # Creating an array of sales values
        self.sales = np.array(self.sales_data['Sales'])
        # Initializing the parameter estimates to None
        self.popt = None
        
    def fit(self):
         # Defining the Bass Diffusion Model function
        def bass(t, p, q, m):
            return m * ((p + q)**2 / p * np.exp(-(p+q)*t)) / ((1 + q/p * np.exp(-(p+q)*t))**2)
        
        # Fitting the Bass Diffusion Model function to the sales data
        self.popt, _ = curve_fit(bass, self.t, self.sales)
    
    def plot(self):
        # Fitting the model to the data, if the estimates are not available
        if self.popt is None:
            self.fit()
            
        # Extracting the parameter estimates
        p, q, m = self.popt
        
        # Plotting the data and the fitted curve
        plt.plot(self.t, self.sales, 'o', label='Data')
        plt.plot(self.t, bass(self.t, p, q, m), label='Fit')
        plt.title('Bass Diffusion Model Fit')
        plt.xlabel('Time')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()
    
    def problemsolver(self):
        # Fitting the model to the data, if the estimates are not available
        if self.popt is None:
            self.fit()
        # Extracting the parameter estimates
        p, q, m = self.popt
        
        # Returning a string with the estimated parameter values
        return f"The estimated parameters of the Bass Diffusion Model are: p={p:.3f}, q={q:.3f}, and m={m:.3f}."

    def summary(self):
        # Fitting the model to the data, if the estimates are not available
        if self.popt is None:
            self.fit()
        
        # Extracting the parameter estimates
        p, q, m = self.popt
        
        # Calculating the root-mean-square error (RMSE) of the fit
        n = len(self.sales_data)
        rmse = np.sqrt(np.sum((self.sales - bass(self.t, p, q, m))**2) / n)
        
        # Returning a string with a summary of the model fit
        summary_str = f"Bass Diffusion Model summary:\n"
        summary_str += f"Estimated parameters:\n"
        summary_str += f"p={p:.3f}, q={q:.3f}, m={m:.3f}\n"
        summary_str += f"RMSE: {rmse:.2f}\n"
        return summary_str



