import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class BassDiffusionModel:
    def __init__(self, sales_data_file):
        self.sales_data = pd.read_csv(sales_data_file)
        self.t = np.array(range(len(self.sales_data)))
        self.sales = np.array(self.sales_data['Sales'])
        self.popt = None
        
    def fit(self):
        def bass(t, p, q, m):
            return m * ((p + q)**2 / p * np.exp(-(p+q)*t)) / ((1 + q/p * np.exp(-(p+q)*t))**2)
        
        self.popt, _ = curve_fit(bass, self.t, self.sales)
    
    def plot(self):
        if self.popt is None:
            self.fit()
        
        p, q, m = self.popt
        
        plt.plot(self.t, self.sales, 'o', label='Data')
        plt.plot(self.t, bass(self.t, p, q, m), label='Fit')
        plt.title('Bass Diffusion Model Fit')
        plt.xlabel('Time')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()
    
    def problemsolver(self):
        if self.popt is None:
            self.fit()
        
        p, q, m = self.popt
        return f"The estimated parameters of the Bass Diffusion Model are: p={p:.3f}, q={q:.3f}, and m={m:.3f}."

    def summary(self):
        if self.popt is None:
            self.fit()
        
        p, q, m = self.popt
        n = len(self.sales_data)
        rmse = np.sqrt(np.sum((self.sales - bass(self.t, p, q, m))**2) / n)
        
        summary_str = f"Bass Diffusion Model summary:\n"
        summary_str += f"Estimated parameters:\n"
        summary_str += f"p={p:.3f}, q={q:.3f}, m={m:.3f}\n"
        summary_str += f"RMSE: {rmse:.2f}\n"
        return summary_str



