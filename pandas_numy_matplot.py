import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import numpy as np

column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

df = pd.read_csv('auto-mpg.data.txt', sep='\s+', names=column_names)

df_selected = df[['mpg', 'weight']]

np.random.seed(0)
subset_indices = np.random.choice(df_selected.index, size=10, replace=False)
df_subset = df_selected.loc[subset_indices]

x = df_subset['weight'].values
y = df_subset['mpg'].values
lagrange_poly = lagrange(x, y)


x_range = np.linspace(df_selected['weight'].min(), df_selected['weight'].max(), 500)

offset = (x_range.max() - x_range.min()) * 0.001
var = np.polyfit(x, y, 10)
vandermonde_poly = np.poly1d(var)

plt.scatter(df_selected['weight'], df_selected['mpg'], alpha=0.5, label=' Data')

plt.plot(x_range + offset, lagrange_poly(x_range), color='blue', linestyle='-', label='Lagrange Interpolation')

plt.plot(x_range + offset, vandermonde_poly(x_range), color='red', linestyle='--', label='Vandermonde Matrix Interpolation')

plt.title('Interpolation of MPG vs. Weight')
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.legend()
y_min = df_selected['mpg'].min()
y_max = df_selected['mpg'].max()
plt.ylim(y_min, y_max)
plt.show()
