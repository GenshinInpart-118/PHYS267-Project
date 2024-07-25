#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Data reading
fin = pd.read_csv("data.csv")
selected = fin.loc[0:31, ["distance(m)", "height(m)"]]
d = selected['distance(m)'].to_numpy()
h = selected['height(m)'].to_numpy()

ms = np.array([0.9, 1.0, 1.0, 0.9, 1.0, 0.9, 0.9, 0.9, 1.0])
m = np.mean(ms)
g = 9.80436
p = m*d*np.sqrt(g/(2*h))

# Plot histogram
plt.hist(p, density=True, alpha=0.6, color='g',bins=7, label = "Datas' histogram")

# Fit a normal distribution to the data
mu, std = norm.fit(p)

# Plot the PDF
xmin, xmax = plt.xlim()
x = np.linspace(4, xmax, 100)
p_pdf = norm.pdf(x, mu, std)
plt.plot(x, p_pdf, 'k', linewidth=2, label = "Normal distribution fit")
title = "Distribution of Impulse"

plt.title(title)
plt.xlabel("Impulse(N*s)")
plt.ylabel("Probabililty")
plt.legend()

plt.show()

# Calculate the mean of p
mean_p = np.mean(p)
print(f"Mean of p: {mean_p}")


# In[ ]:




