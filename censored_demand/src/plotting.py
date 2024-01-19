import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_rentals(daily_rentals, daily_stock):
    fig, ax = plt.subplots(1,1,figsize=(12,5))
    t = np.arange( len(daily_stock))
    ax.fill_between(x=t, y1=0, y2=daily_stock, alpha=0.3, color='r' )
    ax.plot(t, daily_stock, alpha=0.3, color='r', label='Stock' )
    ax.plot(t, daily_rentals,color='k', label='Rentals' )
    ax.set(xlabel='Time (days)', ylabel='Count', title='Daily Rentals (black), Constrained by stock levels (red)')
    ax.legend()
    return ax
