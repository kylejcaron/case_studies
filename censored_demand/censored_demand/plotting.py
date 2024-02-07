import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_rentals(daily_rentals, daily_stock):
    fig, ax = plt.subplots(1,1,figsize=(12,5))
    #t = np.arange( len(daily_stock))
    t = daily_rentals.index

    ax.fill_between(x=t, y1=0, y2=daily_stock, alpha=0.3, color='r' )
    ax.plot(t, daily_stock, alpha=0.3, color='r', label='Stock' )
    ax.plot(t, daily_rentals,color='k', label='Rentals' )
    ax.set(xlabel='Time (days)', ylabel='Count', title='Daily Rentals (black), Constrained by stock levels (red)')
    ax.legend()
    return ax


def plot_rentals(daily_rentals, daily_stock, show_stock=True, show_stockouts=False):

    if show_stockouts:
        fig, axes = plt.subplots(1,2, 
                    gridspec_kw= dict(width_ratios=[15], height_ratios=[10,1]), 
                    figsize=(12,5),
                    sharex=True)
    else:
        fig, ax = plt.subplots(1,1,figsize=(12,5))
    
    #t = np.arange( len(daily_stock))
    t = daily_rentals.index

    # plot 1
    axis = axes[0] if show_stockouts else ax
    if show_stock:
        axis.fill_between(x=t, y1=0, y2=daily_stock, alpha=0.3, color='r' )
        axis.plot(t, daily_stock, alpha=0.3, color='r', label='Stock' )
    else:
        axis.axhline(0, color='r', lw=2, ls='--')

    axis.plot(t, daily_rentals,color='k', label='Rentals' )
    axis.set(xlabel='Time (days)', ylabel='Count', title='Daily Rentals (black), Constrained by stock levels (red)')
    axis.legend()

    # Plot 2: stockouts
    if show_stockouts:
        axes[-1].vlines(t[:100][(daily_stock==0).values],0,1, color='r', lw=4)
        axes[-1].set_yticks([])


    return ax