import json, warnings, gc, math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics import r2_score

from types import SimpleNamespace
import argparse

VISUALIZE_FIRST_N = 0

def get_random_sentence(
    max_words: int = 6,
    min_words: int = 3,
    similar_word_map: dict = {},
    places: list = [],
    numericals: list = [],
):
    """
    Get a random title or axis label for each plot
    """
    
    # Get title length
    title_length = np.random.randint(min_words, max_words+1)
    
    if title_length == 1:
        return np.random.choice(similar_word_map[np.random.randint(0, len(similar_word_map))])
    
    # Select random words (can be duplicated)
    title = [np.random.choice(similar_word_map[i]) for i in np.random.choice(range(0, len(similar_word_map)), size=title_length-1)]
    
    # Select a place
    title[-1] = np.random.choice(places)
    
    # Shuffle final title
    np.random.shuffle(title)
    
    # Add numerical
    title.append(np.random.choice(numericals))
    
    return " ".join(title)[:50]
    
def sample_polynomial(polynomial_coeffs, x_min, x_max, num_samples):
    """Samples random values from a polynomial.

    Args:
        polynomial_coeffs (array-like): Coefficients of the polynomial from highest to lowest degree.
        x_min (float): Minimum x value to sample.
        x_max (float): Maximum x value to sample.
        num_samples (int): Number of samples to take.

    Returns:
        ndarray: A 2D array of shape (num_samples, 2) containing (x, y) pairs.
        
    Source: ChatGPT
    """
    # Create an array of x values to sample
    x = np.random.uniform(x_min, x_max, num_samples)

    # Evaluate the polynomial at the x values to get the y values
    y = np.polyval(polynomial_coeffs, x)
    return y

def get_numerical_series(
    poly_pct: float = 0.6,
    linear_pct: float = 0.2,
    part_numerical_pct: float = 0.2,
    series_size: int = 8,
    data_type: str = 'float',
    random_outlier_pct: float = 0.1,
    force_positive: bool = True,
    cat_type: str = "",
    line_plot_xs: bool = False,
    scatter_plot_xs: bool = False,
):
    if line_plot_xs:
        val = np.random.choice([1,2,3,4,5,6,7,8,9,10,15,20,25,30,50,100,250,1000])
        if np.random.random() > 0.5:
            res = [(val*i) for i in range(series_size)]
        else:
            val *= np.random.randint(1,10)
            res = [(val*i) for i in range(1, series_size+1)]
        return res
    
    if scatter_plot_xs and data_type == "int" and np.random.random() < 0.5:
        val = np.random.choice([1,2,3,4,5,6,7,8,9,10,15,20,25,30,50,100,250,1000])
        start = np.random.randint(0, 10_000)
        res = [start+(val*i) for i in range(series_size)]
        return res
        
    
    # Array to return
    res = []
    
    # Multiplication factor
    mult_prob = np.random.choice([0.01, 0.1, 10, 100, 1000, 1000000], 1, p=[0.04, 0.15, 0.55, 0.15, 0.10, 0.01])[0]
    mf = np.random.normal(mult_prob)
    
    # Polynomial or linear sampler
    cat_type = np.random.choice(["poly", "linear"], 1, p=[0.5, 0.5])[0]
    if cat_type == "poly":
        res = sample_polynomial(
            polynomial_coeffs = [np.random.randint(1, 6), np.random.randint(1, 6), np.random.randint(0, 6)], 
            x_min=-1, 
            x_max=1, 
            num_samples=series_size,
        )
    elif cat_type == "linear":
        # Note: Not completely linear, as the last coeff acts as noise
        res = sample_polynomial(
            polynomial_coeffs = [np.random.randint(1, 6), np.random.randint(1, 6), 1+np.random.random()/10], 
            x_min=-1, 
            x_max=1, 
            num_samples=series_size,
        )
    
    # Force Positive
    if force_positive:
        res = [x * -1 if x < 0 else x for x in res]
        
    # Add multiplication factor
    res = [x*mf for x in res]
    
    # Set data type
    if data_type == "float":
        res = [np.clip(np.round(x, 2), 0, 10_000_000) for x in res] # rounding to 2 decimals, clipping negatives
        res = [x if x > max(res)/500 else max(res) for x in res] # making sure values are in range
        res = [np.round(np.random.uniform(0.2, 1.0), 2) if x < 0.00000001 else x for x in res] # corrects values that are all 0
    else:
        res = [np.clip(int(x), 0, 10_000_000) for x in res] # rounding to 2 decimals, cliping negatives
        res = [x if x > max(res)/500 else max(res)//2 for x in res] # making sure values are in range
        res = [np.random.randint(1, 20) if x==0 else x for x in res] # corrects values that are all 0

        
    # Apply random outlier
    if np.random.random() < random_outlier_pct:
        res[np.random.randint(0, len(res))] *= 2
        
    return res

def get_date_series(
    series_size: int = 8,
    line_plot: bool = False,
):
    
    # Only take months when less than 12
    if series_size <= 7:
        date_format = np.random.choice(["year", "month_short", "month_long", "week_short", "week_long"], p=[0.1, 0.1, 0.1, 0.35, 0.35])
    elif series_size <= 12:
        date_format = np.random.choice(["year", "month_short", "month_long"], p=[0.35, 0.35, 0.3])
    else:
        date_format = "year"
    
    if date_format == "year":
        gap = np.random.choice([1,2,5,10])
        start = np.random.randint(1800, 2100 - (series_size*gap))
        res = [int(start + i*gap) for i in range(series_size)]
        if line_plot == False and series_size <= 8 and np.random.random() < 0.1:
            res = ["{}-{}".format(x, x+gap) for x in res]
    
    # Max 12 values
    else:
        if date_format == "month_short":
            res = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            if np.random.random() < 0.5:
                # add day to the month
                if line_plot == False:
                    if np.random.random() < 0.5:
                        res = ["{}-{}".format(x, np.random.randint(1, 30)) for x in res]
                    else:
                        res = ["{}-{}".format(np.random.randint(1, 30), x) for x in res]
        elif date_format == "week_short":
            res = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            if np.random.random() < 0.5:
                # add day to the month
                if line_plot == False:
                    if np.random.random() < 0.5:
                        res = ["{}-{}".format(x, np.random.randint(1, 30)) for x in res]
                    else:
                        res = ["{}-{}".format(np.random.randint(1, 30), x) for x in res]

        elif date_format == "month_long":
            res = ["January", "Febuary", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        elif date_format == "week_long":
            res = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Clip to size of series
        if len(res) > series_size:
            start = np.random.randint(0, len(res)-series_size)
            res = res[start:start+series_size]
        
    res = [str(x) for x in res]
    return res

def get_categorical_series(
    noun_pct: float = 0.1,
    place_pct: float = 0.6,
    part_numerical_pct: float = 0.3,
    chart_type: str = "vertical_bar",
    series_size: int = 8,
    similar_word_map: dict = {},
    places: list = [],
):
    # Array to return
    res = []
    cat_type = np.random.choice(["noun", "place", "part_num"], 1, p=[noun_pct, place_pct, part_numerical_pct])[0]
    
    # Catches case when word is duplicated
    while True:

        # Select based on type
        if cat_type == "noun":
            vals = similar_word_map[np.random.randint(0, len(similar_word_map))]

            add_words = np.random.randint(0,2)
            for i in range(add_words):
                vals2 = similar_word_map[np.random.randint(0, len(similar_word_map))]
                vals2 = np.random.choice(vals2, size=len(vals), replace=True)
                vals = ["{} {}".format(x, y) for x,y in zip(vals, vals2)]

            if len(vals) <= series_size:
                res = vals
            else:
                res = np.random.choice(vals, series_size, replace=False)
                
        if cat_type == "place":
            # Vertical bars are awful with long labels
            if series_size >= 10 and chart_type == "vertical_bar":
                cat_type = "part_num"
            else:
                res = np.random.choice(places, series_size, replace=True)
                res = [x.capitalize() for x in res]
        if cat_type == "part_num":
            while True:
                res = get_numerical_series(
                    series_size = series_size,
                    data_type = "int"
                )
                if max(res) <= 90:
                    break
            
            res = ["{} to {}".format(i, i+np.random.randint(1, 10)) for i in res]

            # Randomly add symbol after number
            symbol = np.random.choice(['in', 'ft', 'yd', 'mi', 'cm', 'm', 'km', 'mg', 'kg', 'oz', 'lb', 'btu', 'cal', 'J', 'kWh', 'V', 'A', 'W', 'Hz', 'min', 'yr', '%', '$', 'ID'])
            if np.random.random() > 0.5:
                res = ["{}{}".format(x, symbol) for x in res]
        
        # Convert to Strings
        res = [str(x) for x in res]
        
        # Random Capitalization
        if np.random.random() > 0.2:
            res = [x.capitalize() for x in res]
        
        # Making sure strings are not too long
        res = [x[:25] + ".." if len(x) > 25 else x for x in res]
        
        # Catches random case when word is repeated in the data
        if len(list(set(res))) == len(res):
            break
            
    return res

def plot_graph(
    xs: list = [],
    ys: list = [], 
    series_size: int = 6,
    remove_spines: bool = False,
    remove_ticks: bool = False,
    bar_color: str = "tab:blue",
    style: str = "classic",
    font_family: str = "DejaVu Sans",
    font_size: int = 10,
    edge_color: str = "black",
    font_style: str = "normal",
    font_weight: str = "normal",
    title_pct: float = 0.5,
    axis_title_pct: float = 0.5,
    fnum: int = 0,
    chart_type: str = "vertical_bar",
    line_plot: bool = False,
    scatter_plot: bool = False,
    dot_plot: bool = False,
    hide_half: bool = False,
):  
    # Do denote what has been trimmed
    n_rem = None
    
    # For label purposes
    if line_plot == True:
        xs = [str(x) for x in xs]
    
    # Set default style
    with plt.style.context(style):
        
        # Font params
        plt.rcParams['font.size'] = font_size
        plt.rcParams["font.family"] = font_family
        plt.rcParams["font.style"] = font_style
        plt.rcParams['font.weight'] = font_weight
        
        # Create figure
        if chart_type == "vertical_bar":
            if dot_plot == True:
                width = 6.5 + (np.random.random()*2)
                height = 3.2 + np.random.random()
            else:
                width = 6 + (np.random.random()*2)
                height = 3.7 + np.random.random()
                
        elif scatter_plot == True:
            width = 5 + np.random.random()
            height = 4 + np.random.random()   
            
        elif chart_type == "horizontal_bar" or line_plot == True:
            width = 4.6 + np.random.random()
            height = 4.6 + np.random.random()   
            
        fig, ax = plt.subplots(figsize=(width, height), num=1, clear=True)
        
        
        # Add variation in the margins
        ax.margins(x=np.random.uniform(0.0, 0.05), y=np.random.uniform(0.08, 0.15))

        # Display the plot
        if chart_type == "vertical_bar":
            if type(xs[0]) == str:
                if line_plot == True:
                    marker = np.random.choice([None, "."], p=[0.8, 0.2])
                    markersize = np.random.randint(10, 20)
                    
                    # Random Smoothing of Line plot
                    if np.issubdtype(type(ys[0]), np.floating) and np.random.random() > 0.20 and len(ys) >= 5:
                
                        # Fit spline and plot
                        n_dots = 1000
                        ys_copy = [float(y) for y in ys]
                        xs_copy = [_ for _ in range(len(xs))]
                        cs = CubicSpline(xs_copy, ys_copy)
                        spline_xs = np.linspace(min(xs_copy), max(xs_copy), n_dots)
                        smoothed_ys = cs(spline_xs)
                        
                        # randomly remove ends of line (from graph conventions)
                        seg = int(n_dots/series_size)
                        n_rem = random_n_remove(ss=series_size)
                        if n_rem != None:
                            txs = spline_xs[n_rem[0]*seg:n_rem[1]*seg]
                            tys = smoothed_ys[n_rem[0]*seg:n_rem[1]*seg]
                            
                            # Only setting if larger than 2 values
                            if len(txs) >= 2:
                                spline_xs = txs
                                smoothed_ys = tys
                                
                        # Plot original data for labels
                        ax.plot(xs, ys, marker=marker, markersize=markersize, color=bar_color, alpha=0.0)
                        
                        # Randomly add dots
                        if np.random.random() > 0.9 and n_rem != None:
                            ax.scatter(xs[n_rem[0]:n_rem[1]], ys[n_rem[0]:n_rem[1]], marker=marker, s=np.random.uniform(50, 75), color=bar_color)
                        
                        ax.plot(spline_xs, smoothed_ys, color=bar_color)
                    else:
                        n_rem = random_n_remove(ss=series_size)
                        if n_rem != None:
                            txs = xs[n_rem[0]:n_rem[1]]
                            tys = ys[n_rem[0]:n_rem[1]]
                        if n_rem != None and len(txs) >= 2:
                            xs = txs
                            ys = tys
                            ax.plot(xs, ys, marker=marker, markersize=markersize, color=bar_color)
                        else:
                            n_rem = None
                            ax.plot(xs, ys, marker=marker, markersize=markersize, color=bar_color)
                        pass
                elif dot_plot == True:
                    dotplot(xs, ys, ax=ax, color=bar_color)
                else:
                    # Need some bars smushed together
                    if np.random.random() < 0.1:
                        ax.bar(xs, ys, color=bar_color, width=1, zorder=3, edgecolor="black")
                    else:
                        ax.bar(xs, ys, color=bar_color, width=np.random.uniform(0.5, 0.9), zorder=3, edgecolor=edge_color)

            else:
                if line_plot == True:
                    marker = np.random.choice([None, "."], p=[0.8, 0.2])
                    markersize = np.random.randint(10, 20)
                    ax.plot(xs, ys, color=bar_color)
                    ax.set_xmargin(np.random.choice([0, np.random.uniform(0, 0.2)]))
                elif scatter_plot == True:
                    markersize=np.random.choice([30,75])
                    ax.scatter(xs, ys, color=bar_color, s=markersize)
                    ax.set_xmargin(np.random.uniform(0.05, 0.2))
                    ax.set_ymargin(np.random.uniform(0.05, 0.2))
                elif dot_plot == True:
                    dotplot(xs, ys, ax=ax, color=bar_color)
                else:
                    ax.bar(xs, ys, color=bar_color, width=((max(xs) - min(xs))/len(xs))*np.random.uniform(0.75,0.9), zorder=3, edgecolor=edge_color)
                
        elif chart_type == "horizontal_bar":
            if type(xs[0]) == str:
                # Need some bars smushed together
                if np.random.random() < 0.06:
                    ax.barh(ys, xs, color=bar_color, height=1, zorder=3, edgecolor="black")
                else:
                    ax.barh(ys, xs, color=bar_color, height=np.random.uniform(0.5, 0.9), zorder=3, edgecolor=edge_color)
            else:
                # Need some bars smushed together
                if np.random.random() < 0.06:
                    ax.barh(ys, xs, color=bar_color, height=1, zorder=3, edgecolor="black")
                else:
                    ax.barh(ys, xs, color=bar_color, height=np.random.uniform(0.5, 0.9), zorder=3, edgecolor=edge_color)
                
        if remove_spines == True:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        if remove_ticks == True:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')

        # WORDS Only: Rotate categorical labels when they likely overlap
        if chart_type == "vertical_bar":
            if type(xs[0]) == str and hide_half == False and (series_size >= 8 or sum([len(x) for x in xs]) > 32 or max([len(x) for x in xs]) >= 7):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Catches
                    done = False
                    for val in "".join(xs):
                        if val not in "0123456789":
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=np.random.choice([45,70]), ha="right")
                            done = True
                            break
                    if done == False:
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=np.random.choice([45,70]))
                            
        
        # Adding Symbols to Numerical Data
        if chart_type == "vertical_bar":
            if max(ys) <= 100 and dot_plot == False:
                if (scatter_plot == True and np.random.random() < 0.5) or (np.random.random() < 0.75):
                    if np.issubdtype(type(ys[0]), np.floating): precision = np.random.randint(1,2)
                    else: precision = np.random.randint(0,2)
                    def formatter(x, pos): return f"{x:.{precision}f}%"
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter))
            elif max(ys) > 1000:
                if np.random.random() < 0.5:
                    if np.random.random() < 0.5: 
                        def formatter(x, pos): return "{:_}".format(round(x))
                    else: 
                        def formatter(x, pos): return "{:,}".format(round(x))
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter))
                    
        elif chart_type == "horizontal_bar":
            if max(xs) <= 100:
                if (np.random.random() < 0.2 and dot_plot == True) or (np.random.random() < 0.75):
                    if np.issubdtype(type(xs[0]), np.floating): precision = np.random.randint(1,2)
                    else: precision = np.random.randint(0,2)
                    def formatter(x, pos): return f"{x:.{precision}f}%"
                    ax.xaxis.set_major_formatter(ticker.FuncFormatter(formatter))
            elif max(xs) > 1000:
                if np.random.random() < 0.5:
                    if np.random.random() < 0.5: 
                        def formatter(x, pos): return "{:_}".format(round(x))
                    else: 
                        def formatter(x, pos): return "{:,}".format(round(x))
                    ax.xaxis.set_major_formatter(ticker.FuncFormatter(formatter))
        
        # Remove axis lines 
        if chart_type == "vertical_bar":
            prob = np.random.random()
            if (prob < 0.03 and dot_plot == False) or (scatter_plot == True and prob < 0.1):
                ax.xaxis.grid(True)
                ax.yaxis.grid(True)
            elif prob > 0.3:
                ax.xaxis.grid(False)
                ax.yaxis.grid(False)
            else:
                ax.xaxis.grid(False)
                ax.yaxis.grid(True)
                
        elif chart_type == "horizontal_bar":
            prob = np.random.random()
            if prob < 0.03:
                ax.xaxis.grid(True)
                ax.yaxis.grid(True)
            elif prob > 0.85:
                ax.xaxis.grid(False)
                ax.yaxis.grid(False)
            else:
                ax.xaxis.grid(True)
                ax.yaxis.grid(False)
                
        # 50% hide dot plot y axis
        if dot_plot == True and np.random.random() < 0.8:
            ax.yaxis.set_tick_params(labelleft=False)
            ax.yaxis.set_ticks_position('none') 
                    
        # Add Titles
        if np.random.random() < title_pct:
            title = get_random_sentence(
                max_words=6,
                min_words=3,
                similar_word_map=similar_word_map,
                places=places,
                numericals=numericals,
            )   
            plt.title(title, pad=np.random.randint(15, 20))

        if np.random.random() < axis_title_pct:
            # Set X axis label
            xtitle = get_random_sentence(
                max_words=6,
                min_words=3,
                similar_word_map=similar_word_map,
                places=places,
                numericals=numericals,
            )   
            plt.xlabel(xtitle)

            # Set Y Axis Label
            ytitle = get_random_sentence(
                max_words=4,
                min_words=1,
                similar_word_map=similar_word_map,
                places=places,
                numericals=numericals,
            )   
            plt.ylabel(ytitle)
        
        # 90% of the time to whitespace
        if np.random.random() < 0.90:
            ax.set_facecolor("white")
            
        if dot_plot == True and hide_half == True and np.random.random() > 0.5:
            for index, label in enumerate(ax.get_xticklabels()):
                if index % 2 == 0:
                    label.set_visible(False)

    #         # Plot width x weight
    #         fig_width, fig_height = plt.gcf().get_size_inches()*fig.dpi
    #         print(fig_width, fig_height)

        # Show plot
        fig.savefig(config.out_file + '/{}.jpg'.format(fnum), bbox_inches='tight')
        if fnum < VISUALIZE_FIRST_N:
            plt.show()

    # Reset default font size
    plt.rcParams['font.size'] = 10
    
    # Uncomment if running in .py script
    # NOTE: If these are uncommented in .iynb file, the execution time goes up exponentially
    plt.close("all")
    plt.close()
    gc.collect()

    return n_rem

def generate_random_color(low=40, high=200, grey=False):
    """
    Function to generate a random color.
    
    The higher the colors, the lighter the color
    """
    if grey == True:
        red = green = blue = np.random.randint(30, 190)
    else:
        red = np.random.randint(low, high)
        green = np.random.randint(low, high)
        blue = np.random.randint(low, high)
    return (red/255, green/255, blue/255)

def nr(y_true, y_pred):
    return sigmoid((1 - np.clip(r2_score(y_true, y_pred), 0, 1) ** 0.5))

def sigmoid(x):
    return 2 - 2 / (1 + np.exp(-x))

def reduce_precision(arr):
    for i in range(-4, 2):
        # Round array
        prec = np.round(arr, decimals=i)
        if i <= 0:
            prec = prec.astype(int)
        prec = list(prec)
        # Check if nrmse is close enough
        if nr(arr, prec) >= 0.98:
            return prec
    return arr

def create_bars(n, horizontal=False):
    """
    Creates N bar charts.
    """
    
    # Possible parameter combinations
    remove_spines = [True, False]
    remove_ticks = [True, False]
    mpl_styles = ["_mpl-gallery", "_mpl-gallery-nogrid", "fivethirtyeight", "ggplot", "seaborn-v0_8-whitegrid", "Solarize_Light2", "bmh", "seaborn-v0_8-darkgrid"]
    font_families = ["Microsoft Sans Serif", "Calibri", "Arial", "Times New Roman", "Comic Sans MS"]
    font_families = ["Calibri", "DejaVu Sans", "Tahoma", "Verdana"]
    font_sizes = range(7, 9)
    edge_colors = ["none", "black"]
    font_styles = ["normal", "italic"]
    font_weights = ["normal", "bold"]
    series_sizes = range(2, 20)
    grey_pct = 0.5

    
    meta_data = []
    for i in tqdm(range(n)):

        # Sample from styles
        rspines = np.random.choice(remove_spines)
        rticks = np.random.choice(remove_ticks)
        bcolor = generate_random_color(grey=np.random.choice([False, True], p=[1-grey_pct, grey_pct])) 
        mplstyle = np.random.choice(mpl_styles)
        ffamily = np.random.choice(font_families)
        fsize = np.random.choice(font_sizes)
        ecolor = np.random.choice(edge_colors)
        fstyle = np.random.choice(font_styles)
        fweight  = np.random.choice(font_weights)
        
        # Sample chart types
        if horizontal == True:
            chart_type = "horizontal_bar"
            ss = np.random.choice(series_sizes, p=[0.011, 0.011, 0.103, 0.229, 0.069, 0.069, 0.103, 0.069, 0.069, 0.056, 0.034, 0.034, 0.034, 0.029, 0.029, 0.017, 0.017, 0.017])
        else:
            ss = np.random.choice(series_sizes, p=[0.016, 0.058, 0.103, 0.202, 0.117, 0.148, 0.089, 0.036, 0.076, 0.04 , 0.052, 0.007, 0.022, 0.009, 0.002, 0.009, 0.007, 0.007])
            chart_type = "vertical_bar"

        # Sample data series type based on probability from calculation in Frequencies section 
        x_type = np.random.choice(x_counts[chart_type]['dst'], p=x_counts[chart_type]['pct'])
        y_type = np.random.choice(y_counts[chart_type]['dst'], p=y_counts[chart_type]['pct'])
        
        while True:
            # Categorical data series
            if (x_type == 'str' and chart_type == "vertical_bar") or (y_type == 'str' and chart_type == "horizontal_bar"):
                xs = get_categorical_series(
                    series_size = ss,
                    places = places,
                    similar_word_map = similar_word_map,
                    chart_type = chart_type,
                    noun_pct = 0.2,
                    place_pct = 0.6,
                    part_numerical_pct = 0.2,
                )   
            else:
                xs = get_date_series(
                    series_size = ss,
                )
            ys = get_numerical_series(
                series_size = ss,
                data_type = y_type,
            )

            # flip if horizontal
            if chart_type == "horizontal_bar":
                xs, ys = ys, xs

            if len(xs) == len(ys):
                break

        # Create graph
        plot_graph(
            xs = xs,
            ys = ys,
            series_size = ss,
            remove_spines = rspines,
            remove_ticks = rticks,
            bar_color = bcolor,
            style = mplstyle,
            font_family = ffamily,
            font_size = fsize,
            edge_color = ecolor,
            font_style = fstyle,
            font_weight = fweight,
            fnum = i,
            chart_type = chart_type,
        )
        
        # Sanity check: log data
        if i < VISUALIZE_FIRST_N:
            if chart_type == "horizontal_bar":
                print(xs[::-1], ys[::-1])
            else:
                print(xs, ys)
        
        # # Reduce precision
        # if isinstance(xs[0], str) == False:
        #     xs = reduce_precision(xs)
        # if isinstance(ys[0], str) == False:
        #     ys = reduce_precision(ys)
        
        # Sanity check: log data
        if i < VISUALIZE_FIRST_N:
            if chart_type == "horizontal_bar":
                print(xs[::-1], ys[::-1])
            else:
                print(xs, ys)
        
        # Prepare input data
        if chart_type == "vertical_bar":
            text = " <0x0A> ".join(["{} | {}".format(x,y) for x,y in zip(xs, ys)])
            meta_data.append({'file_name': 'graphs_v/{}.jpg'.format(i), 'text': text})
        elif chart_type == "horizontal_bar":
            # Traverse x and y in reverse for horizontal
            text = " <0x0A> ".join(["{} | {}".format(x,y) for x,y in zip(xs[::-1], ys[::-1])])
            meta_data.append({'file_name': 'graphs_h/{}.jpg'.format(i), 'text': text})
        
    # Close last img
    plt.close()
        
    # Write MetaData to Disk
    meta_data = pd.DataFrame(meta_data)
    meta_data["validation"] = 0
    meta_data["count"] = meta_data["text"].str.count("<0x0A>") + 1
    meta_data["chart_type"] = chart_type
    meta_data.to_csv(config.out_file + "/metadata.csv", index=False)
    return meta_data

def random_n_remove(ss):
    """
    For line plots. Select a random number of points to remove.
    """
    n_rem = None
    if np.random.random() > 0.80:
        if ss < 8 or np.random.random() > 0.3:
            n_rem = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            if n_rem == 1: n_rem = (1,-1)
            elif n_rem == 2: n_rem = (2,-2)
            else:
                if np.random.random() > 0.5: n_rem = (1,-2)
                else: n_rem = (2,-1)
        else:
            n_rem = np.random.randint(1, 5)
            if np.random.random() > 0.5: n_rem = (n_rem, ss)
            else: n_rem = (0, -1*n_rem)
    return n_rem

def create_lines(n):
    """
    Creates N line plots charts.
    
    NOTE: uses "vertical_bar" as code is similar
    """
    
    # Possible parameter combinations
    remove_spines = [True, False]
    remove_ticks = [True, False]
    mpl_styles = ["_mpl-gallery", "_mpl-gallery-nogrid", "fivethirtyeight", "ggplot", "seaborn-v0_8-whitegrid", "Solarize_Light2", "bmh", "seaborn-v0_8-darkgrid"]
    font_families = ["Microsoft Sans Serif", "Calibri", "Arial", "Times New Roman", "Comic Sans MS"]
    font_families = ["Calibri", "DejaVu Sans", "Tahoma", "Verdana"]
    font_sizes = range(7, 9)
    edge_colors = ["none", "black"]
    font_styles = ["normal", "italic"]
    font_weights = ["normal", "bold"]
    series_sizes = range(2, 20)
    grey_pct = 0.4
    
    meta_data = []
    for i in tqdm(range(n)):

        # Sample from styles
        rspines = np.random.choice(remove_spines)
        rticks = np.random.choice(remove_ticks)
        bcolor = generate_random_color(grey=np.random.choice([False, True], p=[1-grey_pct, grey_pct])) 
        mplstyle = np.random.choice(mpl_styles)
        ffamily = np.random.choice(font_families)
        fsize = np.random.choice(font_sizes)
        ecolor = np.random.choice(edge_colors)
        fstyle = np.random.choice(font_styles)
        fweight  = np.random.choice(font_weights)
        ss = np.random.choice(series_sizes, p=[0.017, 0.044, 0.096, 0.189, 0.101, 0.101, 0.066, 0.054, 0.071, 0.084, 0.049, 0.027, 0.029, 0.007, 0.017, 0.022, 0.015, 0.011])
        
        # Sample chart types
        chart_type = "vertical_bar"

        # Sample data series type based on probability from calculation in Frequencies section 
        x_type = np.random.choice(x_counts[chart_type]['dst'], p=x_counts[chart_type]['pct'])
        y_type = np.random.choice(y_counts[chart_type]['dst'], p=y_counts[chart_type]['pct'])
        
        # Categorical data series
        if x_type == 'str':
            xs = get_date_series(
                series_size = ss,
                line_plot = True,
            )

        # Year data series (as int)
        else:
            xs = get_numerical_series(
                series_size = ss,
                line_plot_xs = True,
            )
            
        ys = get_numerical_series(
            series_size = ss,
            data_type = y_type,
        )

        # Create graph
        n_rem = plot_graph(
            xs = xs,
            ys = ys,
            series_size = ss,
            remove_spines = rspines,
            remove_ticks = rticks,
            bar_color = bcolor,
            style = mplstyle,
            font_family = ffamily,
            font_size = fsize,
            edge_color = ecolor,
            font_style = fstyle,
            font_weight = fweight,
            fnum = i,
            chart_type = chart_type,
            line_plot = True,
        )
        print(n_rem)
        if n_rem != None:
            xs = xs[n_rem[0]:n_rem[1]]
            ys = ys[n_rem[0]:n_rem[1]]
        
        # Sanity check: log data
        if i < VISUALIZE_FIRST_N:
            print(xs, ys)
        
        # # Reduce precision (xs are always categorical)
        # if isinstance(ys[0], str) == False:
        #     ys = reduce_precision(ys)
        
        # Sanity check: log data
        if i < VISUALIZE_FIRST_N:
            print(xs, ys)
        
        # Prepare input data
        if chart_type == "vertical_bar":
            text = " <0x0A> ".join(["{} | {}".format(x,y) for x,y in zip(xs, ys)])
        meta_data.append({'file_name': 'graphs_l/{}.jpg'.format(i), 'text': text})
        
        
    # Close last img
    plt.close()
        
    # Write MetaData to Disk
    meta_data = pd.DataFrame(meta_data)
    meta_data["validation"] = 0
    meta_data["count"] = meta_data["text"].str.count("<0x0A>") + 1
    meta_data["chart_type"] = "line"
    meta_data.to_csv(config.out_file + "/metadata.csv", index=False)
    return meta_data

def dotplot(xs, ys, ax, **args):
    """
    Function that creates dot plots.
    """
    # Convert 1D input into 2D array
    scatter_x = [] # x values 
    scatter_y = [] # corresponding y values
    for x, y in zip(xs, ys):
        for z in range(y):
            scatter_x.append(x)
            scatter_y.append(z+0.5)
            
    # draw dot plot using scatter() 
    ax.scatter(scatter_x, scatter_y, **args, s=np.random.uniform(300, 330), zorder=3)
    
    # Show all unique x-values
    ax.set_xticks(xs)
    
    # Change major ticks to show every 1 value
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Margin so dots fit on screen
    ax.set_xmargin(np.random.uniform(0.04, 0.2))
    ax.set_ylim([0, 10])
    
def create_dots(n):
    """
    Creates N dot plots.
    
    NOTE: uses "vertical_bar" as code is similar.
    """
    
    # Possible parameter combinations
    remove_spines = [True, False]
    remove_ticks = [True, False]
    mpl_styles = ["_mpl-gallery", "_mpl-gallery-nogrid", "fivethirtyeight", "ggplot", "seaborn-v0_8-whitegrid", "Solarize_Light2", "bmh", "seaborn-v0_8-darkgrid"]
    font_families = ["Microsoft Sans Serif", "Calibri", "Arial", "Times New Roman", "Comic Sans MS"]
    font_families = ["Calibri", "DejaVu Sans", "Tahoma", "Verdana"]
    font_sizes = range(7, 9)
    edge_colors = ["none", "black"]
    font_styles = ["normal", "italic"]
    font_weights = ["normal", "bold"]
    series_sizes = range(2, 21)
    grey_pct = 0.2
    
    meta_data = []
    for i in tqdm(range(n)):

        # Sample from styles
        rspines = np.random.choice(remove_spines)
        rticks = np.random.choice(remove_ticks)
        bcolor = generate_random_color(grey=np.random.choice([False, True], p=[1-grey_pct, grey_pct])) 
        mplstyle = np.random.choice(mpl_styles)
        ffamily = np.random.choice(font_families)
        fsize = np.random.choice(font_sizes)
        ecolor = np.random.choice(edge_colors)
        fstyle = np.random.choice(font_styles)
        fweight  = np.random.choice(font_weights)
        ss = np.random.choice(series_sizes, p=[0.009, 0.002, 0.009, 0.007, 0.011, 0.016, 0.058, 0.11, 0.202, 0.117, 0.148, 0.078, 0.036, 0.076, 0.04, 0.052, 0.007, 0.011, 0.011])
        
        # Sample chart types
        chart_type = "vertical_bar"
        hide_half = False

        # Sample data series type based on probability from calculation in Frequencies section 
        x_type = np.random.choice(["date", "str", "int"])
        
        if x_type == "str":
            xs = get_categorical_series(
                series_size = ss,
                places = places,
                similar_word_map = similar_word_map,
                chart_type = chart_type,
                noun_pct = 0.2,
                place_pct = 0.6,
                part_numerical_pct = 0.2,
            )   
        elif x_type == "date":
            xs = get_date_series(
                series_size = ss,
            )
        else:
            if np.random.random() > 0.5:
                xs_start = 0
            else:
                xs_start = np.random.randint(0, 100)
            xs = [str(xs_start + i) for i in range(ss)]
            hide_half = True

        # Choose datatype
        ys = [np.random.randint(1, 10) for i in range(ss)]        
        
        # Create graph
        plot_graph(
            xs = xs,
            ys = ys,
            series_size = ss,
            remove_spines = rspines,
            remove_ticks = rticks,
            bar_color = bcolor,
            style = mplstyle,
            font_family = ffamily,
            font_size = fsize,
            edge_color = ecolor,
            font_style = fstyle,
            font_weight = fweight,
            fnum = i,
            chart_type = chart_type,
            dot_plot = True,
            hide_half = hide_half,
        )
        
        # Sanity check: log data
        if i < VISUALIZE_FIRST_N:
            print(xs, ys)
        
        # Prepare input data
        if chart_type == "vertical_bar":
            text = " <0x0A> ".join(["{} | {}".format(x,y) for x,y in zip(xs, ys)])
        meta_data.append({'file_name': 'graphs_d/{}.jpg'.format(i), 'text': text})
        
    # Close last img
    plt.close()
        
    # Write MetaData to Disk
    meta_data = pd.DataFrame(meta_data)
    meta_data["validation"] = 0
    meta_data["count"] = meta_data["text"].str.count("<0x0A>") + 1
    meta_data["chart_type"] = "dot"
    meta_data.to_csv(config.out_file + "/metadata.csv", index=False)
    return meta_data

def random_polynomial(x):
    degree = np.random.choice([1,2,3], p=[0.05, 0.50, 0.45])  # Random degree between 1 and 4
    coefficients = np.random.uniform(-4, 4, size=degree + 1)
    y = np.polyval(coefficients, x)
    return y

def scale_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = (arr - min_val) / (max_val - min_val)
    return scaled_arr

def get_scatter_line_sample(ss, x_mean, y_mean):
    """
    MAX 40 (maybe less.)
    """

    x_min = -4
    x_max = 4
    x_values = np.linspace(x_min, x_max, ss)
    y_values = random_polynomial(x_values)

    x_values = np.round(scale_array(x_values)*x_mean, decimals=2)
    y_values = np.round(scale_array(y_values)*y_mean, decimals=2)
    return x_values, y_values
    
    return x_values, y_values

def get_meshgrid(N):
    cn, rn = 15, 15  # number of columns/rows
    xs = np.linspace(-cn/2, cn/2, cn)
    ys = np.linspace(-rn/2, rn/2, rn)

    # meshgrid will give regular array-like located points
    Xs, Ys = np.meshgrid(xs, ys)  #shape: rn x cn

    # create some uncertainties to add as random effects to the meshgrid
    mean = (0, 0)
    varx, vary = 0.02, 0.02
    # adjust these number to suit your need
    cov = [[varx, 0], [0, vary]]
    uncerts = np.random.multivariate_normal(mean, cov, (rn, cn))

    # Create meshgrid
    xs = (Xs+uncerts[:,:,0]).flatten()
    ys = (Ys+uncerts[:,:,1]).flatten()

    # Scale values
    xs = (xs-np.min(xs))/(np.max(xs)-np.min(xs))
    ys = (ys-np.min(ys))/(np.max(ys)-np.min(ys))

    # Select IDXs
    idxs = np.random.choice(range(2, len(xs)), size=N, replace=False)
    xs = xs[idxs]
    ys = ys[idxs]
    return xs, ys

def create_scatters(n):
    """
    Creates N scatter plot charts.
    
    NOTE: uses "vertical_bar" as code is similar.
    """
    
    # Possible parameter combinations
    remove_spines = [True, False]
    remove_ticks = [True, False]
    mpl_styles = ["_mpl-gallery", "_mpl-gallery-nogrid", "fivethirtyeight", "ggplot", "seaborn-v0_8-whitegrid", "Solarize_Light2", "bmh", "seaborn-v0_8-darkgrid"]
    font_families = ["Microsoft Sans Serif", "Calibri", "Arial", "Times New Roman", "Comic Sans MS"]
    font_families = ["Calibri", "DejaVu Sans", "Tahoma", "Verdana"]
    font_sizes = range(7, 9)
    edge_colors = ["none", "black"]
    font_styles = ["normal", "italic"]
    font_weights = ["normal", "bold"]
    series_sizes = range(3, 86)
    grey_pct = 0.2
    
    meta_data = []
    for i in tqdm(range(n)):

        # Sample from styles
        rspines = np.random.choice(remove_spines)
        rticks = np.random.choice(remove_ticks)
        bcolor = generate_random_color(grey=np.random.choice([False, True], p=[1-grey_pct, grey_pct])) 
        mplstyle = np.random.choice(mpl_styles)
        ffamily = np.random.choice(font_families)
        fsize = np.random.choice(font_sizes)
        ecolor = np.random.choice(edge_colors)
        fstyle = np.random.choice(font_styles)
        fweight  = np.random.choice(font_weights)
        ss = np.random.choice(series_sizes)
        
        # Sample chart types
        chart_type = "vertical_bar"

        # Sample data series type based on probability from calculation in Frequencies section 
        x_type = np.random.choice(["int", "float"])
        y_type = np.random.choice(["int", "float"])
        
        # Forces points to be not overlap each other
        while True:
            xs = get_numerical_series(
                series_size = ss,
                data_type = x_type,
                scatter_plot_xs = True,
            )

            ys = get_numerical_series(
                        series_size = ss,
                        data_type = y_type,
                    )
            
            if np.random.random() < 0.99 and len(xs) <= 30:
                xs, ys = get_scatter_line_sample(len(xs), np.nanmean(xs), np.nanmean(ys))
            else:
                # Make random sample more efficient
                mxs, mys = get_meshgrid(ss)
                xs = np.round(mxs*np.mean(xs), decimals=2)
                ys = np.round(mys*np.mean(ys), decimals=2)
            break
        
        # Make sure order of data is correct
        ys = [y for _, y in sorted(zip(xs, ys))]
        xs = sorted(xs)

        # Create graph
        plot_graph(
            xs = xs,
            ys = ys,
            series_size = ss,
            remove_spines = rspines,
            remove_ticks = rticks,
            bar_color = bcolor,
            style = mplstyle,
            font_family = ffamily,
            font_size = fsize,
            edge_color = ecolor,
            font_style = fstyle,
            font_weight = fweight,
            fnum = i,
            chart_type = chart_type,
            scatter_plot = True,
        )
        
        # Sanity check: log data
        if i < VISUALIZE_FIRST_N:
            print(xs, ys)
        
        # # Reduce precision
        # xs = reduce_precision(xs)
        # ys = reduce_precision(ys)
        
        # Sanity check: log data
        if i < VISUALIZE_FIRST_N:
            print(xs, ys)
        
        # Prepare input data
        if chart_type == "vertical_bar":
            text = " <0x0A> ".join(["{} | {}".format(x,y) for x,y in zip(xs, ys)])
        meta_data.append({'file_name': 'graphs_s/{}.jpg'.format(i), 'text': text})
        
    # Close last img
    plt.close()
    
    # Write MetaData to Disk
    meta_data = pd.DataFrame(meta_data)
    meta_data["validation"] = 0
    meta_data["count"] = meta_data["text"].str.count("<0x0A>") + 1
    meta_data["chart_type"] = "scatter"
    meta_data.to_csv(config.out_file + "/metadata.csv", index=False)
    return meta_data

def generation_helper(chart_type, n, all_charts=False):
    """
    Helper function to make chart generation easier.
    """
    if chart_type not in "vhlsd" or len(chart_type) != 1:
        raise ValueError(f"{chart_type} not a valud chart type.")
    if chart_type == "v" or all_charts == True:
        meta_data = create_bars(n, horizontal=False)
    if chart_type == "h" or all_charts == True:
        meta_data = create_bars(n, horizontal=True)
    if chart_type == "l" or all_charts == True:
        meta_data = create_lines(n)
    if chart_type == "s" or all_charts == True:
        meta_data = create_scatters(n)
    if chart_type == "d" or all_charts == True:
        meta_data = create_dots(n)
    return meta_data

def parse_args():
    # Default values
    config = SimpleNamespace(
        out_file="./graphs",
        generate_n_imgs=100_000,
        seed=9090,
        categoricals_file="./categoricals.json",
        ctype='v',
    )
    # Argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out_file", type=str, default=config.out_file, help="Directory to save charts in.")
    parser.add_argument("--generate_n_imgs", type=int, default=config.generate_n_imgs, help="Number of charts to generate.")
    parser.add_argument("--seed", type=int, default=config.seed, help="Seed used to generate data.")
    parser.add_argument("--ctype", type=str, default=config.ctype, help="Flag to determine which graph to generate. (v, h, l, s, d)" )
    args = parser.parse_args()

    # Update config w/ parameters passed through CLI
    for key, value in vars(args).items():
        setattr(config, key, value)

    return config

if __name__ == "__main__":

    # Get config
    config = parse_args()

    # Make directory for output
    if not os.path.exists(config.out_file):
        os.mkdir(config.out_file)

    # Get random categorical variables
    x_counts = {'dot': {'dst': ['int', 'str'], 'pct': [0.26, 0.74]}, 'line': {'dst': ['int', 'str'], 'pct': [0.37, 0.63]}, 'scatter': {'dst': ['float'], 'pct': [0.99]}, 'vertical_bar': {'dst': ['int', 'str'], 'pct': [0.3, 0.7]}, 'horizontal_bar': {'dst': ['int', 'float'], 'pct': [0.16, 0.84]}}
    y_counts = {'dot': {'dst': ['int'], 'pct': [1.0]}, 'line': {'dst': ['int', 'float'], 'pct': [0.12, 0.88]}, 'scatter': {'dst': ['int', 'float'], 'pct': [0.13, 0.87]}, 'vertical_bar': {'dst': ['int', 'float'], 'pct': [0.16, 0.84]}, 'horizontal_bar': {'dst': ['int', 'str'], 'pct': [0.3, 0.7]}}

    numericals = ['milliwatts', 'carbon emissions', 'town', 'tax', 'tonnes', 'seasonal', 'daltons', 'densities', 'biodiversity index', 'AUD', 'easterlin', 'milliliters', 'having', 'almonds', 'maison', 'cornstarch', 'terabit', 'aeterna', 'page', 'low', 'rains', 'volts', 'balancing', 'shortfall', 'came', 'salary', 'ton', 'squash', 'toxic chemicals management', 'simmer', 'mu', 'capacity', 'than', 'foot', 'inventory', 'rambow', 'centimeters', 'fall', 'spent', 'prices', 'transparency of environmental reporting', 'grit size', 'missed', 'billions', 'bentley', 'monsoon', 'longitudes', 'intensity', 'becquerels', 'wind energy production', 'mmhg', 'persons', 'skeat', 'TWD', 'cheese', 'mopsus', 'koss', 'ohms', 'gas', 'heart rate', 'punt', 'gain', 'predominantly', 'water quality index', 'scoring', 'eurytus', 'manoir', 'gwh', 'george', 'wage', 'SGD', 'sulfite content', 'fee', 'fiscal year', 'terms', 'carbs', 'waiting', 'borrowing', 'cubic meters', 'lists', 'rotates', 'ounce', 'buying', 'now', 'kilometre', 'milliampere', 'fat content', 'kb', 'angular', 'starch content', 'cubic inches', 'living', 'gbps', 'gw', 'psi', 'decibel', 'derived', 'annually', 'kilobits', 'declines', 'weierstrass', 'acid deposition', 'interest', 'carbon dioxide concentration', 'disk', 'benefits', 'ftlbf', 'month', 'kick', 'amd', 'heav', 'lenders', 'dlrs', 'threatened species protection', 'levels', 'wildfire', 'kilometers', 'our', 'consume', 'om', 'entries', 'city', 'turmeric', 'butter', 'nini', 'polyphenol content', 'mb', 'aged', 'swept', 'cylinder', 'kartheiser', 'co2 emissions per capita', 'pi', 'oregano', 'melting point', 'efficiency', 'outlook', 'milliohm', 'kilograms', 'communities', 'water activity', 'htc', 'amount', 'picofarad', 'levins', 'wildfires', 'raisins', 'salinity level', 'registered', 'belletti', 'megapascals', 'maintaining', 'kirkeby', 'kilohertz', 'persichetti', 'national parks coverage', 'circumference', 'creuset', 'since', 'money', 'mbit', 'standard', 'computes', 'milligrammes', 'altitude', 'carbohydrate', 'pension', 'catches', 'loop', 'rainstorms', 'credit', 'lufthansa', 'market', 'measurements', 'earlier', 'ounces per square inch', 'distance', 'needed', 'nh', 'kilobytes', 'global warming potential', 'megapixels', 'petrol', 'million', 'ratio', 'unpaid', 'crape', 'dollars', 'floods', 'coriander', 'kilometer', 'manning', 'periods', 'westinghouse', 'imported', 'invest', 'older', 'serca', 'greenhouse gas emissions', 'area', 'philip', 'nanogram', 'bpu', 'proposing', 'needs', 'alarmingly', 'blood', 'fees', 'body mass index', 'tons', 'deferred', 'plummeting', 'one', 'ficus', 'temperature', 'grain size', 'kgs', 'loads', 'decreases', 'toxicity index', 'daimler', 'school enrollment', 'imrt', 'days', 'tau', 'waist', 'from', 'habitat fragmentation', 'revolutions per minute', 'metric', 'euros', 'manufacturing', 'shafayat', 'gottlieb', 'gibson', 'inflation', 'repayment', 'voicestream', 'ventresca', 'fritz', 'introduction', 'long', 'kilocalories', 'vitality score', 'freshwater scarcity index', 'hovered', 'compensation', 'per', 'night', 'gasoline', 'outputs', 'milliseconds', 'rain', 'teaspoons', 'oxidation stability', 'barometric', 'thick', 'relative humidity', 'investing', 'far', 'cooler', 'square inches', 'village', 'during', 'danta', 'kib', 'KRW', 'fiber length', 'registers', 'johnny', 'premium', 'measuring', 'lived', 'variance', 'threads', 'hydropower', 'water pollution index', 'length', 'vinegar', 'water scarcity', 'mwe', 'generators', 'thz', 'inflorescence', 'stein', 'ampere', 'infestation level', 'gigabytes', 'span', 'horsepower', 'aira', 'stimulus', 'out', 'millisieverts', 'renewable water resources', 'sliced', 'girth', 'eick', 'threw', 'mg', 'extinguish', 'cilantro', 'arntzen', 'mpa', 'consumer', 'itt', 'powder', 'holway', 'snowfall', 'books', 'kvs', 'arat', 'reducing', 'investments', 'point', 'utility cost', 'bags', 'prius', 'terrestrial protected areas coverage', 'precipitation', 'mile', 'years', 'terabits', 'products', 'diced', 'fumbled', 'electricity consumption', 'cranks', 'dama', 'galerie', 'werner', 'khz', 'investment', 'slowdown', 'scorching', 'when', 'landslides', 'item', 'pascals', 'wyville', 'ago', 'coral reef health index', 'cinnamon', 'kia', 'pound', 'kev', 'located', 'm', 'mercury emissions', 'devastating', 'budgets', 'rainfalls', 'rebates', 'lorazepam', 'dough', 'strips', 'mostly', 'vmax', 'dimensions', 'share', 'ETH', 'picocuries', 'carl', 'earnings', 'megabits', 'fice', 'decibels', 'bbb', 'papers', 'excess', 'nitrogen oxide emissions', 'paid', 'enrollment rate', 'minced', 'flavius', 'undergraduate', 'depression score', 'next', 'USD', 'ev', 'square meters', 'marine protected areas coverage', 'octane', 'surface', 'nantha', 'taruna', 'forecasts', 'meridian', 'radius', 'spewed', 'angle', 'harrison', 'sales', 'microliter', 'energy efficiency', 'rcc', 'endocrine disruptors', 'microgram', 'henriksen', 'nutritional', 'pixels', 'cumin', 'bushfires', 'bottle', 'score', 'forest cover', 'already', 'gram', 'kilometeres', 'caseload', 'veilleux', 'qaisar', 'mhz', 'social security', 'yearly income', 'benthic habitat quality', 'gigawatts', 'kw', 'tailpipe', 'du', 'wappingers', 'microns', 'turbines', 'inventories', 'humidity', 'twh', 'wielder', 'kilobyte', 'profit margin', 'pint', 'payments', 'last', 'layers', 'soluble fiber content', 'purchases', 'above', 'ratios', 'microsecond', 'beneath', 'illustrated', 'chives', 'batteries', 'just', 'mev', 'weights', 'gundobad', 'volume', 'parallel', 'flour protein content', 'fraport', 'autoregressive', 'byte', 'ozone depletion potential', 'urban heat island intensity', 'medicare', 'fund', 'shortest', 'lifespan', 'teaspoon', 'begin', 'saic', 'nauk', 'gost', 'environmental governance index', 'gross energy value', 'line', 'nanometers', 'gallons', 'water content', 'time', 'theta', 'phase', 'kmh', 'lipid oxidation', 'monopulse', 'inches', 'weight', 'months', 'thickness', 'surplus', 'generating', 'lux', 'ocean acidification index', 'children', 'latitudes', 'stormwater management', 'loi', 'rental rate', 'employment rate', 'payment', 'exceeds', 'radiofrequency', 'mth', 'candela', 'shrank', 'milepost', 'meter', 'food', 'schenk', 'copy', 'clothing', 'compares', 'dynes', 'depends', 'east', 'buk', 'strawberries', 'expectancy', 'mtoe', 'dry matter content', 'melting', 'bloch', 'kg', 'size', 'comprise', 'torso', 'map', 'morning', 'sulfur dioxide emissions', 'alpha', 'femtosecond', 'passes', 'pieces', 'fat', 'reach', 'phi', 'afternoon', 'honeywell', 'diskette', 'moody', 'separator', 'minimum wage', 'fires', 'output', 'assets', 'shipment', 'hefty', 'corresponds', 'mudslides', 'mental health score', 'price', 'bruce', 'torque', 'rpms', 'weekend', 'liter', 'fatty acid profile', 'temperatures', 'arity', 'curing time', 'centimetres', 'kilos', 'runs', 'durations', 'mw', 'but', 'heating', 'hence', 'costs', 'mainly', 'confidence', 'interception', 'solid waste generation', 'monochrome', 'nutrient density', 'watts', 'impellers', 'centimeter', 'increases', 'erg', 'typical', 'silverado', 'water use efficiency', 'heat', 'budgetary', 'affluence', 'roentgens', 'thomas', 'lines per inch', 'glycemic index', 'powerbook', 'minutes', 'kompany', 'cores', 'gigabits', 'karry', 'ph level', 'besco', 'water consumption per capita', 'expenditures', 'lending', 'flood risk', 'starting', 'iota', 'cholesterol', 'water consumption', 'milligram', 'kilowatt', 'celsius', 'henry', 'payable', 'birth', 'flagon', 'child', 'decades', 'dividend', 'phosphorus balance', 'equivalent', 'halflings', 'stockpiles', 'web', 'calculating', 'turbo', 'grubs', 'charles', 'least', 'birthrate', 'drams', 'ardh', 'speeding', 'btus', 'adolphe', 'cssd', 'octets', 'thyme', 'trehalose', 'rest', 'megs', 'deciliter', 'bowl', 'debt', 'boeremag', 'kilocycles', 'evening', 'parsley', 'billion', 'wacc', 'once', 'total solids content', 'nearly', 'consumed', 'lowest', 'valuation', 'dietary fiber content', 'celeron', 'touchdown', 'kms', 'land use change', 'potassium content', 'shares', 'favorability', 'bachelor', 'electron volts', 'GBP', 'packages', 'bytes', 'gigahertz', 'megawatts', 'millimeters', 'watt', 'yield strength', 'dreier', 'taxes', 'cloves', 'stock', 'tablespoons', 'ionising', 'sodium', 'stockholders', 'p/e ratio', 'corpuscular', 'doppler', 'ratings', 'bosomy', 'emr', 'lactic acid bacteria count', 'around', 'miles', 'male', 'analysts', 'residual sugar level', 'radians', 'touchdowns', 'shrinks', 'pensions', 'lambda', 'millimetres', 'magazines', 'cent', 'budget', 'leaving', 'balance', 'llamas', 'projected', 'hertz', 'pfn', 'sector', 'gunther', 'microprocessors', 'robert', 'microfarad', 'regions', 'beta', 'volt', 'altimetry', 'bottles', 'megabytes', 'franz', 'blood pressure', 'downpours', 'lbp', 'preferably', 'turbocharged', 'ft', 'fuel', 'kilojoules', 'duration', 'frequencies', 'frederick', 'cookie', 'legendre', 'sls', 'dots', 'keeping', 'beyond', 'nanoseconds', 'winkleman', 'carob', 'salaries', 'usage', 'seed count', 'achieve', 'retailers', 'plastic waste generation', 'rate', 'increased', 'granulated', 'kwh', 'after', 'ulam', 'families', 'cros', 'HKD', 'shots', 'keystrokes', 'salting', 'kiloliters', 'chopped', 'printouts', 'saturated', 'peak', 'curvature', 'mazur', 'worth', 'generates', 'guaranteed', 'metamaterials', 'CHF', 'MXN', 'decade', 'megalitres', 'negative', 'lumens', 'free fatty acid content', 'rebate', 'marine ecosystem health index', 'crime rate', 'theorem', 'rushing', 'furthermore', 'electricity consumption per capita', 'centigrade', 'hz', 'torrential', 'inflation rate', 'patient satisfaction', 'particulate matter concentration', 'cubes', 'lasts', 'financing', 'weeks', 'whey protein content', 'boil', 'essays', 'humidity level', 'texture profile analysis', 'items', 'cash', 'pretax', 'contract', 'tenskwatawa', 'previous', 'elizabeth', 'CNY', 'feet', 'retirement', 'buyback', 'value', 'stac', 'diems', 'nomi', 'kilobit', 'quarterly', 'diameter', 'nth', 'interest rate', 'karmazin', 'weller', 'present', 'milligrams', 'premiums', 'mortality', 'teenie', 'sir', 'amk', 'cannon', 'age', 'max', 'before', 'ferrars', 'disks', 'ecological footprint', 'chevy', 'BTC', 'powdered', 'mozzarella', 'electric current', 'grams', 'approximate', 'mileage', 'phytic acid content', 'shorten', 'thermal', 'week', 'megajoules', 'kgo', 'rather', 'inhabitants', 'wetland coverage', 'aveo', 'pay', 'zero', 'micrometres', 'modem', 'euler', 'macronutrient ratio', 'urban sprawl', 'mccs', 'where', 'mbps', 'clerc', 'obesity rate', 'roi', 'normal', 'riccati', 'modems', 'tall', 'materials', 'income', 'roughly', 'ilm', 'cocoa percentage', 'same', 'micrograms', 'sattva', 'bulk', 'material footprint', 'weiss', 'bowls', 'bpmn', 'infrasound', 'cruze', 'measured', 'defect count', 'raged', 'medroxyprogesterone', 'hitting', 'unemployement', 'royce', 'land area', 'heroin', 'coming', 'diable', 'protein content', 'cholesterol content', 'kilogrammes', 'profits', 'stronger', 'financial', 'total assets', 'expenses', 'proportion', 'ended', 'climate change vulnerability', 'sizes', 'excluding', 'mpg', 'heavy metal contamination', 'newman', 'grammes', 'spoon', 'asset', 'chatta', 'expenditure', 'insoluble fiber content', 'along', 'gross', 'cm', 'combination', 'stress', 'coar', 'petabyte', 'grobler', 'payout', 'accumulations', 'ball', 'touchscreen', 'mm', 'chunks', 'near', 'gigabyte', 'printed', 'amino acid score', 'jewelry', 'rocknroll', 'funding', 'species richness', 'banking', 'paying', 'sugar content', 'unsold', 'uncompressed', 'meiert', 'joules', 'rainy', 'moisture', 'ohm', 'megahertz', 'henchoz', 'sax', 'ghz', 'rating', 'iodine value', 'samuel', 'gauss', 'cup', 'mortgage', 'unacceptably', 'omega', 'stock price', 'differs', 'rated', 'milliarcseconds', 'total debt', 'serving size', 'trans fat content', 'opl', 'hunger rate', 'methane emissions', 'body temperature', 'minus', 'invested', 'hourly wage', 'carbohydrates', 'turbofans', 'james', 'air quality index', 'terabyte', 'john', 'growth', 'nanometres', 'preservative level', 'nease', 'anemic', 'shf', 'savings', 'articles', 'cialdini', 'ptf', 'kilovolt', 'groundwater depletion rate', 'gains', 'mayonnaise', 'ravn', 'pesticide use', 'fish stocks', 'vitamin', 'geographical', 'ability', 'life', 'capella', 'chemical pollution index', 'letters', 'winds', 'shareholders', 'hpa', 'vlf', 'lamborghini', 'per capita waste generation', 'lundi', 'handfuls', 'populated', 'west', 'had', 'hotter', 'energy density', 'geothermal', 'microplastic pollution', 'kilogram', 'nahai', 'centimetre', 'nitrate content', 'cuts', 'drought index', 'sells', 'megabyte', 'saucepan', 'jardin', 'property value', 'geographic', 'bernstein', 'particulate matter emissions', 'working capital', 'MYR', 'loan', 'fraternity', 'schiavelli', 'microlitre', 'pesticide residue level', 'kbps', 'cognitive ability', 'diameters', 'food safety index', 'specified', 'vary', 'harsiese', 'boxes', 'probability', 'EUR', 'natural disaster risk', 'cut', 'charger', 'affected', 'motors', 'richard', 'pontiac', 'tablespoon', 'pages', 'continuous', 'daha', 'renewable energy consumption', 'vogel', 'ifr', 'laptop', 'mazda', 'twice', 'units sold', 'collection', 'hazardous waste generation', 'nvidia', 'bank', 'kilometers per hour', 'maximum', 'centrifuge', 'micrometers', 'hours', 'mixing', 'refers', 'yeast and mold count', 'inr', 'current', 'laiho', 'due', 'boltzmann', 'microgrammes', 'chevrolet', 'sensory acceptability score', 'digestibility score', 'this', 'debts', 'fumarate', 'bucks', 'air pollution', 'exported', 'priorities', 'liters', 'ecliptic', 'sips', 'gdp', 'declination', 'soars', 'definition', 'seconds', 'moyne', 'nearby', 'water efficiency', 'oxygen demand index', 'dividends', 'change', 'femminile', 'been', 'downgraded', 'funds', 'baud', 'megawatt', 'supply', 'salt', 'calories', 'herman', 'repay', 'gallon', 'dried', 'mbpd', 'inch', 'ranges', 'dissolved oxygen concentration', 'encode', 'read', 'XRP', 'tesla', 'william', 'each', 'year', 'net income', 'selling', 'organic chemical pollution', 'flooding', 'square centimeters', 'increase', 'text', 'wastewater treatment', 'repayments', 'payouts', 'nanofarad', 'ericson', 'less', 'lot size', 'food waste generation', 'weber', 'fertility rate', 'severance', 'mapped', 'birthrates', 'fleurs', 'frequency', 'eastern', 'stories', 'nearest', 'shortly', 'THB', 'throughput', 'avis', 'wreaked', 'servings', 'brix value', 'fumble', 'immigration rate', 'millihenry', 'barrels', 'caffeine', 'mustard', 'maintain', 'antibiotic use in livestock', 'periodic', 'cavus', 'storage', 'nehrling', 'phosphorus content', 'pecans', 'forecast', 'market share', 'overdrive', 'uses', 'genetic variation', 'linguine', 'plan', 'elevation', 'cpu', 'stack', 'then', 'cregg', 'grain', 'wheat gluten content', 'cendant', 'CAD', 'need', 'populations', 'boosted', 'comparison', 'kilopascals', 'floppies', 'sustainable agriculture policy stringency', 'kiloliter', 'raise', 'JPY', 'example', 'risk score', 'aquifer depletion rate', 'sitric', 'libations', 'fmu', 'calorie', 'latitude', 'kash', 'glut', 'mmx', 'bonuses', 'farenheit', 'start', 'athlon', 'on', 'bits', 'vitamin a content', 'direction', 'weighing', 'southern', 'density', 'spewing', 'gigaflops', 'megabit', 'arthur', 'iron content', 'subzero', 'panasonic', 'sak', 'mold count', 'quarter', 'total phenolic content', 'decline', 'riou', 'pectin content', 'fares', 'higher', 'hydrogenation level', 'speeds', 'net', 'thawing time', 'minute', 'scheme', 'ludwig', 'horizontal', 'metre', 'profit', 'abdulhakim', 'annuity', 'kappa', 'kegs', 'off', 'engines', 'equal', 'carbohydrate content', 'global temperature', 'joblessness', 'fahrenheit', 'bailout', 'quarts', 'revenues', 'PHP', 'caceres', 'only', 'merchandise', 'proofed', 'ravaged', 'sublimation', 'total fat content', 'eberhard', 'spending', 'lumix', 'sea level rise', 'downgrades', 'gamma', 'banks', 'wholesale', 'starts', 'ecosystem services value', 'radioactive pollution', 'wages', 'quality score', 'kilowatts', 'litres', 'business', 'carapace', 'square feet', 'km', 'dabhol', 'sprinkle', 'thread', 'gel strength', 'krw', 'difference', 'dots per inch', 'coordinates', 'spilled', 'areas', 'engulfed', 'water availability', 'luminance', 'pesticide regulation stringency', 'edward', 'day', 'opteron', 'farad', 'below', 'fiscal', 'values', 'retail', 'to', 'frames per second', 'wpm', 'expense', 'infinitesimals', 'oil content', 'sustainable fisheries', 'eoc', 'clean energy production', 'drank', 'suvs', 'skim', 'particular', 'millilitre', 'distances', 'elongated', 'gwathmey', 'loans', 'equator', 'within', 'available', 'storms', 'reform', 'damphousse', 'vertical', 'nu', 'trade balance', 'berdahl', 'coastal habitat quality', 'cocaine', 'shallots', 'blaze', 'cooling', 'electricity', 'francis', 'goods', 'hugh', 'litre', 'revenue', 'shurik', 'cubic feet', 'bmus', 'pints', 'droughts', 'portfolio', 'ochratoxin a level', 'ounces', 'undesirably', 'NZD', 'flatus', 'moisture content', 'newtons', 'reference', 'fluid ounces', 'miles per hour', 'pickups', 'sandisk', 'processors', 'soleil', 'wind speed', 'hashish', 'biochemical oxygen demand', 'terabytes', 'snowfalls', 'kramer', 'barely', 'liouville', 'heavy metal pollution index', 'water holding capacity', 'protected area coverage', 'kulasegaran', 'water solubility index', 'wetland health index', 'accessories', 'meyer', 'generations', 'karl', 'mannesmann', 'friedrich', 'short', 'intervals', 'cubic', 'vitamin c content', 'moreover', 'sigma', 'seismicity', 'nne', 'couple', 'cups', 'rainfall', 'square miles', 'editions', 'downgrade', 'metres', 'toricelli', 'product', 'kilometres', 'shorter', 'paprika', 'pepper', 'nuclear waste generation', 'range', 'mccoy', 'tannin content', 'terahertz', 'montano', 'ages', 'plaintext', 'volant', 'outside', 'job satisfaction', 'half', 'people', 'stagnating', 'grayscale', 'height', 'beto', 'buyout', 'interceptions', 'firestorms', 'shaw', 'passing', 'pixel', 'alone', 'land degradation', 'pounds per square inch', 'caloric density', 'broth', 'northeast', 'hayes', 'degree', 'glabrous', 'speed', 'width', 'shipped', 'singa', 'engel', 'soil health index', 'lbs', 'unemployment rate', 'population', 'operating expenses', 'mbar', 'coldest', 'account', 'adjacent', 'azria', 'cleary', 'saturated fat content', 'quart', 'garlic', 'northwest', 'taxpayers', 'minimal', 'beck', 'offset', 'colder', 'estimated', 'expectations', 'rtl', 'mwh', 'fig', 'book', 'river health index', 'albers', 'longitude', 'strength', 'pounds', 'stability', 'volatile organic compound emissions', 'ridership', 'total antioxidant capacity', 'leibniz', 'cazorla', 's', 'yards', 'reliability', 'gross profit', 'ubu', 'equity', 'mixture', 'bennett', 'tackles', 'raging', 'cash flow', 'towns', 'vlahos', 'credit score', 'hour', 'in', 'southwest', 'epsilon', 'life expectancy', 'comparable', 'rechargeable', 'minimum', 'resistor', 'applies', 'petabytes', 'package', 'meters', 'degress', 'again', 'gramme', 'availability', 'putting', 'pentium', 'degrees', 'hasse', 'insert', 'birgeneau', 'firms', 'payroll', 'audible', 'INR', 'anaemic', 'boiling', 'purchasing', 'karrer', 'depending', 'fabricio', 'schneider', 'cards', 'vitamin d content', 'consumes', 'recently', 'northern', 'engine', 'blazes', 'ppm', 'square millimeters', 'cost', 'water stress', 'hydroelectricity', 'data', 'upsilon', 'thermal conductivity', 'doms', 'processor', 'joseph', 'soil erosion', 'reduced', 'powerpc', 'census', 'scd', 'insurance', 'dyne', 'print', 'millivolt', 'enrollments', 'pene', 'weighs', 'addresses', 'sodium content', 'level', 'varies', 'toys', 'equation', 'recycling rate', 'management', 'celcius', 'reduces', 'tax rate', 'exceeded', 'jobless', 'supercharged']
    places = ['monaco', 'dupage', 'storey', 'anasco', 'guinea-bissau', 'vinton', 'dc', 'leflore', 'lewis', 'snyder', 'powell', 'meade', 'decatur', 'schley', 'norway', 'india', 'woodruff', 'willacy', 'chatham', 'phillips', 'niobrara', 'bertie', 'stillwater', 'emporia', 'calvert', 'towner', 'swift', 'kalkaska', 'winkler', 'tuolumne', 'duchesne', 'ouachita', 'egypt', 'salinas', 'gillespie', 'clayton', 'palm', 'sheridan', 'houghton', 'dunklin', 'coal', 'pipestone', 'dewitt', 'mcmullen', 'baldwin', 'geneva', 'stanley', 'canovanas', 'randolph', 'weld', 'halifax', 'barbour', 'beckham', 'catron', 'conway', 'chaves', 'bossier', 'de', 'new mexico', 'wake', 'north', 'yakutat', 'rabun', 'huntingdon', 'gooding', 'nicaragua', 'dickey', 'tippecanoe', 'evans', 'bronx', 'nacogdoches', 'tanzania', 'appomattox', 'sebastian', 'arthur', 'garvin', 'ponce', 'clark', 'freestone', 'ivory coast', 'okanogan', 'deschutes', 'converse', 'rockcastle', 'colorado', 'north dakota', 'stafford', 'boulder', 'steele', 'braxton', 'cotton', 'bosnia herzegovina', 'wetzel', 'tuvalu', 'monroe', 'ralls', 'benzie', 'montenegro', 'teller', 'quebradillas', 'thurston', 'kingman', 'nash', 'duval', 'coleman', 'eastland', 'lumpkin', 'lafayette', 'yolo', 'kaufman', 'corozal', 'ia', 'washoe', 'slope', 'thailand', 'grand', 'lenoir', 'walla', 'bedford', 'sarpy', 'morrill', 'mccone', 'piute', 'kandiyohi', 'transylvania', 'canadian', 'george', 'waldo', 'blackford', 'hughes', 'sherburne', 'tooele', 'dolores', 'sargent', 'mcminn', 'kane', 'minidoka', 'kalawao', 'fallon', 'hudson', 'sanilac', 'ohio', 'motley', 'dixon', 'surry', 'lac', 'grafton', 'walton', 'carlisle', 'nelson', 'saratoga', 'aransas', 'bear', 'il', 'scott', 'nv', 'hanson', 'greenwood', 'venango', 'bryan', 'argentina', 'manatee', 'orocovis', 'laurel', 'grant', 'malaysia', 'brevard', 'dauphin', 'kittson', 'peach', 'belknap', 'cottonwood', 'southampton', 'gurabo', 'dekalb', 'ontonagon', 'gasconade', 'iraq', 'oregon', 'titus', 'east timor', 'beadle', 'gray', 'asotin', 'columbus', 'glascock', 'samoa', 'eastern', 'terrebonne', 'mcduffie', 'ringgold', 'albemarle', 'zimbabwe', 'meeker', 'calloway', 'telfair', 'nicollet', 'tom', 'tinian', 'windham', 'pottawatomie', 'abbeville', 'allen', 'nowata', 'leavenworth', 'wadena', 'allendale', 'greece', 'ketchikan', 'benin', 'chickasaw', 'harnett', 'chattahoochee', 'dooly', 'ritchie', 'mozambique', 'pine', 'dare', 'livingston', 'qatar', 'toombs', 'oh', 'dawes', 'hartley', 'blair', 'saipan', 'merced', 'millard', 'humphreys', 'nm', 'dade', 'renville', 'lunenburg', 'bristol', 'tallapoosa', 'missouri', 'mohave', 'ravalli', 'ware', 'mccracken', 'hettinger', 'dixie', 'flagler', 'warren', 'greene', 'barber', 'bledsoe', 'fredericksburg', 'herkimer', 'isabella', 'liberty', 'sibley', 'ozaukee', 'cidra', 'dakota', 'tama', 'milwaukee', 'yavapai', 'massachusetts', 'upton', 'seminole', 'stewart', 'wicomico', 'clatsop', 'suffolk', 'frio', 'spartanburg', 'meriwether', 'granite', 'bernalillo', 'lithuania', 'wilson', 'leslie', 'washington', 'pend', 'marion', 'gordon', 'amador', 'northern america', 'whiteside', 'schleicher', 'charlton', 'bond', 'letcher', 'loudon', 'waller', 'boundary', 'barceloneta', 'menominee', 'sierra', 'harney', 'mongolia', 'jerome', 'russell', 'ut', 'fall', 'waushara', 'malawi', 'rapides', 'north america', 'yoakum', 'hormigueros', 'brewster', 'washita', 'oconto', 'iberville', 'ontario', 'obion', 'payne', 'dinwiddie', 'tuscaloosa', 'athens', 'dillon', 'wilkinson', 'shasta', 'el salvador', 'mathews', 'slovenia', 'plumas', 'holmes', 'ellsworth', 'towns', 'nd', 'spink', 'dougherty', 'oldham', 'clay', 'concordia', 'hardee', 'vermilion', 'curry', 'clarion', 'southern africa', 'sarasota', 'norman', 'mississippi', 'marengo', 'bates', 'brunswick', 'catano', 'fulton', 'collin', 'saluda', 'charlottesville', 'yauco', 'fentress', 'genesee', 'ak', 'ciales', 'coahoma', 'kay', 'king', 'kanawha', 'wrangell', 'casey', 'yemen', 'alger', 'ferry', 'dubois', 'bailey', 'california', 'audrain', 'addison', 'philippines', 'vt', 'mahaska', 'panola', 'jack', 'chaffee', 'lynchburg', 'anoka', 'alcona', 'val', 'tazewell', 'sauk', 'louisa', 'lake', 'person', 'brookings', 'nigeria', 'central asia', 'greer', 'ct', 'alameda', 'kosciusko', 'weston', 'el', 'gonzales', 'ziebach', 'wichita', 'pittsburg', 'cyprus', 'irwin', 'brazoria', 'waseca', 'kazakhstan', 'ascension', 'patrick', 'saline', 'delaware', 'isanti', 'onondaga', 'dunn', 'douglas', 'otoe', 'churchill', 'coosa', 'pitt', 'rolette', 'south america', 'coweta', 'archer', 'cuyahoga', 'stanton', 'rio', 'hood', 'okeechobee', 'ottawa', 'eddy', 'geauga', 'vernon', 'highland', 'kinney', 'nez', 'lassen', 'orange', 'muscatine', 'effingham', 'cuming', 'sacramento', 'barrow', 'vermillion', 'luna', 'doniphan', 'hickman', 'mexico', 'fluvanna', 'woodbury', 'custer', 'sac', 'colfax', 'mifflin', 'minnehaha', 'platte', 'boone', 'westchester', 'ionia', 'muscogee', 'la', 'wheatland', 'real', 'missaukee', 'kentucky', 'cannon', 'guthrie', 'glacier', 'clermont', 'itasca', 'barnwell', 'montezuma', 'silver', 'kit', 'trujillo', 'va', 'crawford', 'onslow', 'morrison', 'washburn', 'broadwater', 'al', 'alpena', 'hatillo', 'gates', 'ulster', 'muskegon', 'aguadilla', 'leake', 'anson', 'estill', 'south africa', 'fairfax', 'habersham', 'maui', 'clackamas', 'tift', 'blaine', 'kiowa', 'benton', 'tuscarawas', 'winn', 'okmulgee', 'greenbrier', 'swaziland', 'taylor', 'mille', 'bath', 'alfalfa', 'benewah', 'alexandria', 'hinds', 'pierce', 'dominican republic', 'stone', 'ford', 'pennington', 'meigs', 'mcdonough', 'ozark', 'gallia', 'garza', 'sweetwater', 'magoffin', 'goodhue', 'fl', 'rincon', 'chautauqua', 'caribbean', 'arenac', 'pocahontas', 'eritrea', 'aiken', 'texas', 'kershaw', 'nottoway', 'hooker', 'hancock', 'hayes', 'starr', 'tattnall', 'berkeley', 'cook', 'schuyler', 'loup', 'bosque', 'moody', 'llano', 'hillsdale', 'nevada', 'coffey', 'covington', 'creek', 'taiwan', 'clarke', 'skagway', 'haiti', 'lackawanna', 'mckenzie', 'malheur', 'barry', 'chowan', 'roberts', 'sri lanka', 'crittenden', 'stokes', 'bourbon', 'golden', 'delta', 'vanderburgh', 'colquitt', 'syria', 'lebanon', 'southern europe', 'siskiyou', 'kootenai', 'medina', 'niger', 'maverick', 'montmorency', 'miller', 'langlade', 'page', 'penobscot', 'wells', 'haywood', 'defiance', 'gunnison', 'aibonito', 'lagrange', 'tompkins', 'menard', 'hamblen', 'karnes', 'otero', "o'brien", 'mchenry', 'sequoyah', 'miner', 'coamo', 'davie', 'solano', 'alleghany', 'merrick', 'caroline', 'cooke', 'ri', 'taliaferro', 'naranjito', 'franklin', 'baylor', 'cherokee', 'pemiscot', 'latah', 'pope', 'nobles', 'lubbock', 'micronesia', 'maricopa', 'cowley', 'hudspeth', 'tripp', 'stanislaus', 'wasco', 'italy', 'dale', 'wayne', 'montgomery', 'carson', 'baltimore', 'san', 'ramsey', 'san marino', 'rockingham', 'muskogee', 'south sudan', 'mariposa', 'korea south', 'moore', 'garrard', 'malta', 'rutland', 'vieques', 'albania', 'skamania', 'poland', 'bland', 'cibola', 'iceland', 'de', 'mo', 'peoria', 'highlands', 'riley', 'bollinger', 'erie', 'hendry', 'burkina', 'louisiana', 'montcalm', 'trousdale', 'cape', 'edgecombe', 'armenia', 'mcculloch', 'equatorial guinea', 'bradley', 'md', 'bolivar', 'callahan', 'sedgwick', 'currituck', 'gloucester', 'deer', 'luxembourg', 'walker', 'santa', 'alamance', 'putnam', 'johnson', 'oneida', 'macomb', 'northern', 'aurora', 'cochise', 'polynesia', 'finney', 'western africa', 'saginaw', 'porter', 'marshall', 'teton', 'mcintosh', 'southern asia', 'kearney', 'wade', 'st.', 'burundi', 'pa', 'davison', 'lee', 'wahkiakum', 'lamoure', 'la', 'denali', 'faulk', 'gadsden', 'carver', 'weber', 'catahoula', 'litchfield', 'taney', 'bradford', 'afghanistan', 'rwanda', 'penuelas', 'candler', 'scioto', 'griggs', 'prince', 'morton', 'graves', 'bladen', 'niagara', 'mower', 'tensas', 'yazoo', 'peru', 'rota', 'emmet', 'hale', 'powder', 'walsh', 'linn', 'ector', 'buckingham', 'comanche', 'preston', 'orleans', 'belgium', 'charles', 'spotsylvania', 'dukes', 'suwannee', 'guayama', 'oscoda', 'toole', 'arecibo', 'caswell', 'australia and new zealand', 'hill', 'giles', 'sandusky', 'alamosa', 'tunisia', 'catoosa', 'harper', 'adair', 'multnomah', 'tioga', 'grenada', 'traverse', 'roane', 'cooper', 'tx', 'portugal', 'blue', 'oceania', 'nebraska', 'forest', 'kingfisher', 'mahoning', 'pinellas', 'long', 'hardin', 'lavaca', 'berks', 'whatcom', 'darke', 'racine', 'winona', 'cabarrus', 'providence', 'craven', 'nodaway', 'rusk', 'minnesota', 'spokane', 'bulloch', 'nc', 'valley', 'bolivia', 'lafourche', 'thayer', 'audubon', 'wilcox', 'archuleta', 'elkhart', 'sutton', 'coos', 'botetourt', 'hamlin', 'scotts', 'loiza', 'pushmataha', 'rappahannock', 'vanuatu', 'josephine', 'aitkin', 'chenango', 'belarus', 'camden', 'jennings', 'iran', "manu'a", 'toa', 'bowie', 'screven', 'wisconsin', 'seychelles', 'somerset', 'auglaize', 'bahrain', 'juniata', 'barren', 'turkey', 'morehouse', 'vega', 'ray', 'quay', 'whitley', 'indiana', 'kenai', 'lancaster', 'ashland', 'yakima', 'hansford', 'baker', 'maries', 'watauga', 'coke', 'brooks', 'summers', 'ogle', 'france', 'early', 'lipscomb', 'grayson', 'ceiba', 'manati', 'dorado', 'chisago', 'romania', 'graham', 'tehama', 'maricao', 'colombia', 'burleson', 'crosby', 'natrona', 'armstrong', 'pike', 'ashley', 'etowah', 'camuy', 'shelby', 'lanier', 'gaston', 'saudi arabia', 'craig', 'mineral', 'morovis', 'sevier', 'fisher', 'trigg', 'garden', 'mauritania', 'africa', 'baxter', 'grady', 'orangeburg', 'guaynabo', 'bucks', 'white', 'cecil', 'rosebud', 'choctaw', 'monmouth', 'charlevoix', 'st lucia', 'tyrrell', 'sampson', 'japan', 'aguada', 'fajardo', 'iredell', 'gosper', 'jefferson', 'koochiching', 'gilliam', 'nassau', 'clarendon', 'des', 'togo', 'ransom', 'searcy', 'cole', 'appling', 'richland', 'barnstable', 'hodgeman', 'butte', 'keith', 'ma', 'sumter', 'hampshire', 'oklahoma', 'chicot', 'arroyo', 'miami', 'wood', 'pecos', 'martin', 'runnels', 'windsor', 'oglethorpe', 'lauderdale', 'elko', 'licking', 'irion', 'winnebago', 'ouray', 'ingham', 'guadalupe', 'st kitts & nevis', 'kiribati', 'waukesha', 'butts', 'rose', 'azerbaijan', 'hinsdale', 'barron', 'yukon-koyukuk', 'crowley', 'stephens', 'hockley', 'idaho', 'larimer', 'becker', 'macoupin', 'emanuel', 'conecuh', 'cortland', 'humboldt', 'harmon', 'staunton', 'upshur', 'logan', 'cleveland', 'seward', 'rankin', 'beaverhead', 'robeson', 'gilchrist', 'foard', 'keya', 'uzbekistan', 'waupaca', 'nemaha', 'allegheny', 'tolland', 'mclean', 'arlington', 'inyo', 'carter', 'benson', 'caguas', 'stutsman', 'mellette', 'sagadahoc', 'escambia', 'gibson', 'colusa', 'cochran', 'bullock', 'pleasants', 'cloud', 'preble', 'trempealeau', 'sully', 'chelan', 'marathon', 'bartow', 'lincoln', 'monongalia', 'hubbard', 'oconee', 'zavala', 'dickinson', 'island', 'gilmer', 'richardson', 'merrimack', 'pickett', 'victoria', 'andorra', 'culberson', 'desha', 'mt', 'keweenaw', 'isle', 'quitman', 'wasatch', 'leon', 'eureka', 'villalba', 'estonia', 'mccormick', 'oakland', 'gage', 'natchitoches', 'doddridge', 'sanborn', 'roseau', 'hart', 'brooke', 'wilkes', 'vermont', 'rooks', 'harrison', 'costilla', 'sweet', 'howard', 'jeff', 'metcalfe', 'hoonah-angoon', 'wabaunsee', 'lowndes', 'perry', 'petroleum', 'todd', 'jasper', 'goshen', 'berkshire', 'jo', 'grundy', 'blount', 'smyth', 'gove', 'okaloosa', 'harris', 'jenkins', 'mcdowell', 'chemung', 'harford', 'dickenson', 'massac', 'cross', 'raleigh', 'autauga', 'lycoming', 'wright', 'owen', 'haakon', 'melanesia', 'pennsylvania', 'woodward', 'lyman', 'kauai', 'fond', 'maury', 'gulf', 'rockdale', 'clare', 'winchester', 'hocking', 'huntington', 'laclede', 'chippewa', 'fergus', 'dallas', 'haralson', 'robertson', 'tunica', 'midland', 'christian', 'australia', 'somervell', 'lehigh', 'accomack', 'cleburne', 'atchison', 'mobile', 'eastern asia', 'copiah', 'rockland', 'grimes', 'az', 'talbot', 'guanica', 'yuba', 'sangamon', 'central africa', 'scurry', 'spencer', 'schoolcraft', 'atascosa', 'chad', 'comal', 'harding', 'hampton', 'ashtabula', 'nj', 'cayuga', 'yadkin', 'comoros', 'coryell', 'chilton', 'albany', 'larue', 'shawnee', 'prairie', 'valdez-cordova', 'bartholomew', 'olmsted', 'divide', 'kern', 'shoshone', 'kosovo', 'loving', 'uintah', 'forsyth', 'greenup', 'rhea', 'dodge', 'edmunds', 'butler', 'treasure', 'breckinridge', 'jewell', 'price', 'issaquena', 'galax', 'garfield', 'kanabec', 'lampasas', 'noxubee', 'bandera', 'plymouth', 'weakley', 'dillingham', 'ashe', 'morgan', 'avoyelles', 'mauritius', 'newport', 'hillsborough', 'wharton', 'pershing', 'dane', 'europe', 'switzerland', 'elmore', 'rutherford', 'nantucket', 'nye', 'wabash', 'guilford', 'alaska', 'davis', 'hardy', 'mecosta', 'deuel', 'ross', 'camp', 'gaines', 'perquimans', 'korea north', 'saint vincent & the grenadines', 'bonner', 'kittitas', 'hunterdon', 'henderson', 'ne', 'central african rep', 'steuben', 'in', 'pueblo', 'gregory', 'chambers', 'simpson', 'northern europe', 'pondera', 'columbia', 'dickson', 'napa', 'imperial', 'bingham', 'beauregard', 'western europe', 'daviess', 'monona', 'northumberland', 'edwards', 'fannin', 'nolan', 'algeria', 'mono', 'attala', 'trimble', 'nome', 'paulding', 'sabine', 'bamberg', 'fort', 'hennepin', 'morocco', 'lamar', 'rogers', 'power', 'queen', 'deaf', 'moultrie', 'navarro', 'ness', 'portage', 'cuba', 'craighead', 'williams', 'ellis', 'sabana', 'concho', 'seneca', 'las', 'sitka', 'mccook', 'cheboygan', 'fremont', 'cumberland', 'reynolds', 'hardeman', 'holt', 'kenosha', 'hertford', 'libya', 'garrett', 'macedonia', 'umatilla', 'tippah', 'elk', 'geary', 'cullman', 'neosho', 'sanpete', 'pointe', 'camas', 'howell', 'briscoe', 'echols', 'bayfield', 'talladega', 'cache', 'chattooga', 'esmeralda', 'honolulu', 'bent', 'wirt', 'ocean', 'dent', 'tillman', 'florence', 'borden', 'taos', 'spain', 'northampton', 'champaign', 'parker', 'anchorage', 'mora', 'woodson', 'granville', 'gabon', 'cambria', 'henrico', 'salt', 'canyon', 'kitsap', 'scotland', 'roanoke', 'grainger', 'kent', 'kossuth', 'childress', 'morrow', 'luquillo', 'williamsburg', 'kyrgyzstan', 'kearny', 'clinton', 'sequatchie', 'gallatin', 'major', 'stevens', 'czech republic', 'yancey', 'vigo', 'liberia', 'alpine', 'troup', 'andrew', 'district of columbia', 'kenya', 'terry', 'juneau', 'wilbarger', 'walthall', 'hand', 'bennington', 'luce', 'shackelford', 'missoula', 'suriname', 'hot', 'roosevelt', 'hall', 'portsmouth', 'hungary', 'macon', 'new hampshire', 'carroll', 'manistee', 'rich', 'elliott', 'tishomingo', 'luzerne', 'falls', 'furnas', 'latvia', 'trumbull', 'los', 'belmont', 'izard', 'grays', 'bergen', 'billings', 'drew', 'hopkins', 'western asia', 'tulsa', 'beltrami', 'hidalgo', 'sonoma', 'del', 'wallowa', 'nauru', 'montrose', 'hopewell', 'hendricks', 'matagorda', 'lane', 'colonial', 'tajikistan', 'schuylkill', 'chase', 'bibb', 'juana', 'corson', 'buena', 'botswana', 'sheboygan', 'clinch', 'venezuela', 'moldova', 'apache', 'kalamazoo', 'alachua', 'black', 'iberia', 'conejos', 'utah', 'culpeper', 'culebra', 'pickens', 'burt', 'johnston', 'honduras', 'amherst', 'marin', 'collier', 'somalia', 'summit', 'glynn', 'perkins', 'id', 'fairfield', 'chittenden', 'chester', 'united kingdom', 'guernsey', 'pinal', 'lorain', 'cameroon', 'uruguay', 'hemphill', 'hunt', 'patillas', 'otsego', 'monterey', 'dominica', 'lonoke', 'mecklenburg', 'aleutians', 'pakistan', 'lynn', 'payette', 'lawrence', 'hi', 'carteret', 'cottle', 'otter', 'harrisonburg', 'slovakia', 'independence', 'pratt', 'maine', 'salem', 'posey', 'cavalier', 'lea', 'van', 'western', 'huerfano', 'pasquotank', 'hitchcock', 'south-eastern asia', 'marinette', 'chariton', 'gilpin', 'sharp', 'clearfield', 'hickory', 'union', 'ecuador', 'labette', 'owsley', 'wheeler', 'dearborn', 'rowan', 'routt', 'arapahoe', 'mercer', 'kenedy', 'schenectady', 'pima', 'burke', 'georgia', 'barbados', 'arizona', 'davidson', 'allegany', 'cassia', 'southeast', 'rockwall', 'gogebic', 'antrim', 'bottineau', 'presidio', 'danville', 'shawano', 'whitman', 'knott', 'mayes', 'kewaunee', 'caledonia', 'williamson', 'jessamine', 'philadelphia', 'broomfield', 'broward', 'costa rica', 'elbert', 'gwinnett', 'denmark', 'edgar', 'guinea', 'hays', 'south dakota', 'pepin', 'citrus', 'cayey', 'duplin', 'glades', 'crockett', 'essex', 'refugio', 'sterling', 'jamaica', 'castro', 'ms', 'andrews', 'campbell', 'thomas', 'bullitt', 'oswego', 'yuma', 'cabell', 'sudan', 'webster', 'ireland', 'denton', 'wagoner', 'donley', 'sutter', 'banks', 'pembina', 'stearns', 'cape verde', 'piscataquis', 'newaygo', 'swain', 'chile', 'manitowoc', 'kemper', 'overton', 'utuado', 'lenawee', 'new zealand', 'spalding', 'nance', 'petersburg', 'codington', 'tuscola', 'trego', 'lajas', 'augusta', 'gentry', 'galveston', 'montana', 'richmond', 'placer', 'harlan', 'austria', 'osage', 'amelia', 'trinidad & tobago', 'angola', 'pawnee', 'kerr', 'fairbanks', 'wise', 'terrell', 'fayette', 'cheshire', 'juncos', 'ks', 'charleston', 'shiawassee', 'calhoun', 'burlington', 'evangeline', 'mitchell', 'love', 'anne', 'burnett', 'milam', 'dubuque', 'clear', 'mountrail', 'oman', 'denver', 'jim', 'brantley', 'kings', 'le', 'lamb', 'naguabo', 'jersey', 'woodford', 'papua new guinea', 'kodiak', 'fillmore', 'tarrant', 'saguache', 'ogemaw', 'vance', 'claiborne', 'chouteau', 'meagher', 'tennessee', 'big', 'jordan', 'broome', 'pacific', 'comerio', 'fleming', 'nicholas', 'day', 'guatemala', 'uinta', 'united states', 'yamhill', 'stephenson', 'iosco', 'ben', 'north carolina', 'middlesex', 'twin', 'croatia', 'banner', 'twiggs', 'rush', 'sandoval', 'torrance', 'freeborn', 'nh', 'dallam', 'haskell', 'cass', 'ok', 'bulgaria', 'colleton', 'allegan', 'calaveras', 'chesterfield', 'phelps', 'pickaway', 'ward', 'jay', 'crane', 'emery', 'wibaux', 'coffee', 'hawkins', 'sumner', 'muskingum', 'mcleod', 'bienville', 'laurens', 'gem', 'eaton', 'ballard', 'zambia', 'frontier', 'yabucoa', 'cascade', 'colbert', 'hempstead', 'ventura', 'avery', 'humacao', 'ga', 'paraguay', 'okfuskee', 'james', 'gladwin', 'redwood', 'atlantic', 'loudoun', 'guayanilla', 'greenlee', 'dickens', 'coles', 'forrest', 'kimble', 'ripley', 'foster', 'little', 'edgefield', 'georgetown', 'fresno', 'menifee', 'wexford', 'iron', 'tonga', 'wabasha', 'will', 'faribault', 'hoke', 'cerro', 'buncombe', 'pontotoc', 'lemhi', 'wapello', 'carbon', 'keokuk', 'guam', 'crow', 'indonesia', 'gregg', 'sierra leone', 'iroquois', 'prentiss', 'michigan', 'red', 'flathead', 'woods', 'centre', 'bowman', 'story', 'vatican city', 'ochiltree', 'kankakee', 'carlton', 'sharkey', 'warrick', 'floyd', 'cattaraugus', 'buchanan', 'bon', 'durham', 'iowa', 'austin', 'lares', 'palo', 'maunabo', 'houston', 'adjuntas', 'dorchester', 'tate', 'caddo', 'gambia', 'worcester', 'crisp', 'nuckolls', 'yankton', 'lapeer', 'germany', 'ada', 'emmons', 'winston', 'goliad', 'green', 'greensville', 'lander', 'aroostook', 'oliver', 'modoc', 'vietnam', 'henry', 'darlington', 'itawamba', 'cimarron', 'piatt', 'tulare', 'mckean', 'brule', 'schoharie', 'solomon islands', 'turner', 'frederick', 'smith', 'ste.', 'rockbridge', 'reagan', 'swains', 'breathitt', 'chesapeake', 'oxford', 'powhatan', 'passaic', 'kingsbury', 'brunei', 'stark', 'throckmorton', 'sao tome & principe', 'kimball', 'mn', 'acadia', 'stonewall', 'pottawattamie', 'fauquier', 'wythe', 'collingsworth', 'sublette', 'young', 'mesa', 'assumption', 'singapore', 'carolina', 'neshoba', 'lexington', 'wallace', 'caldwell', 'pearl', 'greeley', 'columbiana', 'outagamie', 'treutlen', 'kennebec', 'mckinley', 'box', 'volusia', 'sioux', 'norfolk', 'burma', 'snohomish', 'clallam', 'mingo', 'valencia', 'laos', 'branch', 'cowlitz', 'congo', 'mcdonald', 'serbia', 'wa', 'pulaski', 'daniels', 'liechtenstein', 'atkinson', 'leelanau', 'judith', 'fiji', 'panama', 'dyer', 'sanders', 'pittsylvania', 'ny', 'mi', 'strafford', 'tipton', 'moca', 'tallahatchie', 'musselshell', 'aguas', 'eau', 'poweshiek', 'blanco', 'tucker', 'midway', 'northern africa', 'mills', 'bhutan', 'presque', 'boyle', 'mccurtain', 'webb', 'connecticut', 'norton', 'wyoming', 'riverside', 'bennett', 'kendall', 'klickitat', 'faulkner', 'rock', 'hawaii', 'mayaguez', 'ky', 'knox', 'illinois', 'hampden', 'dewey', 'rawlins', 'barranquitas', 'buffalo', 'osceola', 'haines', 'rensselaer', 'zapata', 'djibouti', 'west', 'greenville', 'bay', 'mason', 'unicoi', 'coshocton', 'united arab emirates', 'harvey', 'callaway', 'shenandoah', 'calumet', 'susquehanna', 'bracken', 'alexander', 'alcorn', 'vilas', 'mclennan', 'ukraine', 'arkansas', 'travis', 'dawson', 'jones', 'caribou', 'newton', 'finland', 'traill', 'mendocino', 'potter', 'lucas', 'ar', 'queens', 'dimmit', 'russian federation', 'sherman', 'mcnairy', 'baca', 'pasco', 'sussex', 'hutchinson', 'dona', 'tangipahoa', 'brazos', 'muhlenberg', 'new jersey', 'juab', 'pitkin', 'crook', 'lesotho', 'whitfield', 'tyler', 'fountain', 'owyhee', 'levy', 'laramie', 'gila', 'bureau', 'garland', 'tn', 'wi', 'cobb', 'morris', 'anderson', 'cameron', 'bremer', 'kuwait', 'murray', 'parke', 'cherry', 'cambodia', 'kenton', 'boise', 'bethel', 'new', 'door', 'brown', 'west virginia', 'bonneville', 'senegal', 'poinsett', 'saunders', 'wyandotte', 'kidder', 'palau', 'ida', 'ethiopia', 'marquette', 'limestone', 'roscommon', 'ca', 'bell', 'mackinac', 'virginia', 'montague', 'wv', 'cocke', 'washakie', 'atoka', 'desoto', 'new york', 'navajo', 'stoddard', 'maldives', 'bexar', 'moniteau', 'me', 'sunflower', 'south carolina', 'boyd', 'florida', 'barnes', 'charlotte', 'klamath', 'namibia', 'eastern europe', 'poquoson', 'lasalle', 'bannock', 'china', 'huron', 'parmer', 'bastrop', 'dundy', 'gratiot', 'martinsville', 'burnet', 'kansas', 'allamakee', 'uvalde', 'reno', 'northern asia', 'hamilton', 'beaufort', 'glenn', 'mali', 'turkmenistan', 'netherlands', 'burleigh', 'montour', 'bangladesh', 'berrien', 'cabo', 'randall', 'kleberg', 'starke', 'washtenaw', 'dutchess', 'calcasieu', 'uganda', 'beaver', 'antelope', 'jerauld', 'swisher', 'pender', 'osborne', 'radford', 'edmonson', 'mccreary', 'plaquemines', 'crenshaw', 'mcclain', 'catawba', 'miami-dade', 'worth', 'marlboro', 'bahamas', 'heard', 'or', 'horry', 'eagle', 'glasscock', 'cheyenne', 'coconino', 'prowers', 'bleckley', 'lamoille', 'pendleton', 'pamlico', 'shannon', 'yellowstone', 'york', 'goochland', 'co', 'sawyer', 'lyon', 'canada', 'adams', 'madera', 'district', 'mcpherson', 'moffat', 'wilkin', 'oktibbeha', 'bee', 'belize', 'roger', 'daggett', 'wolfe', 'bayamon', 'nepal', 'sc', 'hyde', 'ghana', 'trinity', 'watonwan', 'stanly', 'nueces', 'latimer', 'hartford', 'sullivan', 'contra', 'wakulla', 'myanmar', 'matanuska-susitna', 'polk', 'appanoose', 'sweden', 'amite', 'sd', 'cedar', 'androscoggin', 'erath', 'madagascar', 'jackson', 'westmoreland', 'manassas', 'reeves', 'clearwater', 'isabela', 'upson', 'indian', 'east', 'rhode island', 'pettis', 'waynesboro', 'hernando', 'antigua & deps', 'park', 'yellow', 'wyandot', 'marshall islands', 'maryland', 'skagit', 'wy', 'yell', 'rains', 'madison', 'bacon', 'oceana', 'baraga', 'live', 'socorro', 'mahnomen', 'asia', 'yalobusha', 'angelina', 'northwest', 'jayuya', 'eastern africa', 'barton', 'walworth', 'noble', 'rice', 'israel', 'tillamook', 'cheatham', 'yates', 'alabama', 'republic', 'central america', 'newberry', 'laporte', 'hanover', 'winneshiek', 'guyana']

    # arr too big to store in code
    with open(config.categoricals_file, "r") as f:
        similar_word_map = json.load(f)
        similar_word_map = {int(k): v for k, v in similar_word_map.items()}

    # Generate Graphs
    generation_helper(config.ctype, n=config.generate_n_imgs, all_charts=False)
