import matplotlib.pyplot as plt
import pandas as pd

def generate_boxplot(dataframe, x_tick_labels=None, y_label=None, title=None, figsize=(10, 7), 
                     save_path=None, show_plot=True):
    """
    Generate a boxplot of accuracy metrics. The function takes a dataframe and create box plots. The number of box plot is determined by the number of colums and each 
    box plot is generated based on the data provided in each columns. 

    Args:
        dataframe (pd.DataFrame): DataFrame containing data, where rows correspond to different data
                                  and columns correspond to different models.
        x_tick_labels (list, optional): Custom tick labels for the x-axis.
        y_label (str, optional): Label for the y-axis.
        title (str, optional): Title for the plot.
        figsize (tuple, optional): Figure size (width, height).
        save_path (str, optional): Path to save the generated image. If None, the image will not be saved.
        show_plot (bool, optional): Whether to display the plot (True) or not (False).
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle missing x_tick_labels
    if x_tick_labels is None:
        x_tick_labels = dataframe.columns.tolist()
    
    plt.boxplot(dataframe.values)
    
    ax.set_xticklabels(x_tick_labels, fontsize=20)
    
    if title:
        plt.title(title, fontsize=20)
    
    plt.ylabel(y_label, fontsize=20)
    plt.tick_params(axis='both', which='major', length=10, width=2, labelsize=20)
    
    if save_path:
        plt.savefig(save_path)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
