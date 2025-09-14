def overview(df , count = True , just_null = False , just_object = False):
    import pandas as pd
    
    print(f"Number of Rows    = {df.shape[0]:,}")
    print(f"Number of Columns = {df.shape[1]:,}")

    table = pd.DataFrame([])
    table["Missing Percentage"] = (df.isna().sum().to_frame() / df.shape[0] * 100).round(2)
    table["Missing Count"] = df.isna().sum()
    table["Data Type"] = df.dtypes
    table.index.name = "Column"
    
    table = table.sort_values(by = ["Missing Count" , "Data Type"] , ascending = [False , False])
    
    if just_object == True:
        table = table.loc[table["Data Type"] == "object"]
    if just_null == True:
        table = table.loc[table["Missing Count"] > 0] 
    if count == False:
        table = table.drop("Missing Count" , axis = 1)
    return table

 
def get_corrolation_heatmap(df , dpi = 300 , save = False):
    import seaborn as sbn
    import numpy as np
    import matplotlib.pyplot as plt
    
    df = df.select_dtypes(include = [np.number]).dropna()
    object_columns = df.dtypes[df.dtypes == "object"].index.values
    if len(object_columns) > 0:
        df = df.drop(columns = object_columns)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool)) # to avoid repetition
    fig , ax = plt.subplots(1,1)
    fig.set_size_inches(20 , 18)
    sbn.heatmap(corr,
                ax = ax,
                mask = mask,
                cmap='coolwarm',
                annot=True,
                yticklabels = df.columns,
                xticklabels = df.columns)
    ax.set_title('Correlation Matrix')
    
    if save == True:
        fig.savefig("corrolation_heatmap.png" , dpi = dpi)
    plt.show()
    
def do_JB_test(A , alpha = 0.05):
    import scipy.stats as st
    n = A.shape[0]
    Z = (A - A.mean()) / A.std(ddof = 1)
    S_hat = (1 / n) * (Z ** 3).sum()
    K_hat = (1 / n) * (Z ** 4).sum()
    JB = n/6 * (S_hat**2 + ((K_hat-3)**2)/4)
    Chi2_alpha_2 = st.chi2.ppf(1-alpha,df=2)
    p_value = 1 - st.chi2.cdf(x = JB , df = 2)
    print('H0: Data IS normally distributed.\nH1: Data is NOT normally distributed.')
    print(50 * "-")
    print("S_hat =" , S_hat.round(4))
    print("K_hat =" , K_hat.round(4))
    print("JB =" , JB.round(4))
    print("Chi2_0.95_2 =" , Chi2_alpha_2.round(4))
    print(50 * "-")
    print("P-value =" , f"{p_value.round(4)*100}%")
    print(50 * "-")
    if JB <= Chi2_alpha_2:
        print(f"Accept H0; the distribution at α = {alpha} IS normal.")
    else:
        print(f"Reject H0; the distribution at α = {alpha} is NOT normal.")

def get_var_name(var):
    import inspect
    frame = inspect.currentframe().f_back
    for name, val in frame.f_locals.items():
        if val is var:
            return name
        
def get_histplot(data , save = False , dpi = 300 , data_name = "Data"):
    import seaborn as sbn
    import matplotlib.pyplot as plt
    import pandas as pd
    plot_title = f"Distribution of {data_name}"
    data = pd.Series(data)
    fig , ax = plt.subplots(figsize=(10, 6))
    sbn.histplot(data , ax=ax, color="mediumseagreen", kde=True , edgecolor="white")
    ax.axvline(data.mean() , color='black', linestyle='--', linewidth=1.5 , label = "Mean")
    ax.axvline(data.median() , color='red', linestyle='--', linewidth=1.5 , label = "Mean")
    ax.set_title(plot_title , fontsize=16 , fontweight='bold')
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save == True:
        fig.savefig(plot_title + "png" , dpi = dpi)
    plt.show()
    
def get_global_var_name(var , namespace = globals()):
    return [name for name , val in namespace.items() if val is var][0]

def get_local_var_name(var , namespace = locals()):
    return [name for name , val in namespace.items() if val is var][0]

def get_size_MB(var , Return = False , local = False):
    from sys import getsizeof
    usage_B = getsizeof(var)
    usage_MB = round(usage_B * 2 ** (-20) , 2)
    if local == True:
        var_name = get_local_var_name(var)
    else:
        var_name = get_global_var_name(var)
    print(f"Memory usage of \"{var_name}\" = {usage_MB} MB.")
    if Return == True:
        return usage_MB
    
def calculate_vif(df):
    import numpy as np
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = df.select_dtypes(include = [np.number]).dropna()
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values , i) for i in range(X.shape[1])]
    return vif_data

def get_histplot_all(df , save = False , dpi = 600 , data_name = "Data"):
    import seaborn as sbn
    import matplotlib.pyplot as plt
    import numpy as np
    numeric_df = df.select_dtypes(include = [np.number]).dropna()
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_df.columns) / n_cols))
    fig , axes = plt.subplots(n_rows , n_cols , figsize = (n_cols*5 , n_rows*4))
    axes = axes.flatten()
    for i , col in enumerate(numeric_df.columns):
        sbn.histplot(numeric_df[col] , ax = axes[i] , color = "mediumseagreen" , kde = True , edgecolor = "white")
        axes[i].axvline(numeric_df[col].mean() , color = 'black' , linestyle = '--' , linewidth = 1.5 , label = "Mean")
        axes[i].axvline(numeric_df[col].median() , color = 'red' , linestyle = '--' , linewidth = 1.5 , label = "Median")
        axes[i].set_title(f"Distribution of {col}" , fontsize = 12 , fontweight = 'bold')
        axes[i].set_xlabel("Value" , fontsize = 10)
        axes[i].set_ylabel("Frequency" , fontsize = 10)
        axes[i].legend()
        axes[i].grid(True , linestyle = '--' , alpha = 0.5)
    for j in range(i+1 , len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    if save == True:
        fig.savefig(f"Distribution of {data_name}.png" , dpi = dpi)
    plt.show()
    
def get_plot_knn_boundaries(X , Y , K , save = False):
    from sklearn.neighbors import KNeighborsClassifier
    from matplotlib.colors import ListedColormap
    from warnings import filterwarnings
    import matplotlib.pyplot as plt
    import seaborn as sbn
    import numpy as np
    
    filterwarnings("ignore")
    col1 , col2 = X.columns[0] , X.columns[1]
    X.columns = ["X1" , "X2"]
    clf = KNeighborsClassifier(n_neighbors = K)
    clf.fit(X , Y)
    x_min , x_max = X["X1"].min() - 1 , X["X1"].max() + 1
    y_min , y_max = X["X2"].min() - 1 , X["X2"].max() + 1
    xx , yy = np.meshgrid(np.linspace(x_min , x_max , 500) , np.linspace(y_min , y_max , 500))
    Z = clf.predict(np.c_[xx.ravel() , yy.ravel()]).reshape(xx.shape)
    unique_labels = np.unique(Y)
    base_colors = ['red' , 'green' , 'blue']
    extra_colors = sbn.color_palette("husl" , len(unique_labels) - 3) if len(unique_labels) > 3 else []
    colors = base_colors[:len(unique_labels)] + [sbn.utils.hex_to_rgb(sbn.utils.rgb2hex(c)) for c in extra_colors]
    cmap_light = ListedColormap([sbn.desaturate(c , 0.6) for c in colors])
    cmap_bold = ListedColormap(colors)
    plt.figure(figsize = (8 , 6))
    plt.contourf(xx , yy , Z , cmap = cmap_light , alpha = 0.8)
    plt.contour(xx , yy , Z , colors = "black" , linewidths = 0.1)
    plt.scatter(X["X1"] , X["X2"] , c = Y , cmap = cmap_bold , edgecolor = 'k' , s = 15)
    plt.title(f"Decision Boundaries for KNN, K = {K}")
    plt.xlabel(col1)
    plt.ylabel(col2)
    if save == True:
        plt.savefig("KNN_Decision_Boundary.png" , dpi = 600)
    plt.show()
    
def reshaper(text: str) -> str:
    import arabic_reshaper
    from bidi.algorithm import get_display
    """
    Prepares Persian/Arabic text for proper display in matplotlib.

    Parameters:
        text (str): The input Persian/Arabic string.

    Returns:
        str: The reshaped and bidi-corrected string.
    """
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

def do_Shapiro_test(A , alpha = 0.05):
    import scipy.stats as st
    stat , p_value = st.shapiro(A.dropna())
    print('H0: Data IS normally distributed.\nH1: Data is NOT normally distributed.')
    print(50 * "-")
    print("Test Statistic =" , stat.round(4))
    print("P-value =" , f"{p_value.round(4)*100}%")
    print(50 * "-")
    if p_value > alpha:
        print(f"Accept H0; the distribution at α = {alpha} IS normal.")
    else:
        print(f"Reject H0; the distribution at α = {alpha} is NOT normal.")
        
def get_boxplot(data , save = False , dpi = 300 , data_name = "Data"):
    import seaborn as sbn
    import matplotlib.pyplot as plt
    import pandas as pd
    plot_title = f"Boxplot of {data_name}"
    data = pd.Series(data)
    fig , ax = plt.subplots(figsize=(10, 6))
    sbn.boxplot(x=data , ax=ax , color="mediumseagreen")
    ax.set_title(plot_title , fontsize=16 , fontweight='bold')
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("") 
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save == True:
        fig.savefig(plot_title + ".png" , dpi = dpi)
    plt.show()
    
def get_boxplot_all(df , save = False , dpi = 300 , data_name = "Data"):
    import seaborn as sbn
    import matplotlib.pyplot as plt
    import numpy as np
    numeric_df = df.select_dtypes(include = [np.number]).dropna()
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_df.columns) / n_cols))
    fig , axes = plt.subplots(n_rows , n_cols , figsize = (n_cols*5 , n_rows*4))
    axes = axes.flatten()
    for i , col in enumerate(numeric_df.columns):
        sbn.boxplot(x=numeric_df[col] , ax = axes[i] , color = "mediumseagreen")
        axes[i].set_title(f"Boxplot of {col}" , fontsize = 12 , fontweight = 'bold')
        axes[i].set_xlabel("Value" , fontsize = 10)
        axes[i].set_ylabel("") 
        axes[i].grid(True , linestyle = '--' , alpha = 0.5)
    for j in range(i+1 , len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    if save == True:
        fig.savefig(f"Boxplots of {data_name}.png" , dpi = dpi)
    plt.show()
