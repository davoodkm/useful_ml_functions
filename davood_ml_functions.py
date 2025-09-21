def overview(df , count = True , just_null = False , just_object = False):
    import pandas as pd
    
    print(f"  Number of Rows  = {df.shape[0]:,}")
    print(f"Number of Columns = {df.shape[1]:,}")

    table = pd.DataFrame([])
    table["Missing Percentage"] = (df.isna().sum().to_frame() / df.shape[0] * 100).round(1)
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
    else:
        table["Missing Count"] = table["Missing Count"].apply(lambda x: f"{x:,}")
    return table

 
def get_corrolation_heatmap(df , target = "", dpi = 300 , save = False):
    import seaborn as sbn
    import numpy as np
    import matplotlib.pyplot as plt
    
    df = df.select_dtypes(include = [np.number , np.bool  , int])
    object_columns = df.dtypes[df.dtypes == "object"].index.values
    if len(object_columns) > 0:
        df = df.drop(columns = object_columns)
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(float)
    if target != "":
        df = df[[c for c in df.columns if c != target] + [target]]
    corr = df.corr() * 100
    mask = np.triu(np.ones_like(corr, dtype=bool)) # to avoid repetition
    fig , ax = plt.subplots(1,1)
    fig.set_size_inches(20 , 18)
    sbn.heatmap(corr,
                ax = ax,
                mask = mask,
                cmap='coolwarm',
                annot = True,
                annot_kws = {"size": 14},
                fmt = ".1f",
                yticklabels = df.columns,
                xticklabels = df.columns)
    ax.set_title("Correlation Matrix(%)" , fontsize = 20 , fontweight = "bold")
    ax.tick_params(axis = 'x' , labelsize = 11)
    ax.tick_params(axis = 'y' , labelsize = 11)
    if save == True:
        fig.savefig("corrolation_heatmap.png" , dpi = dpi)
    plt.show()
    
    if save == True:
        fig.savefig("corrolation_heatmap.png" , dpi = dpi)
    plt.show()
    
def get_corrolation_matrix(df , target = "y" , vector = True):
    import pandas as pd
    import numpy as np
    
    df = df.select_dtypes(include = [np.number , np.bool , int])
    object_columns = df.dtypes[df.dtypes == "object"].index.values
    if len(object_columns) > 0:
        df = df.drop(columns = object_columns)
    
    corr = df.corr().drop(target)
    corr = corr[[target] + [c for c in corr.columns if c != target]]
    corr.index.name = ""
    
    corr = corr.reindex(corr[target].abs().sort_values(ascending = False).index)
    for col in corr.columns:
        corr[col] = (corr[col] * 100).apply(lambda x: f"{round(x , 1):.1f}%")
    if vector == True:
        corr = corr[[ target]]
    return corr
    
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
    ax.axvline(data.median() , color='red', linestyle='--', linewidth=1.5 , label = "Median")
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
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values , i) for i in range(X.shape[1])]
    vif = vif.sort_values(by = "VIF" , ascending = False)
    vif["VIF"] = vif["VIF"].apply(lambda x: f"{round(x , 2):.2f}")
    
    vif.index.name = ""
    vif = vif.reset_index(drop = True)
    vif.index += 1

    return vif

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
    filterwarnings("ignore")
    import matplotlib.pyplot as plt
    import seaborn as sbn
    import numpy as np
    
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
    
def feature_transformer(df , target = "y" , select = 3 , apply = False):
    import pandas as pd
    import numpy as np
    from scipy.stats import boxcox
    from sklearn.preprocessing import PowerTransformer
    
    num_cols = df.select_dtypes(include = np.number).columns.tolist()
    if target in num_cols: num_cols.remove(target)
    num_cols = [col for col in num_cols if df[col].nunique() > 2]
    transformations = ["original" , "log" , "sqrt" , "square" , "cube" , "boxcox" , "yeojohnson"]
    transformed_data = {}
    for col in num_cols:
        transformed_data[col] = {}
        col_values = df[col].values
        transformed_data[col]["original"] = col_values
        if np.all(col_values > 0):
            transformed_data[col]["log"] = np.log(col_values)
            transformed_data[col]["sqrt"] = np.sqrt(col_values)
            transformed_data[col]["boxcox"] = boxcox(col_values)[0]
        transformed_data[col]["square"] = np.power(col_values , 2)
        transformed_data[col]["cube"] = np.power(col_values , 3)
        pt = PowerTransformer(method = "yeo-johnson")
        transformed_data[col]["yeojohnson"] = pt.fit_transform(col_values.reshape(-1 , 1)).flatten()
    result_rows = []
    for col in num_cols:
        correlations = {}
        for t_name , t_values in transformed_data[col].items():
            correlations[t_name] = abs(np.corrcoef(t_values , df[target].values)[0 , 1])
        best_trans = sorted(correlations.items() , key = lambda x: x[1] , reverse = True)[:select]
        for name , corr in best_trans:
            result_rows.append([f"{col} ({name})" , corr])
    result = pd.DataFrame(result_rows , columns = ["Column Name" , "Corrolation"]).sort_values(by = "Corrolation" , ascending = False)
    result["Corrolation(%)"] = result["Corrolation"].apply(lambda x: f"{round(x*100 , 2):.2f}%")
    result = result.drop(columns = "Corrolation")
    if apply:
        for col in num_cols:
            best_name = max(transformed_data[col].items() , key = lambda x: abs(np.corrcoef(x[1] , df[target].values)[0 , 1]))[0]
            df[f"{col}_best"] = transformed_data[col][best_name]
        df = df[[target] + [f"{col}_best" for col in num_cols] + num_cols]
        return df
    return result

def simple_transformer(df , target = "y" , apply = False):
    import pandas as pd
    import numpy as np
    
    num_cols = df.select_dtypes(include = np.number).columns.tolist()
    if target in num_cols: num_cols.remove(target)
    num_cols = [col for col in num_cols if df[col].nunique() > 2]
    transformed_data = {}
    for col in num_cols:
        transformed_data[col] = {}
        col_values = df[col].values
        transformed_data[col]["original"] = col_values
        if np.all(col_values > 0):
            transformed_data[col]["log"] = np.log(col_values)
        transformed_data[col]["square"] = np.power(col_values , 2)
        transformed_data[col]["cube"] = np.power(col_values , 3)
    result_rows = []
    for col in num_cols:
        correlations = {}
        for t_name , t_values in transformed_data[col].items():
            correlations[t_name] = abs(np.corrcoef(t_values , df[target].values)[0 , 1])
        for name , corr in correlations.items():
            result_rows.append([f"{col} ({name})" , corr])
    result = pd.DataFrame(result_rows , columns = ["Column Name" , "Corrolation"]).sort_values(by = "Corrolation" , ascending = False)
    result["Corrolation(%)"] = result["Corrolation"].apply(lambda x: f"{round(x*100 , 2):.2f}%")
    result = result.drop(columns = "Corrolation")
    if apply:
        for col in num_cols:
            best_name = max(transformed_data[col].items() , key = lambda x: abs(np.corrcoef(x[1] , df[target].values)[0 , 1]))[0]
            df[f"{col}_best"] = transformed_data[col][best_name]
        df = df[[target] + [f"{col}_best" for col in num_cols] + num_cols]
        return df
    return result

def get_classification_report(y_real , y_pred):
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score
    import matplotlib.pyplot as plt
    import seaborn as sbn
    
    cm = confusion_matrix(y_real , y_pred)
    accuracy = accuracy_score(y_real , y_pred)
    f1 = f1_score(y_real , y_pred)
    recall = recall_score(y_real , y_pred)
    precision = precision_score(y_real , y_pred)
    npv = cm[0 , 0] / (cm[0 , 0] + cm[1 , 0])
    specificity = cm[0 , 0] / (cm[0 , 0] + cm[0 , 1])


    print(f"Accuracy = {accuracy*100:.2f}%")
    print(f"F1-score = {f1*100:.2f}%")
    print(f"Recall = {recall*100:.2f}%")
    print(f"Precision = {precision*100:.2f}%")
    print(f"NPV = {npv*100:.2f}%")
    print(f"Specificity = {specificity*100:.2f}%")
    
    fig , ax = plt.subplots(figsize = (6 , 5))
    sbn.heatmap(cm , annot = True , fmt = ',' , cmap = 'viridis' ,
                linewidths = 0.65 , linecolor = 'black' , cbar = False,
                ax = ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.show()
