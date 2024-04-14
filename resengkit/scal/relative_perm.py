import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from scipy.optimize import curve_fit
from  . import rel_perm_helper
from itertools import combinations
import seaborn as sns   
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import ceil


class Normalizer:
    def __init__(self, samples_curves , samples_data):
        """
        Initialize the Normalizer class.

        Parameters:
        - samples_curves (pd.DataFrame): Dataframe containing the curves data.
        - samples_data (pd.DataFrame): Dataframe containing the samples data.
        """
        self.samples_cruves = samples_curves
        self.samples_data = samples_data 
        
        self.process_data()
        
        
    def sample_parameter_correlation_summary(self):
        """
        Show summary of the samples and the relationships between sample parameters
        """
        features = self.samples_data.select_dtypes(include=['float64']).columns

        # Create a figure and axes with matplotlib
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8), dpi=100)
        
        fig.suptitle("Samples Parameters correlation Summary", fontsize=18, fontweight='bold', color='#17202A')
        # Create a heatmap
        ax[0].set_title("Non-Linear Relationship `spearman ` Cooffecient", fontsize=10, fontweight='bold', color='#333')
        sns.heatmap(self.samples_data[features].corr(method ="spearman"), annot=True, ax=ax[0])

        # Create a heatmap
        ax[1].set_title("Linear Relationship `Pearson` Cooffecient", fontsize=10, fontweight='bold', color='#333')
        sns.heatmap(self.samples_data[features].corr(), annot=True, ax=ax[1])

        return fig
    
    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def plot_fit(self,df , x , y , ax):
        model = LinearRegression()
        model.fit(df[x].to_numpy().reshape(-1,1), df[y])
        y_pred = model.predict(df[x].to_numpy().reshape(-1,1))
        
        # add the equation between the two variables to the plot as text
        coef = model.coef_[0]
        intercept = model.intercept_
        line_eq = f'y = {coef:.2f} * X + {intercept:.2f}'
        
        
        ax.scatter(df[x], df[y])
        ax.plot(df[x], y_pred, color='red')
        ax.set_title(f'{x} vs {y}',fontsize=10, fontweight='bold', color='#333')
        ax.set(xlabel=x, ylabel=y )
        ax.grid(True)
        ax.text(0.05, 0.95, f'R2: {r2_score(df[y], y_pred):.2f}\nRMSE: {self.rmse(df[y], y_pred):.2f}\n{line_eq}', 
                transform=ax.transAxes, verticalalignment='top')
        
    
    def sample_parameter_relationships(self):
        """
        Show summary of the samples and the relationships between sample parameters
        """
        features = self.samples_data.select_dtypes(include=['float64']).columns
        d = self.samples_data[features]
        pairs = list(combinations(features, 2))
        max_cols = 4
        rows = len(pairs)// max_cols + 1
        fig, axs = plt.subplots(nrows=rows, ncols=max_cols, figsize=(15, 15), dpi=200)
        fig.suptitle("Samples Parameters Relationships", fontsize=18, fontweight='bold', color='#17202A')
        for i , pair in enumerate(pairs):
            self.plot_fit(d, pair[0], pair[1], axs[i//max_cols, i%max_cols])
            
        fig.tight_layout(pad=1.8)
        
        return fig 

        
    def process_data(self) -> None:
        """
        Process the data by converting percentage values to numbers.
        """
        # get the unique samples names
        self.samples = self.samples_data.iloc[:,0].unique()
        
        # convert the the percentange into numbers
        self.samples_cruves["sw"] = (self.samples_cruves["sw"] / 100 ).round(3)
        self.samples_data["swi"] = (self.samples_data["swi"] / 100 ).round(3)
        self.samples_data["sor"] = (self.samples_data["sor"] / 100 ).round(3)
        self.samples_data["phi"] = (self.samples_data["phi"] / 100 ).round(3)
        
    def correct_curves_for_base(self, old_base : str , new_base :str) -> pd.DataFrame:
        """
        Correct the curves data for a new base.

        Parameters:
        - old_base (str): The old base parameter.
        - new_base (str): The new base parameter.

        Returns:
        - corrected_data (pd.DataFrame): The corrected curves data.
        """
        self.corrected_data = self.samples_cruves.copy()
        
        for sample in self.samples:
            
            # define the filter to get only the sample data in both dataframes
            mask_curves = self.corrected_data["sample"] == sample
            mask_parameters = self.samples_data["sample"] == sample
            
            # apply the changes to the new dataframe
            base_correction = self.samples_data.loc[mask_parameters, old_base].values[0] / self.samples_data.loc[mask_parameters, new_base].values[0]
            self.corrected_data.loc[mask_curves , "kro"] = self.corrected_data.loc[mask_curves , "kro"] * base_correction
            self.corrected_data.loc[mask_curves , "krw"] = self.corrected_data.loc[mask_curves , "krw"] * base_correction
            
        return self.corrected_data
    
    def normalize(self, corrected_data : pd.DataFrame, sw_col : str , swc_col :str , sor_col :str , kro_col :str , krw_col :str) -> pd.DataFrame:
        """
        Normalize the corrected data.

        Parameters:
        - corrected_data (pd.DataFrame): The corrected curves data.
        - sw_col (str): Column name for the sw values.
        - swc_col (str): Column name for the swc values.
        - sor_col (str): Column name for the sor values.
        - kro_col (str): Column name for the kro values.
        - krw_col (str): Column name for the krw values.

        Returns:
        - normalized_data (pd.DataFrame): The normalized curves data.
        """
        normalized_data = corrected_data.copy()
        
        for sample in self.samples:
            
            # apply filter to get the sample data
            mask_curves = normalized_data["sample"] == sample
            mask_parameters = self.samples_data["sample"] == sample
            
            # get the sample parameters
            swc = self.samples_data.loc[mask_parameters, swc_col].values[0]
            sor = self.samples_data.loc[mask_parameters, sor_col].values[0]
            
            # normalize the sw values
            sw = normalized_data.loc[mask_curves , sw_col]
            sw_corrected =  (sw - swc) / (1 - swc - sor)
            
            normalized_data.loc[mask_curves , sw_col] = sw_corrected
            normalized_data.loc[mask_curves , kro_col] = normalized_data.loc[mask_curves , kro_col] / normalized_data.loc[mask_curves, kro_col].max()
            normalized_data.loc[mask_curves , krw_col] = normalized_data.loc[mask_curves , krw_col] / normalized_data.loc[mask_curves, krw_col].max()
            
        return normalized_data.round(3)
    
        
class RelPerm:
    def __init__(self, data : pd.DataFrame , samples_col : str , swn_col : str, kron_col : str, krwn_col : str):
        """
        Initialize the RelPerm class.

        Parameters:
        - data (pd.DataFrame): Dataframe containing the data.
        - samples_col (str): Column name for the samples.
        - swn_col (str): Column name for the swn values.
        - kron_col (str): Column name for the kron values.
        - krwn_col (str): Column name for the krwn values.
        """
        self.data = data
        self.swn_col = swn_col
        self.kron_col = kron_col
        self.krwn_col = krwn_col
        self.samples_col = samples_col
        
        # setting the protected attributes
        self._oil_fitted_n = None 
        self._water_fitted_n = None 
        
        # process the data 
        self.data_process()
        
    def data_process(self) -> None :
        """
        Process the data by adding the son column and getting the unique samples.
        
        """
        self.data["Son"] = 1-self.data[self.swn_col]
        self.samples = self.data[self.samples_col].unique()
        
        # This is very important for the fitting to work
        self.data = self.data.round(2)
        
        
    def plot_samples(self):
        """
        Plot the samples data.

        Returns:
        - fig (matplotlib.figure.Figure): The figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6),dpi=100)

        for sample in self.samples:
        # plot swn in the x axis and kron,Krwn in the y axis
            x = self.data[self.data[self.samples_col] == sample][self.swn_col]
            kro = self.data[self.data[self.samples_col] == sample][self.kron_col]
            krw = self.data[self.data[self.samples_col] == sample][self.krwn_col]
            ax.scatter(x , kro , label = f'kron-{sample}',s=5)
            ax.scatter(x , krw , label = f'krwn-{sample}',s=5)
          
            
        ax.set_xlabel('swn',fontsize=13, fontweight='medium', color='#17202A')
        ax.set_ylabel('kron, krwn',fontsize=13, fontweight='medium', color='#17202A') 
        # add a title with font size 18 and medium weight and padding and color
        ax.set_title('kron, krwn vs swn', fontsize=18, fontweight='bold', color='#17202A',pad=20)
            
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        return (fig , ax)
    
    def fit_general_trend(self):
        """
        Fit the general trend of the data using cory Funciton.

        Returns:
        - fig (matplotlib.figure.Figure): The figure object.
        - results (dict): Dictionary containing the fitting results.
        """ 
        simple_cory = lambda x, n: x**n
        # define the fiting parameters for oil and water
        swn = self.data[self.swn_col]
        son = self.data["Son"]
        kron = self.data[self.kron_col]
        krwn = self.data[self.krwn_col]
        
        results = rel_perm_helper.fit_simple_cory(swn, son, kron, krwn)
        oil_n = results["N Oil"]
        water_n = results["N Water"]
        
        # setting the protected attributes
        self._oil_fitted_n = oil_n
        self._water_fitted_n = water_n
        
        
        # get the curves for the fit
        swn_curve = np.linspace(0,1,100)
        kron_curve = simple_cory(1-swn_curve, oil_n)
        krwn_curve = simple_cory(swn_curve, water_n)
        
        fig , ax = self.plot_samples()
            
        # plot the fitted lines
        ax.plot(swn_curve, kron_curve, label = f'kron-fit')
        ax.plot(swn_curve, krwn_curve, label = f'krwn-fit')
        
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        return   fig, results
    
    def create_curves_for_samples(self) :
        """
        Create curves for each sample.

        Returns:
        - kron_df (pd.DataFrame): Dataframe containing the kron curves for each sample.
        - krwn_df (pd.DataFrame): Dataframe containing the krwn curves for each sample.
        - swn_avgeraging (np.ndarray): Array of swn values used for averaging.
        """
        # first get fit a line to all the data, samples 
        total_params_oil = []
        total_params_water = []
        
        simple_cory = lambda x, n: x**n
        
        for sample in self.samples:
            dd = self.data[self.data[self.samples_col] == sample]
            sample_oil_params, _ = curve_fit(simple_cory, dd["Son"], dd[self.kron_col])
            sample_water_params, _ = curve_fit(simple_cory, dd[self.swn_col], dd[self.krwn_col])
            total_params_oil.append(sample_oil_params)
            total_params_water.append(sample_water_params)
            
            
        # after getting the parameters for all the samples , we can create the curves
        swn_avgeraging = np.arange(0,1.1,0.1)
        self.swn_avgeraging = swn_avgeraging
        son_avgeraging = 1 - swn_avgeraging
        
        # for each point in swn an son averaging get the corresponding values of kron and krwn of the fitted lines for each sample 
       
        samples_data = {}

        for I,sample in enumerate(self.samples):
            krwn_sample = simple_cory(swn_avgeraging, *total_params_water[I])
            kron_sample = simple_cory(son_avgeraging, *total_params_oil[I])
            samples_data[sample] = {'krw*':krwn_sample, 'kro*':kron_sample}
            
        #create a dataframe for krwn curves and kron curves combining all the samples to do the math
        kron_df = pd.DataFrame({s:perm_data["kro*"] for s , perm_data in samples_data.items()})
        krwn_df = pd.DataFrame({s:perm_data["krw*"] for s , perm_data in samples_data.items()})
        
        return kron_df , krwn_df , swn_avgeraging 
    
    
    def get_avg_for_all_samples(self):
        """
        Get the average curves for all samples.

        Returns:
        - fig (matplotlib.figure.Figure): The figure object.
        """
        kron_df , krwn_df , swn  = self.create_curves_for_samples()
        kron_df['kron'] = kron_df.mean(axis=1)
        krwn_df['krwn'] = krwn_df.mean(axis=1)
        
        # create the figure and return the data 
        fig, ax = self.plot_samples()
        
        # plot the fitted lines
        ax.plot(swn, kron_df['kron'], label = f'kron-Avg')
        ax.plot(swn, krwn_df['krwn'], label = f'krwn-Avg')
        
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # create a dataframe of only the saturation and the average values 
        self.avg_data = pd.DataFrame({'swn':swn , 'kron':kron_df['kron'], 'krwn':krwn_df['krwn']})

        return  fig , self.avg_data
        
        
    def get_avg_cruve_using_perm(self, kabs : dict ):
        
        kron_df , krwn_df , swn  = self.create_curves_for_samples()
        
        for i, col in enumerate(kron_df.columns):
            kron_df[col] = kron_df[col] * kabs[col]
            krwn_df[col] = krwn_df[col] * kabs[col]
            
            
        # sum all the columns in kron_df and krwn_df and then devide them by the sum of the kabs values
        kron_df['kron'] = kron_df.sum(axis=1)/sum(kabs.values())
        krwn_df['krwn'] = krwn_df.sum(axis=1)/sum(kabs.values())
        
        # create the figure and return the data 
        fig, ax = self.plot_samples()
            
        # plot the fitted lines
        ax.plot(swn, kron_df['kron'], label = f'kron-Weigted_Avg')
        ax.plot(swn, krwn_df['krwn'], label = f'krwn-Weighted_Avg')
        
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # create a dataframe of only the saturation and the average values 
        self.avg_data_using_perm = pd.DataFrame({'swn':swn , 'kron':kron_df['kron'], 'krwn':krwn_df['krwn']})

        return  fig , self.avg_data_using_perm
        
        
    def compare_fittings(self):
      
        # create the figure and return the data 
        fig, ax = self.plot_samples()
            
        # get the global average curve
        global_avg_curve=pd.DataFrame({
            "Sw*":self.swn_avgeraging,
            "kro_avg":self.avg_data["kron"],
            "krw_avg":self.avg_data["krwn"],
            "kro_avg_k":self.avg_data_using_perm["kron"],
            "krw_avg_k":self.avg_data_using_perm["krwn"],
            "kro_fit": ((1-self.swn_avgeraging) ** self._oil_fitted_n ),
            "krw_fit": (self.swn_avgeraging ** self._water_fitted_n )
        }).set_index("Sw*")
        
        global_avg_curve["kron_total_avg"] = global_avg_curve[["kro_avg","kro_avg_k","kro_fit"]].mean(axis=1)
        global_avg_curve["krwn_total_avg"] = global_avg_curve[["krw_avg","krw_avg_k","krw_fit"]].mean(axis=1)
            
        # show the Average values fitting
        ax.plot(self.swn_avgeraging, self.avg_data["kron"], label = 'Oil_Avg', color = '#2ECC71',marker='o',alpha=0.5)
        ax.plot(self.swn_avgeraging, self.avg_data["krwn"], label = 'Water_Avg', color = '#3498DB',marker='o',alpha=0.5)

        # show the Average values fitting using the average of the kabs values
        ax.plot(self.swn_avgeraging, self.avg_data_using_perm["kron"], label = 'Oil_Avg_K', color = '#27AE60',marker='v',alpha=0.5)
        ax.plot(self.swn_avgeraging, self.avg_data_using_perm["krwn"], label = 'Water_Avg_K', color = '#2471A3',marker='v',alpha=0.5)

        # show the Average values fitting
        # draw the fitted lines 
        ax.plot(self.swn_avgeraging, ((1-self.swn_avgeraging) ** self._oil_fitted_n ), label = 'Oil_Fit', color = '#148F77',marker='s',alpha=0.5)
        ax.plot(self.swn_avgeraging, (self.swn_avgeraging ** self._water_fitted_n ), label = 'Water_Fit', color = '#5D6D7E',marker='s',alpha=0.5)
        
         # draw the fitted lines 
        ax.plot(self.swn_avgeraging, global_avg_curve ["kron_total_avg"], label = 'Oil_Gloabl_Avg', color = 'red',marker='x',alpha=1)
        ax.plot(self.swn_avgeraging, global_avg_curve ["krwn_total_avg"], label = 'Water_Global_Avg', color = 'blue',marker='x',alpha=1)

        # show the minor and the major grid lines   
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='#333')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbb')
        ax.minorticks_on()

        # make the legend outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        return fig , global_avg_curve
        
    def denormalize(self, data : pd.DataFrame , mean_kro_max : str , mean_krw_max : str , mean_sor : str , mean_swc : str):
        
        data["Sw"] = (data.index * (1-mean_swc-mean_sor)) + mean_swc 
        data["kro_d"] = data["kron_total_avg"] * mean_kro_max
        data["krw_d"] = data["krwn_total_avg"] * mean_krw_max
        
        # fit cory function to the data
        def cory_water(sw,n):
            return mean_krw_max * ((sw- mean_swc) / (1-mean_swc- mean_sor) )**n

        def cory_oil(so,n):
            return mean_kro_max * ((so- mean_sor) / (1-mean_swc- mean_sor) )**n
        
        
        params_water , _= curve_fit(cory_water, data["Sw"], data["krw_d"])
        params_oil , _ = curve_fit(cory_oil, 1-data["Sw"], data["kro_d"])
        
        
        # make a general plot for each , for each sample add the corresponding values of kron and krwn to the plot and add a legend 
        fig, ax = plt.subplots(figsize=(15, 6),dpi=500)
            
        ax.set_xlabel('swn',fontsize=13, fontweight='medium', color='#17202A')
        ax.set_ylabel('kron, krwn',fontsize=13, fontweight='medium', color='#17202A') 
        # add a title with font size 18 and medium weight and padding and color
        ax.set_title('kron, krwn vs Sw - Denormalized', fontsize=18, fontweight='bold', color='#17202A',pad=20)
            
        # show the Average values fitting
        ax.scatter(data["Sw"], data["kro_d"], label = 'Oil_Global_Avg', color = '#2ECC71',marker='o',s=18)
        ax.scatter(data["Sw"], data["krw_d"], label = 'Water_Global_Avg', color = '#3498DB',marker='o',s=18)
        
        swn  = np.linspace( data["Sw"].min() , data["Sw"].max() , 100 )
        ax.plot(swn, cory_oil(1-swn,params_oil), label = 'Oil-Cory', color = '#2ECC71', markersize=7 ,alpha=0.5)
        ax.plot(swn, cory_water(swn,params_water), label = 'Water-Cory', color = '#3498DB',markersize=7,alpha=0.5)



        # show the minor and the major grid lines   
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='#333')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbb')
        ax.minorticks_on()

        # make the legend outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
       
        return fig , {"Oil-Cory":params_oil[0] , "Water-Cory":params_water[0]}
        
        
        
        
        
        
        
                
                
                
                
        
        
            
            
            

        
                
        
        