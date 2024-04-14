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
from typing import List, Dict,Tuple

class CoreSystem:
    def __init__(self, systemtype : str) -> None:
        self.sys = systemtype
        # chech if the system type is oil or gas and it must be one of these two
        if self.sys not in ["oil", "gas"]:
            raise ValueError("The system type must be either `oil` or `gas`")
        
    def system_data(self) -> Tuple[List[str], List[str]]:
        if self.sys == "oil":
            curve_cols = ["sample","sw","kro","krw"]
            basic_sample_parameters = ["sample","swc","sor","phi","kabs" ,"Ko@Swc" ,"Kw@Sor","Depth"]
        else:
            curve_cols = ["sample","sw","krw","krg"]
            basic_sample_parameters = ["sample","swc","sgr","phi","kabs" ,"Kw@Sgr" ,"Kg@Swc","Depth"]
            
        return [curve_cols , basic_sample_parameters]
    
            
     
class DataValidator:
    def __init__(self , samples_curves : pd.DataFrame , samples_parameters : pd.DataFrame , system_type : str ) -> None:
        """
        Initialize the Core Data class.

        Parameters:
        - samples_curves (pd.DataFrame): Dataframe containing the curves data.
        - samples_parameters (pd.DataFrame): Dataframe containing the samples data.
        """
        self.samples_curves = samples_curves
        self.samples_parameters = samples_parameters
        self.system_type = system_type
        
    def _data_check(self) -> None:
        
        
        # 1. Check that the samples in the samples_cruves are in samples_parameters
        if not self.samples_curves["sample"].isin(self.samples_parameters["sample"]).all():
            raise ValueError("The samples in the curves data must be in the samples data.")
        
        # 2. Check the values of the curves data
        ## 2.1 Sw values
        if not self.samples_curves["sw"].between(0,1).all():
            if self.samples_curves["sw"].between(0,100).all():
                self.samples_curves["sw"] = self.samples_curves["sw"] / 100
            else:
                raise ValueError("The sw values must be in the range (0,1) or (0,100)")
        
        ## 2.2 krw values
        if not self.samples_curves["krw"].between(0,1).all():
            raise ValueError("The krw values must be in the range (0,1)")
        
        ## 2.3 kro values if oil and krg values if gas
        if self.system_type == "oil":
            if not self.samples_curves["kro"].between(0,1).all():
                raise ValueError("The kro values must be in the range (0,1)")
            
        elif self.system_type == "gas":
            if not self.samples_curves["krg"].between(0,1).all():
                raise ValueError("The krg values must be in the range (0,1)")
            
        
        # 3. Check the samples parameters Data
        ## 3.1 swc values
        if not self.samples_parameters["swc"].between(0,1).all():
            if self.samples_parameters["swc"].between(0,100).all():
                self.samples_parameters["swc"] = self.samples_parameters["swc"] / 100
            else:
                raise ValueError("The swc values must be in the range (0,1) or (0,100)")
            
        ## 3.2 check the sor if the system is oil or the sgr if the system is gas
        if self.system_type == "oil":
            if not self.samples_parameters["sor"].between(0,1).all():
                if self.samples_parameters["sor"].between(0,100).all():
                    self.samples_parameters["sor"] = self.samples_parameters["sor"] / 100
                else:
                    raise ValueError("The sor values must be in the range (0,1)")
        else:
            if not self.samples_parameters["sgr"].between(0,1).all():
                if self.samples_parameters["sgr"].between(0,100).all():
                    self.samples_parameters["sgr"] = self.samples_parameters["sgr"] / 100
                else:
                    raise ValueError("The sgr values must be in the range (0,1)")
            
        ## 3.3 check the phi values are between (0,1)
        if not self.samples_parameters["phi"].between(0,1).all():
            if self.samples_parameters["phi"].between(0,100).all():
                self.samples_parameters["phi"] = self.samples_parameters["phi"] / 100
            else:
                raise ValueError("The phi values must be in the range (0,1) or (0,100)")

class SampleStatistics:
    def __init__(self , samples_curves , samples_data):
        self.samples_curves = samples_curves
        self.samples_data = samples_data
        
    def parameters_correlation(self):
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
        
    
    def parameters_relationships(self):
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
    
    
class Normalizer:
    def __init__(self, samples_curves , samples_data , system_stype):
        """
        Initialize the Normalizer class.

        Parameters:
        - samples_curves (pd.DataFrame): Dataframe containing the curves data.
        - samples_data (pd.DataFrame): Dataframe containing the samples data.
        """
        self.samples_cruves = samples_curves
        self.samples_data = samples_data 
        self.system_stype = system_stype
        self.samples = self.samples_data["sample"].unique()
    
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
            self.corrected_data.loc[mask_curves , "krw"] = self.corrected_data.loc[mask_curves , "krw"] * base_correction
            if self.system_stype == "oil":
                self.corrected_data.loc[mask_curves , "kro"] = self.corrected_data.loc[mask_curves , "kro"] * base_correction
            else:
                self.corrected_data.loc[mask_curves , "krg"] = self.corrected_data.loc[mask_curves , "krg"] * base_correction
            
        return self.corrected_data
    
    def normalize(self, corrected_data : pd.DataFrame) -> pd.DataFrame:
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
            swc = self.samples_data.loc[mask_parameters, "swc"].values[0]
            
            if self.system_stype == "oil":
                sr = self.samples_data.loc[mask_parameters, "sor"].values[0]
            else:
                sr = self.samples_data.loc[mask_parameters, "sgr"].values[0]
            
            # normalize the sw values
            sw = normalized_data.loc[mask_curves , "sw"]
            sw_corrected =  (sw - swc) / (1 - swc - sr)
            
            normalized_data.loc[mask_curves , "sw"] = sw_corrected
            normalized_data.loc[mask_curves , "krw"] = normalized_data.loc[mask_curves , "krw"] / normalized_data.loc[mask_curves, "krw"].max()
            
            if self.system_stype == "oil":
                normalized_data.loc[mask_curves , "kro"] = normalized_data.loc[mask_curves , "kro"] / normalized_data.loc[mask_curves, "kro"].max()
            else:
                normalized_data.loc[mask_curves , "krg"] = normalized_data.loc[mask_curves , "krg"] / normalized_data.loc[mask_curves, "krg"].max()
            
        return normalized_data.round(3)
    
    def plot_samples_normalized(self , normalized_data):
        """
        Plot the samples data.

        Returns:
        - fig (matplotlib.figure.Figure): The figure object.
        """
        data  = normalized_data.copy()
        fig, ax = plt.subplots(figsize=(10, 6),dpi=100)

        for sample in self.samples:
        # plot swn in the x axis and kron,Krwn in the y axis
            x = data[data["sample"] == sample]["sw"]
            krw = data[data["sample"]  == sample]["krw"]
            if self.system_stype == "oil":
                kro = data[data["sample"]  == sample]["kro"]
                ax.scatter(x , krw , label = f'kron-{sample}',s=5)
                ax.scatter(x , kro , label = f'krwn-{sample}',s=5)
                ax.set_ylabel('kron, krwn',fontsize=13, fontweight='medium', color='#17202A')
                ax.set_title('kron, krwn vs swn', fontsize=18, fontweight='bold', color='#17202A',pad=20)
            else:
                krg = data[data["sample"]  == sample]["krg"]
                ax.scatter(x , krw , label = f'krgn-{sample}',s=5)
                ax.scatter(x , krg , label = f'krwn-{sample}',s=5)
                ax.set_ylabel('krgn, krwn',fontsize=13, fontweight='medium', color='#17202A')
                ax.set_title('krgn, krwn vs swn', fontsize=18, fontweight='bold', color='#17202A',pad=20)
        
            
          
    
        ax.set_xlabel('swn',fontsize=13, fontweight='medium', color='#17202A')
        # add a title with font size 18 and medium weight and padding and color
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        return (fig , ax)
        
        
class RelPerm:
    def __init__(self, data : pd.DataFrame  , system_stype : str):
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
        self.system_type = system_stype
        
        # setting the protected attributes
        self._other_fitted_n = None 
        self._water_fitted_n = None 
        
        # process the data 
        self.data_process()
        
    def data_process(self) -> None :
        """
        Process the data by adding the son column and getting the unique samples.
        
        """
        self.data["S_other"] = 1-self.data["sw"]
        self.samples = self.data["sample"].unique()
        
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
            x = self.data[self.data["sample"] == sample]["sw"]
            krw = self.data[self.data["sample"]  == sample]["krw"]
            
            if self.system_type == "oil":
                kro = self.data[self.data["sample"]  == sample]["kro"]
                ax.scatter(x , krw , label = f'kron-{sample}',s=5)
                ax.scatter(x , kro , label = f'krwn-{sample}',s=5)
                ax.set_ylabel('kron, krwn')
                ax.set_title('kron, krwn vs swn')
            else:
                krg = self.data[self.data["sample"]  == sample]["krg"]
                ax.scatter(x , krw , label = f'krgn-{sample}',s=5)
                ax.scatter(x , krg , label = f'krwn-{sample}',s=5)
                ax.set_ylabel('krgn, krwn')
                ax.set_title('krgn, krwn vs swn')
        
        ax.set_xlabel('swn')
        # add a title with font size 18 and medium weight and padding and color
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
        swn = self.data["sw"]
        krwn = self.data["krw"]
        s_other = self.data["S_other"]
        
        if self.system_type == "oil":
            kr_other = self.data["kro"]
        else:
            kr_other = self.data["krg"]
        
        
        results = rel_perm_helper.fit_simple_cory(swn, s_other, kr_other, krwn)
        
        other_n = results["N Oil"]
        water_n = results["N Water"]
        
        # setting the protected attributes
        self._other_fitted_n = other_n
        self._water_fitted_n = water_n
        
        
        # get the curves for the fit
        swn_curve = np.linspace(0,1,100)
        kron_curve = simple_cory(1-swn_curve, other_n)
        krwn_curve = simple_cory(swn_curve, water_n)
        
        fig , ax = self.plot_samples()
            
        # plot the fitted lines
        if self.system_type == "oil":
            ax.plot(swn_curve, kron_curve, label = f'kron-fit')
        else:
            ax.plot(swn_curve, kron_curve, label = f'krgn-fit')
            
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
        total_params_other = []
        total_params_water = []
        
        simple_cory = lambda x, n: x**n
        
        for sample in self.samples:
            dd = self.data[self.data["sample"] == sample]
            if self.system_type =="oil":
                sample_other_params, _ = curve_fit(simple_cory, dd["S_other"], dd["kro"])
            else:
                sample_other_params, _ = curve_fit(simple_cory, dd["S_other"], dd["krg"])
                
            sample_water_params, _ = curve_fit(simple_cory, dd["sw"], dd["krw"])
            
            total_params_other.append(sample_other_params)
            total_params_water.append(sample_water_params)
            
            
        # after getting the parameters for all the samples , we can create the curves
        swn_avgeraging = np.arange(0,1.1,0.1)
        self.swn_avgeraging = swn_avgeraging
        son_avgeraging = 1 - swn_avgeraging
        
        # for each point in swn an son averaging get the corresponding values of kron and krwn of the fitted lines for each sample 
       
        samples_data = {}

        for I,sample in enumerate(self.samples):
            krwn_sample = simple_cory(swn_avgeraging, *total_params_water[I])
            kr_other_sample = simple_cory(son_avgeraging, *total_params_other[I])
            samples_data[sample] = {'krw*':krwn_sample, 'kr*':kr_other_sample}
            
        #create a dataframe for krwn curves and kron curves combining all the samples to do the math
        krn_df = pd.DataFrame({s:perm_data["kr*"] for s , perm_data in samples_data.items()})
        krwn_df = pd.DataFrame({s:perm_data["krw*"] for s , perm_data in samples_data.items()})
        
        return krn_df , krwn_df , swn_avgeraging 
    
    
    def get_avg_for_all_samples(self):
        """
        Get the average curves for all samples.

        Returns:
        - fig (matplotlib.figure.Figure): The figure object.
        """
        krn_df , krwn_df , swn  = self.create_curves_for_samples()
        krn_df['krn'] = krn_df.mean(axis=1)
        krwn_df['krwn'] = krwn_df.mean(axis=1)
        
        # create the figure and return the data 
        fig, ax = self.plot_samples()
        
        # plot the fitted lines
        ax.plot(swn, krn_df['krn'], label = "kron" if self.system_type == "oil" else "krgn-Avg")
        ax.plot(swn, krwn_df['krwn'], label = f'krwn-Avg')
        
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # create a dataframe of only the saturation and the average values 
        if self.system_type == "oil":
            self.avg_data = pd.DataFrame({'swn':swn , 'kron':krn_df['krn'], 'krwn':krwn_df['krwn']})
        else:
            self.avg_data = pd.DataFrame({'swn':swn , 'krgn':krn_df['krn'], 'krwn':krwn_df['krwn']})

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
        if self.system_type == "oil":
            self.avg_data_using_perm = pd.DataFrame({'swn':swn , 'kron':kron_df['kron'], 'krwn':krwn_df['krwn']})
        else:
            self.avg_data_using_perm = pd.DataFrame({'swn':swn , 'krgn':kron_df['kron'], 'krwn':krwn_df['krwn']})

        return  fig , self.avg_data_using_perm
        
        
    def compare_fittings(self):
      
        # create the figure and return the data 
        fig, ax = self.plot_samples()
            
        # get the global average curve
        global_avg_curve=pd.DataFrame({
            "Sw*":self.swn_avgeraging,
            "kr_avg":self.avg_data["kron"] if self.system_type == "oil" else self.avg_data["krgn"],
            "krw_avg":self.avg_data["krwn"],
            "kr_avg_k":self.avg_data_using_perm["kron"] if self.system_type == "oil" else self.avg_data_using_perm["krgn"] ,
            "krw_avg_k":self.avg_data_using_perm["krwn"],
            "kr_fit": ((1-self.swn_avgeraging) ** self._other_fitted_n ),
            "krw_fit": (self.swn_avgeraging ** self._water_fitted_n )
        }).set_index("Sw*")
        
        global_avg_curve["krn_total_avg"] = global_avg_curve[["kr_avg","kr_avg_k","kr_fit"]].mean(axis=1)
        global_avg_curve["krwn_total_avg"] = global_avg_curve[["krw_avg","krw_avg_k","krw_fit"]].mean(axis=1)
        
        # setup the drawing according to the system
        if self.system_type =="oil":
            y_avg = self.avg_data["kron"]
            y_avg_k = self.avg_data_using_perm["kron"]
            
        else:
            y_avg = self.avg_data["krgn"]
            y_avg_k = self.avg_data_using_perm["krgn"]
            
        # show the Average values fitting
        ax.plot(self.swn_avgeraging, y_avg, label = f'{self.system_type}_Avg', color = '#2ECC71',marker='o',alpha=0.5)
        ax.plot(self.swn_avgeraging, self.avg_data["krwn"], label = 'Water_Avg', color = '#3498DB',marker='o',alpha=0.5)

        # show the Average values fitting using the average of the kabs values
        ax.plot(self.swn_avgeraging, y_avg_k, label = f'{self.system_type}_Avg_K', color = '#27AE60',marker='v',alpha=0.5)
        ax.plot(self.swn_avgeraging, self.avg_data_using_perm["krwn"], label = 'Water_Avg_K', color = '#2471A3',marker='v',alpha=0.5)

        # show the Average values fitting
        # draw the fitted lines 
        ax.plot(self.swn_avgeraging, ((1-self.swn_avgeraging) ** self._other_fitted_n ), label = f'{self.system_type}_Fit', color = '#148F77',marker='s',alpha=0.5)
        ax.plot(self.swn_avgeraging, (self.swn_avgeraging ** self._water_fitted_n ), label = 'Water_Fit', color = '#5D6D7E',marker='s',alpha=0.5)
        
         # draw the fitted lines 
        ax.plot(self.swn_avgeraging, global_avg_curve ["krn_total_avg"], label = f'{self.system_type}_Gloabl_Avg', color = 'red',marker='x',alpha=1)
        ax.plot(self.swn_avgeraging, global_avg_curve ["krwn_total_avg"], label = 'Water_Global_Avg', color = 'blue',marker='x',alpha=1)

        # # show the minor and the major grid lines   
        # ax.grid(which='major', linestyle='-', linewidth='0.5', color='#333')
        # ax.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbb')
        # ax.minorticks_on()

        # make the legend outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        return fig , global_avg_curve
        
    def denormalize(self, data : pd.DataFrame , mean_kro_max : str , mean_krw_max : str , mean_sor : str , mean_swc : str):
        
        data["Sw"] = (data.index * (1-mean_swc-mean_sor)) + mean_swc 
        data["kr_d"] = data["krn_total_avg"] * mean_kro_max
        data["krw_d"] = data["krwn_total_avg"] * mean_krw_max
        
        # fit cory function to the data
        def cory_water(sw,n):
            return mean_krw_max * ((sw- mean_swc) / (1-mean_swc- mean_sor) )**n

        def cory_oil(so,n):
            return mean_kro_max * ((so- mean_sor) / (1-mean_swc- mean_sor) )**n
        
        
        params_water , _= curve_fit(cory_water, data["Sw"], data["krw_d"])
        params_oil , _ = curve_fit(cory_oil, 1-data["Sw"], data["kr_d"])
        
        
        # make a general plot for each , for each sample add the corresponding values of kron and krwn to the plot and add a legend 
        fig, ax = plt.subplots(figsize=(15, 6),dpi=500)
            
        if self.system_type == "oil":
            ax.set_xlabel('swn',fontsize=13)
            ax.set_ylabel('kron, krwn') 
            ax.set_title('kron, krwn vs Sw - Denormalized')
        else:
            
            ax.set_xlabel('Sw')
            ax.set_ylabel('Krg, Krw') 
            ax.set_title('Krg, Krw vs Sw - Denormalized')
            
        # show the Average values fitting
        ax.scatter(data["Sw"], data["kr_d"], label = f'{ self.system_type }_Global_Avg', color = '#2ECC71', marker='o',s=18)
        ax.scatter(data["Sw"], data["krw_d"], label = 'Water_Global_Avg', color = '#3498DB',marker='o',s=18)
        
        swn  = np.linspace( data["Sw"].min() , data["Sw"].max() , 100 )
        ax.plot(swn, cory_oil(1-swn,params_oil), label = f'{ self.system_type }-Cory', color = '#2ECC71', markersize=7 ,alpha=0.5)
        ax.plot(swn, cory_water(swn,params_water), label = 'Water-Cory', color = '#3498DB',markersize=7,alpha=0.5)

        # # show the minor and the major grid lines   
        # ax.grid(which='major', linestyle='-', linewidth='0.5', color='#333')
        # ax.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbb')
        # ax.minorticks_on()

        # make the legend outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
       
        return fig , {f"{self.system_type.title()}-Cory":params_oil[0] , "Water-Cory":params_water[0]}
        