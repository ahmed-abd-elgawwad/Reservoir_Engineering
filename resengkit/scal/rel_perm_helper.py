import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def fit_simple_cory( swn : list[float], son : list[float], kron : list[float], krwn : list[float]) -> tuple:
    """
    Fit Corey model to the data
    """
    simple_cory = lambda x , n : x**n
    
    oil_params, _ = curve_fit(simple_cory, son, kron)
    water_params, _ = curve_fit(simple_cory, swn, krwn)
    
    oil_n = oil_params[0]
    water_n = water_params[0]
    
    
    # calculate the r2 , MAE, RMSE for the fit
    kron_fit = simple_cory(son, oil_n)
    krwn_fit = simple_cory(swn, water_n)
    
    kron_mae = np.mean(np.abs(kron - kron_fit))
    krwn_mae = np.mean(np.abs(krwn - krwn_fit))

    kron_rmse = np.sqrt(np.mean((kron - kron_fit)**2))
    krwn_rmse = np.sqrt(np.mean((krwn - krwn_fit)**2))
    
    # calculate the r2 for the two curves
    kron_r2 = 1 - np.sum((kron - kron_fit)**2) / np.sum((kron - np.mean(kron))**2)
    krwn_r2 = 1 - np.sum((krwn - krwn_fit)**2) / np.sum((krwn - np.mean(krwn))**2)
    
    # combine the results in a dictionary
    results = {"N Oil":oil_n , "N Water":water_n , "Kron MAE":kron_mae , "Krwn MAE":krwn_mae , "Kron RMSE":kron_rmse , "Krwn RMSE":krwn_rmse , "Kron R2":kron_r2 , "Krwn R2":krwn_r2}
    
    
    return results

def fit_full_cory( swn : list[float], son : list[float], kron : list[float], krwn : list[float] 
                  , mean_krw_max : float , mean_kro_max : float , mean_swc : float , mean_sor :float ) -> tuple:
    """
    Fit Corey model to the data
    """
    # fit cory function to the data
    def cory_water(sw,n):
        return mean_krw_max * ((sw- mean_swc) / (1-mean_swc- mean_sor) )**n

    def cory_oil(so,n):
        return mean_kro_max * ((so- mean_sor) / (1-mean_swc- mean_sor) )**n
    
    params_water , _= curve_fit(cory_water, swn, krwn)
    params_oil , _ = curve_fit(cory_oil, son, kron)
        
    
    water_n = params_water[0]
    oil_n = params_oil[0]
    
    # calculate the r2 , MAE, RMSE for the fit
    kron_fit = cory_oil(son, oil_n)
    krwn_fit = cory_water(swn, water_n)
    
    kron_mae = np.mean(np.abs(kron - kron_fit))
    krwn_mae = np.mean(np.abs(krwn - krwn_fit))

    kron_rmse = np.sqrt(np.mean((kron - kron_fit)**2))
    krwn_rmse = np.sqrt(np.mean((krwn - krwn_fit)**2))
    
    # calculate the r2 for the two curves
    kron_r2 = 1 - np.sum((kron - kron_fit)**2) / np.sum((kron - np.mean(kron))**2)
    krwn_r2 = 1 - np.sum((krwn - krwn_fit)**2) / np.sum((krwn - np.mean(krwn))**2)
    
    # combine the results in a dictionary
    results = {"N Oil":oil_n  , "N Water":water_n , "Kron MAE":kron_mae , "Krwn MAE":krwn_mae , "Kron RMSE":kron_rmse , "Krwn RMSE":krwn_rmse , "Kron R2":kron_r2 , "Krwn R2":krwn_r2}
    
    
    return results