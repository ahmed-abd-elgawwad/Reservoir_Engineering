import streamlit as st 
import pandas as pd 
from resengkit.scal import relative_perm_2

normalized_curves = None
st.title("SCAL Module")

with st.expander("Data Loading"):
    # define the system to map the columns 
    st.markdown("### Define The System and Map The Columns")
    system_type = st.selectbox("Select The System Type", ["oil","gas"])
    core_sys = relative_perm_2.CoreSystem(system_type)
    curves_cols , parameter_cols = core_sys.system_data()
    
    col1, col2 = st.columns(2)
    
    # the curves data column 
    col1.markdown("### Curves Data")
    curves_file = col1.file_uploader("Upload Samples Curves File", type=[ 'xlsx'])
    if curves_file:
        col1.success("File Uploaded Successfully")
        curves_df = pd.read_excel(curves_file)
        col1.dataframe(curves_df,use_container_width=True)
        
    # the parameters data column
    col2.markdown("### Parameters Data")
    params_file = col2.file_uploader("Upload Samples Parameters File", type=['xlsx'])
    
    if params_file:
        col2.success("File Uploaded Successfully")
        params_df = pd.read_excel(params_file)
        col2.dataframe(params_df,use_container_width=True)
        
if curves_file and params_file:
    col1, col2 = st.columns(2)
    # get the corresponding columns to the system
    curves_cols_maping = {}
    parameters_cols_mapping = {}
    for col in curves_df.columns:
            curves_cols_maping[col] = col1.selectbox(f"Map {col} to ", curves_cols)
    
    for col in params_df.columns:
            parameters_cols_mapping[col] = col2.selectbox(f"Map {col} to ", parameter_cols)
            
    # check that each column is mapped to a valid column only once and note mapped to more than one column
    if len(set(curves_cols_maping.values())) != len(curves_cols_maping.values()):
        col1.error("Some columns are mapped to the same column")
        
    elif len(set(parameters_cols_mapping.values())) != len(parameters_cols_mapping.values()):
        col2.error("Some columns are mapped to the same column")
        
    # Define the New dataframes with the mapped columns 
    curves_df.rename(columns=curves_cols_maping, inplace=True)
    params_df.rename(columns=parameters_cols_mapping, inplace=True)
    
    # QC on the Curves and the parameters columns
    try:
        data_checker = relative_perm_2.DataValidator(curves_df, params_df , system_type)
        data_checker._data_check()
        global data_checked 
        data_checked = True
    except Exception as e:
        st.error(f"Error: {e}")
        
    if data_checked:
        with st.expander("Samples Data Statistics"):
            st.dataframe(params_df.describe() , use_container_width=True)
            
            statistis_summary = relative_perm_2.SampleStatistics(curves_df, params_df)
            correlation_fig = statistis_summary.parameters_correlation()
            st.pyplot(correlation_fig)
            relationship_fig = statistis_summary.parameters_relationships()
            st.pyplot(relationship_fig)
            
            
        with st.expander("Curves Normalization"):

            normalizer = relative_perm_2.Normalizer(curves_df, params_df,system_type)
            corrected_curves = curves_df.copy()
                
            if st.checkbox("Correct The Relative Permeability for new Base curve"):
                try:
                    cols = st.columns(2)
                    old_base_k = cols[0].selectbox("Select Old Base Perm ", params_df.columns)
                    new_base_k = cols[1].selectbox("Select New Base Perm ", params_df.columns)
                    corrected_curves = normalizer.correct_curves_for_base(old_base_k, new_base_k)
                except Exception as e:
                    st.error("There is an error while correcting the curves. Make sure you used valid columns corresponding to permeability values")
            # else:
            #     corrected_curves = normalizer.samples_cruves.copy()
                
            st.markdown("<hr>",unsafe_allow_html=True)    
            st.markdown("### **Final Relative Permeability Curves**")   
            st.dataframe(corrected_curves,use_container_width=True)
            
            st.markdown("<hr>",unsafe_allow_html=True)
            normalized_curves = normalizer.normalize(corrected_data=corrected_curves)
            col1, col2 = st.columns(2)
            col1.dataframe(normalized_curves,use_container_width=True)
            fig , ax = normalizer.plot_samples_normalized(normalized_curves)
            col2.pyplot(fig,use_container_width=True)
            
if normalized_curves is not None:
        with st.expander("Get General Curve"):
            averager = relative_perm_2.RelPerm(normalized_curves,system_type)  
                
            cols = st.columns(3)
            cols[0].markdown("**Fit General Trend Using `Power Law`**")
            fitting_fig  , fitting_results = averager.fit_general_trend()
            cols[0].pyplot(fitting_fig,use_container_width=True)
            cols[0].table(fitting_results)
                
        
            cols[1].markdown("**Get The Avg from Samples**")
            avg_fig , average_curve = averager.get_avg_for_all_samples()
            cols[1].pyplot(avg_fig,use_container_width=True)
            cols[1].dataframe(average_curve,use_container_width=True)
                
                
        
            cols[2].markdown("**Weighted Average**")
            try:
                selected_criteria = cols[2].selectbox("Select Weighting Criteria", params_df.columns)
            
                criteria = {}
                for i , row in params_df[["sample",selected_criteria]].iterrows():
                    criteria[row["sample"]] = row[selected_criteria]
                
                fig , weighted_average_curve = averager.get_avg_cruve_using_perm(criteria)
                cols[2].pyplot(fig,use_container_width=True)
                cols[2].dataframe(weighted_average_curve,use_container_width = True)
                
            except:
                st.error("There is an error while calculating the weighted average. Make sure you selected the correct column")
                    
        with st.expander("Final Comparison"):        
            st.markdown("### **Compare The Three Methods**")
            cols = st.columns(2)
            compare_plot , global_avg = averager.compare_fittings()
            cols[0].pyplot(compare_plot , use_container_width=True)
            cols[1].dataframe(global_avg , use_container_width=True)
            
            if global_avg.empty == False:
                st.session_state.global_avg = global_avg
                st.session_state.step = 2
                    
        with st.expander("Denormalization"):

            cols =  st.columns(2)   
            if system_type == "oil":
                mean_kromax_values = curves_df.groupby("sample")["kro"].max().mean()
            else:
                mean_kromax_values = curves_df.groupby("sample")["krg"].max().mean()
            mean_krwmax_values = curves_df.groupby("sample")["krw"].max().mean()
            
            mean_swc_values = params_df["swc"].mean()
            
            if system_type == "oil":
                mean_sr_values = params_df["sor"].mean()
            else:  
                mean_sr_values = params_df["sgr"].mean()
                
            mean_kromax = cols[0].number_input("Enter Mean **Kro_Max/Krg_Max**", value=mean_kromax_values)
            mean_krwmax = cols[0].number_input("Enter Mean **Krw_Max**", value=mean_krwmax_values)
            mean_swc = cols[1].number_input("Enter Mean **Swc**",value=mean_swc_values )
            mean_sor = cols[1].number_input("Enter Mean **Sor/Sgr**",value = mean_sr_values)
        
                    
            fig , cory_exponents = averager.denormalize(st.session_state.global_avg , mean_kromax, mean_krwmax, mean_swc, mean_sor)    
            
            st.pyplot(fig)
            cols = st.columns(3)
            cols[1].table(cory_exponents)

                            


