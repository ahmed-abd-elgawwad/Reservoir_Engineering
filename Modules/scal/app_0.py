import streamlit as st 
import pandas as pd 
from resengkit.scal import relative_perm

st.title("SCAL Module")


# importing the files
st.sidebar.title("Upload Files")
curves_file = st.sidebar.file_uploader("Upload Samples Curves File", type=[ 'xlsx'])
parameres_file = st.sidebar.file_uploader("Upload Samples Parameters File", type=['xlsx'])

if 'step' not in st.session_state:
    st.session_state.step = 0


if ( curves_file and parameres_file ):
    
    # loading and define the variables in the data
    with st.expander("Show Uploaded Files"):
        
        try:
            curves = pd.read_excel(curves_file)
            params = pd.read_excel(parameres_file)
            
            # save them inside the session state to avoid reloading them again
            st.session_state.curves = curves    
            st.session_state.params = params
            
            # # initialize the corrected curves to be the same as the curves if there is no correction for the base
            # corrected_curves =  st.session_state.curves.copy()
            
            col1 , col2 = st.columns(2)
            with col1:
                st.markdown("### Samples Curves")
                st.dataframe(curves , use_container_width=True)
            
            with col2: 
                st.markdown("### Samples Parameters")
                st.dataframe(params , use_container_width=True)
            
            st.markdown("<hr>",unsafe_allow_html=True)
            
            # select samples columns
            st.markdown("### Identify the Required Columns")
            col1 , col2 = st.columns(2) 
            curves_sample_column = col1.selectbox("Select Samples Column From Curves Data", curves.columns)
            parameters_sample_column = col2.selectbox("Select Samples Column From Parameter Data", params.columns)
            
            st.session_state.curves.rename(columns={curves_sample_column:"sample"}, inplace=True)
            st.session_state.params.rename(columns={parameters_sample_column:"sample"}, inplace=True)
            
            # check if all the samples inside the curves data are inside the parameters data
            if not curves[curves_sample_column].isin(params[parameters_sample_column]).all():
                st.error("Some samples in the curves data are not found in the parameters data")
            
            else:
                st.success("All samples in the curves data are found in the parameters data")
                
                # get the other required inputs 
                cols = st.columns(2)
                st.session_state.sw_col = cols[0].selectbox("Select **Sw** Column From `Curves`", curves.columns)
                st.session_state.kro_col = cols[0].selectbox("Select **Kro** Column From `Curves`", curves.columns)
                st.session_state.krw_col = cols[0].selectbox("Select **Krw** Column From `Curves`", curves.columns)
                st.session_state.swc_col = cols[1].selectbox("Select **Swc** Column From the `Parameters`", params.columns)
                st.session_state.sor_col = cols[1].selectbox("Select **Sor** Column From the `Parameters`", params.columns)
                
                st.markdown("<hr>",unsafe_allow_html=True) 
                
                # make sure we entered the correct columns
                check1 = isinstance(st.session_state.sw_col,str) and isinstance(st.session_state.kro_col,str) and isinstance(st.session_state.krw_col,str)
                check2 = isinstance(st.session_state.swc_col,str) and isinstance(st.session_state.sor_col,str)
                check3 = isinstance(st.session_state.curves,pd.DataFrame) and isinstance(st.session_state.params,pd.DataFrame)
                check4 = ( st.session_state.curves.shape[0] > 0 ) and ( st.session_state.params.shape[0] > 0 ) 
                check5 = (st.session_state.curves[st.session_state.sw_col].dtype != "object" ) and (st.session_state.curves[st.session_state.kro_col].dtype != "object") and (st.session_state.curves[st.session_state.krw_col].dtype != "object")
                check6 = (st.session_state.params[st.session_state.swc_col].dtype != "object") and (st.session_state.params[st.session_state.sor_col].dtype != "object")
                if check1 and check2 and check3 and check4 and check5 and check6:
                    st.session_state.step = 1
                    
        except Exception as e:
            st.exception("Make Sure You Uploaded The Correct Files and the Correct Columns")
            
    if st.session_state.step == 1:
            with st.expander("Normalization"):

                    # initialize the noramalizer 
                    normalizer = relative_perm.Normalizer(st.session_state.curves, st.session_state.params)
                        
                    # show sampl
                    if st.checkbox("Show Samples Parameters Analysis "):
                        correlation_fig = normalizer.sample_parameter_correlation_summary()
                        relationship_fig = normalizer.sample_parameter_relationships()
                        cols = st.columns(2)
                        cols[0].pyplot(correlation_fig)
                        cols[1].pyplot(relationship_fig)
                
                    if st.checkbox("Correct The Relative Permeability for new Base curve"):
                        try:
                            cols = st.columns(2)
                            old_base_k = cols[0].selectbox("Select Old Base Perm ", params.columns)
                            new_base_k = cols[1].selectbox("Select New Base Perm ", params.columns)
                            corrected_curves = normalizer.correct_curves_for_base(old_base_k, new_base_k)
                        except Exception as e:
                            st.error("There is an error while correcting the curves. Make sure you used valid columns corresponding to permeability values")
                    else:
                        corrected_curves = normalizer.samples_cruves.copy()
                        
                    st.markdown("<hr>",unsafe_allow_html=True)    
                    st.markdown("### **Final Relative Permeability Curves**")   
                    st.dataframe(corrected_curves,use_container_width=True)
                    
                    st.markdown("<hr>",unsafe_allow_html=True)
                    st.header("Normalize The Curves")
    
                    # normalize the curves
                    normalized_curves = normalizer.normalize(corrected_data=corrected_curves,sw_col=st.session_state.sw_col, kro_col=st.session_state.kro_col,
                                                            krw_col= st.session_state.krw_col, swc_col= st.session_state.swc_col,  sor_col=st.session_state.sor_col)
                    
                    averager = relative_perm.RelPerm(normalized_curves,"sample",st.session_state.sw_col,st.session_state.kro_col,st.session_state.krw_col)
                    cols = st.columns(2)
                    
                    cols[0].dataframe(normalized_curves , use_container_width=True)
                    cols[1].pyplot(averager.plot_samples()[0] , use_container_width=True)
                    
            with st.expander("Average Relative Permeability Curve"):          
                
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
                    selected_criteria = cols[2].selectbox("Select Weighting Criteria", params.columns)
                
                    criteria = {}
                    for i , row in params[["sample",selected_criteria]].iterrows():
                        criteria[row["sample"]] = row[selected_criteria]
                        
                    
                    fig , weighted_average_curve = averager.get_avg_cruve_using_perm(criteria)
                    cols[2].pyplot(fig,use_container_width=True)
                    cols[2].dataframe(weighted_average_curve,use_container_width=   True)
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
                    
    else:
        st.error("Please Select the Required Columns and make sure you entered the correct data")       
                 
    if st.session_state.step == 2:
        with st.expander("Denormalization"):

            cols =  st.columns(2)   
            mean_kromax = cols[0].number_input("Enter Mean **Kro_Max**", )
            mean_krwmax = cols[0].number_input("Enter Mean **Krw_Max**", )
            mean_swc = cols[1].number_input("Enter Mean **Swc**", )
            mean_sor = cols[1].number_input("Enter Mean **Sor**",)      
                    
            fig , cory_exponents = averager.denormalize(st.session_state.global_avg , mean_kromax, mean_krwmax, mean_swc, mean_sor)    
            
            st.pyplot(fig)
            cols = st.columns(3)
            cols[1].table(cory_exponents)

                        