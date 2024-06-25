import pickle as pk
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def explain():

    file_path = r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\pickle_files\flight_pipeline.pkl'
    with open(file_path, 'rb') as file:
        pipeline = pk.load(file)

    file_path = r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\pickle_files\one_df.pkl'
    with open(file_path, 'rb') as file:
        one_df = pk.load(file)

    file_path = r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\pickle_files\predicted_price.pkl'
    with open(file_path, 'rb') as file:
        predicted_price = pk.load(file)
    # st.text(f'predicted_price is {predicted_price[0]}')

    preprocessor = pipeline.named_steps['preprocessor']
    new_data_point_transformed = preprocessor.transform(one_df)
    new_data_point_dense = new_data_point_transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()

    new_data_point_df = pd.DataFrame(new_data_point_dense, columns=feature_names)
    # st.text(new_data_point_df)




    explainer = shap.TreeExplainer(pipeline.named_steps['regressor'])
    shap_values = explainer(new_data_point_df)

    array = shap_values.values
    base = np.expm1(shap_values.base_values[0])
    lst = []
    for i in array[0]:
        val = np.expm1(i) * (base + 1)
        lst.append(val)
        base_val = val + base
        base = base_val

    scaler = preprocessor.named_transformers_['num']
    new_data_point_df.iloc[:, :3] = scaler.inverse_transform(new_data_point_df.iloc[:, :3])
    # st.text(new_data_point_df)

    
    array_list = np.array(lst)

    result_df = pd.DataFrame([new_data_point_df.iloc[0].values], columns=new_data_point_df.columns)
    # st.text(result_df)


    result_df.columns = result_df.columns.str.replace("num__", "")
    result_df.columns = result_df.columns.str.replace("cat__", "")
    column_array = result_df.columns

    def extract_carrier_name(row, txt):
        for col in result_df.columns:
            if col.startswith(txt) and row[col] == 1:
                return col.replace(txt, '')
        return None

    prefixes = ['carrier_', 'Trip_Type_', 'Airport_Route_', 'stop_', 'Holiday_', 'from_hour_', 'Fly_WeekDay_']

    for prefix in prefixes:
        # Extract the corresponding j from prefix
        j = prefix.replace('_', '')    
        # Apply the function to each row and create the new column
        result_df[j] = result_df.apply(lambda row: extract_carrier_name(row, prefix), axis=1)
        
    # Drop columns starting with 'carrier_'
        result_df = result_df[result_df.columns.drop(list(result_df.filter(regex=prefix)))]

    # st.text(result_df)

    val = np.array(array_list)
    array_df = pd.DataFrame([val], columns=column_array)

    # Function to aggregate values based on prefixes
    def aggregate_by_prefix(df, prefixes):
        aggregated_data = {}
        for prefix in prefixes:
            matching_columns = [col for col in df.columns if col.startswith(prefix)]
            if matching_columns:
                aggregated_data[prefix.rstrip('_')] = df[matching_columns].sum(axis=1).values[0]
        return aggregated_data

    # Aggregate the values
    aggregated_values = aggregate_by_prefix(array_df, prefixes)

    # Create the new DataFrame
    aggregated_df = pd.DataFrame([aggregated_values])

    # Display the new DataFrame
    array_df = pd.concat([array_df[['round_trip_duration','Days_to_Fly','flight_duration_value']],aggregated_df], axis=1)
    # st.text(array_df)    

    # i = 0
    # # Create a SHAP Explanation object for the selected sample
    # explanation = shap.Explanation(
    #     values=np.array(lst),
    #     base_values=np.expm1(shap_values.base_values[i]),
    #     data=new_data_point_df.iloc[i].values,
    #     feature_names=new_data_point_df.columns
    # )

    i = 0
    # Create a SHAP Explanation object for the selected sample
    explanation = shap.Explanation(
        values=np.round(array_df.iloc[0].values, decimals=0),
        base_values=np.round(np.expm1(shap_values.base_values[i]), decimals=0),
        data=result_df.iloc[i].values,
        feature_names=result_df.columns
    )

    

    # Create a waterfall plot for the selected sample
    shap.plots.waterfall(explanation)
    


    
    plt.savefig(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\shapely_plots\explain_waterfall.png', bbox_inches='tight')
    plt.close()

    # Display the saved image in Streamlit
    st.image(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\shapely_plots\explain_waterfall.png')
    



if __name__ == '__main__':
    explain()