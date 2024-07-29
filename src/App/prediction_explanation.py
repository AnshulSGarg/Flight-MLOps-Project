import pickle as pk
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pathlib

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.parent
raw_path = home_dir.as_posix() 
pickle_path = raw_path  + r'/pickle_files/'


def explain():

    file_path = pickle_path + 'flight_pipeline.pkl'
    with open(file_path, 'rb') as file:
        pipeline = pk.load(file)

    file_path = pickle_path + 'one_df.pkl'
    with open(file_path, 'rb') as file:
        one_df = pk.load(file)    


    file_path = pickle_path + 'predicted_price.pkl'
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

    i = 0
    # Create a SHAP Explanation object for the selected sample
    explanation = shap.Explanation(
        values=np.round(array_df.iloc[0].values, decimals=0),
        base_values=np.round(np.expm1(shap_values.base_values[i]), decimals=0),
        data=result_df.iloc[i].values,
        feature_names=result_df.columns
    )   

    # Find the top feature and its value
    top_feature_index = np.argmax(np.abs(explanation.values))
    top_feature_name = explanation.feature_names[top_feature_index]
    top_feature_data = explanation.data[top_feature_index]
    top_feature_value = explanation.values[top_feature_index]
    sum_shap_values = explanation.base_values + np.sum(explanation.values)
    pcent = round(top_feature_value*100/sum_shap_values)

    st.header("Understand Price Prediction")

    if top_feature_value > 0:
        signal = "increasing"
    else:
        signal = "decreasing"
 
    st.write(f"""
    Below plot can help understand how different selections contribute towards flight price prediction.
    - Here, E[f(X)] represents average price of the flights.
    - f(X) is the predicted price based on the selections.
    - {top_feature_name} is contributing {pcent}% towards {signal} the price
             """)



    # Create a waterfall plot for the selected sample
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation, show=False)
    plt.grid(False)
    ax.grid(False)
    
    # Set the background to be transparent
    fig.patch.set_facecolor('none')
    ax.patch.set_facecolor('none')


    # Customize the plot
    plt.gca().xaxis.label.set_color('white')   # X-axis label color
    plt.gca().yaxis.label.set_color('white')   # Y-axis label color
    plt.gca().tick_params(axis='x', colors='white')  # X-axis tick color
    plt.gca().tick_params(axis='y', colors='white')  # Y-axis tick color
    plt.gca().title.set_color('white')         # Title color


    # Set the color and size for text elements
    for text in ax.get_xticklabels():
        text.set_color('white')
    for text in ax.get_yticklabels():
        text.set_color('white')

    ax.spines['bottom'].set_color('white')  # X-axis line color
    ax.spines['bottom'].set_linewidth(1) 


    
    plt.savefig(raw_path + r'\src\visualization\shapely_plots\explain_waterfall.png', bbox_inches='tight')
    plt.close()

    # Display the saved image in Streamlit
    st.image(raw_path + r'\src\visualization\shapely_plots\explain_waterfall.png')
    



if __name__ == '__main__':
    explain()