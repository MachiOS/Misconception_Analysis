'''
Concat data
'''

import pandas as pd
import os



def get_average_response_concat(args_dict):
    # df = pd.read_csv(args_dict['input_file'])
    df1 = pd.read_csv(args_dict['input_file_1'])
    df2 = pd.read_csv(args_dict['input_file_2'])
    df3 = pd.read_csv(args_dict['input_file_3'])

    # Combine the DataFrames
    combined_df = pd.concat([df1, df2, df3])

    # Group by "Problem Name" and "Step Name" and sum the "Count_Response"
    result_df = combined_df.groupby(['Problem Name', 'Step Name'], as_index=False).agg(
        Count_Response=('row_count', 'sum'),        # Sum of Count_Response
        Avg_Word_Count=('avg_word_count', 'mean')
    )

    # Save the result to a new CSV file
    result_df.to_csv(args_dict['out_path']+'problem_step_word_stat_combined.csv', index=False)
    
    print("Data combined and saved to problem_step_word_stat_combined.csv")
    

def compute_averages(df, out_path):
   
    # Filter rows where row_count > 50
    filtered_df = df[df['Count_Response'] > 50]
    
    if not filtered_df.empty:
        # Compute the average of row_count and avg_word_count
        avg_row_count = filtered_df['Count_Response'].mean()
        avg_word_count = filtered_df['Avg_Word_Count'].mean()
        count = filtered_df.shape[0]
    else:
        avg_row_count = avg_word_count = 0
        count = 0

    result_df = pd.DataFrame({
        'average_row_count_larger_than_50': [avg_row_count],
        'average_word_count': [avg_word_count],
        'num_problems': [count]
    })
    result_df.to_csv(out_path+'average_responses_overall.csv', index=False)

    print("Result saved")
    

def main(args_dict):
    # get_average_response_concat(args_dict)
    
    df = pd.read_csv(args_dict['out_path']+'problem_step_word_stat_combined.csv')
    compute_averages(df, args_dict['out_path'])
    
    
    
if __name__ == "__main__":
    
    
    args_dict = {
        'input_file_1':'/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/stat/problem_step_word_stats.csv',
       
        
        'input_file_2':'/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/stat/problem_step_word_stats.csv',
        
        'input_file_3':'/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/stat/problem_step_word_stats.csv',
        
        'out_path': '/Users/machi/Desktop/OLI Biology Data _ALL/Concat/'
        
    }
    
    if not(os.path.exists(args_dict['out_path'])):
        os.makedirs(args_dict['out_path'])
    
    
    main(args_dict)
    