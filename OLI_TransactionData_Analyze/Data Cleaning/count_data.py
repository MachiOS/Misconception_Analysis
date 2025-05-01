'''
Show some statistics of the students data, e.g., number of words in the response.e.t.c.,
'''
import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt

# from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib.ticker import MaxNLocator
# from matplotlib.ticker import MultipleLocator



# Function to dynamically adjust bin size
def adjust_bin_size(max_word_count, initial_bin_size=5, max_bins=10):
    # Calculate the number of bins with the initial bin size
    num_bins = (max_word_count // initial_bin_size) + 1
    
    # If the number of bins exceeds the desired max_bins, increase the bin size
    if num_bins > max_bins:
        # Calculate a new bin size to limit the number of bins
        new_bin_size = (max_word_count // max_bins) + 1
        return new_bin_size
    else:
        return initial_bin_size

def plot_dist(df,out_path):
    
    df['Word Count'] = df['Input'].apply(count_words)

    # Group by 'Problem Name' and 'Step Name' and get the word counts
    grouped = df.groupby(['Problem Name', 'Step Name'])['Word Count'].apply(list).reset_index()

    # Create a PDF file to save all plots
    with PdfPages(out_path+'word_count_distribution.pdf') as pdf:
        # Iterate through each group to create individual plots
        for i, row in grouped.iterrows():
            plt.figure(figsize=(12, 8))
            
            # Define bins with a step size of 5
            max_word_count = max(row['Word Count'])
            
            bin_size = adjust_bin_size(max_word_count, initial_bin_size=5, max_bins=10)
            # bin_size = 5 # if you want to fix the binsize 
            bins = np.arange(0, max_word_count + bin_size, bin_size)
            
            plt.hist(row['Word Count'], bins=bins, alpha=0.5, rwidth=0.8)
            plt.title(f'Distribution of Word Counts in Input: {row["Problem Name"]} - {row["Step Name"]}')
            plt.xlabel('Number of Words')
            plt.ylabel('Frequency')
            # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            
      
            # Set y-axis ticks to align with intervals of 5
            plt.xlim(left=0)
            plt.gca().xaxis.set_major_locator(MultipleLocator(bin_size))
            # plt.gca().yaxis.set_major_locator(MultipleLocator(5))
            plt.grid(True)

            # Save the current plot to the PDF
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()    # Close the figure to free up memory
    
    return
    
    

def count_responses(df,out_path):
    
    df['Word Count'] = df['Input'].apply(count_words)
    # Group by "Problem Name" and "Step Name" and calculate the metrics
    grouped_stats = df.groupby(['Problem Name', 'Step Name']).agg(
        row_count=('Word Count', 'size'),                        # Number of rows
        avg_word_count=('Word Count', 'mean'),                   # Average number of words
        std_word_count=('Word Count', 'std')                     # Standard deviation of word count
    ).reset_index()

    # Fill NaN values in std_word_count with 0, in case there's only one row and std is NaN
    grouped_stats['std_word_count'].fillna(0, inplace=True)
    
    grouped_stats.to_csv(out_path+'problem_step_word_stats.csv', index=False)

    print("The result has been saved to 'problem_step_word_stats.csv'.")
    
    

def count_words(text):
    if pd.isna(text):  # Handle missing (NaN) values in "Input"
        return 0
    return len(text.split())


def count_problems(df,out_path):
    
    grouped_counts = df.groupby(['Problem Name', 'Step Name']).size().reset_index(name='Count_Responses')
    grouped_counts.to_csv(out_path + 'problem_step_counts.csv', index=False)
    print("The result has been saved to 'problem_step_counts.csv'.")
    
    return

def unit_module_problems(df,out_path):
    selected_columns = ['Problem Name', 'Step Name', 'Selection', 'Level (Unit)', 'Level (Module)', 'CF (oli:resourceType)','CF (oli:purpose)','dataset']
    

    # Remove duplicates based on the selected columns
    df_no_duplicates = df[selected_columns].drop_duplicates(subset=['Problem Name', 'Step Name', 'Selection','dataset'])


    df_no_duplicates = df_no_duplicates.sort_values(by=['Level (Unit)', 'Level (Module)'], ascending=[True, True])
    # Save the resulting DataFrame to a new CSV file
    
    # We want to avoid this since there are some cases where Module name is different for the same problem.
    # df_no_duplicates.to_csv(out_path+'problem_data.csv', index=False)
    
    df_no_duplicates.to_csv(out_path+'problem_data_Problem_Step_Selection_Dataset.csv', index=False)
    
    
    return

def merge_same_courseversion(mapping_course_csv, problem_csv, out_path):
    
    mapping_df = pd.read_csv(mapping_course_csv)
    problem_df = pd.read_csv(problem_csv)
    
    merged_df = pd.merge(problem_df, mapping_df, on='dataset', how='left')
    
    
    merged_df = merged_df.drop_duplicates(subset=['Problem Name', 'Step Name','Selection', 'Course_version' ])
    merged_df = merged_df.drop(columns=['dataset'])
    
    merged_df.to_csv(out_path+'problem_data_course_version2.csv',index=False)
    
    
    return
    



def compute_averages(df, out_path):
   
    # Filter rows where row_count > 50
    filtered_df = df[df['row_count'] > 50]
    
    if not filtered_df.empty:
        # Compute the average of row_count and avg_word_count
        avg_row_count = filtered_df['row_count'].mean()
        avg_word_count = filtered_df['avg_word_count'].mean()
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
    
    # df = pd.read_csv(args_dict['input_file'])
    
    # unit_module_problems(df, args_dict['out_path'])
    
    problem_csv = args_dict['out_path']+ 'problem_data_Problem_Step_Selection_Dataset.csv'
    merge_same_courseversion(args_dict['mapping_course_csv'], problem_csv, args_dict['out_path'])
    
    # count_problems(df, args_dict['out_path'])
    
    # count_responses(df,args_dict['out_path'])
    
    
    # plot_dist(df, args_dict['out_path'])
    
    
    # df = pd.read_csv(args_dict['out_path']+'problem_step_word_stats.csv')
    # compute_averages(df, args_dict['out_path'])
    
    
    
   

if __name__ == "__main__":
    
    args_dict = {
        # 'input_file':'/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/shortAns_filtered/Betshune_ShortAns_filtered.csv',
        # 'out_path': '/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/stat/'
       
        
        'input_file':'/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/shortAns_filtered/Heartland_ShortAns_filtered.csv',
        'out_path': '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/stat/',
        'mapping_course_csv': '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/course_version_mapping_data.csv',
        
        # 'input_file':'/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/shortAns_filtered/OLI_ShortAns_filtered.csv',
        # 'out_path': '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/stat/'
        
    }
    
    if not(os.path.exists(args_dict['out_path'])):
        os.makedirs(args_dict['out_path'])
    
    
    main(args_dict)


    
    
    
    