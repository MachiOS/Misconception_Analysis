import pandas as pd
import re
# import numpy as np



# import os
# import matplotlib.pyplot as plt


def merge_csv_files(df_source_w_index, df_transact_w_index, out_path):
   
    df_source_w_index = df_source_w_index.rename(columns={"Assessment ID": "Problem Name"})
    # Merge the files on "Problem Name" and "Step_index" using an inner join
    merged_df = pd.merge(df_source_w_index, df_transact_w_index, on=["Problem Name", "step_index"], how="right")

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(out_path+'problem_data_all_w_explanation.csv', index=False)

    print(f"Merged file saved as '{out_path+'problem_data_with_explanation.csv'}'")

    return


def merge_by_extracted_qid(source_df, transact_df, out_path):
    
    source_df = source_df.rename(columns={"Assessment ID": "Problem Name"})
    
    source_df  = source_df.drop_duplicates(subset=["Problem Name", "extracted_qid"])
    merged_df = pd.merge(source_df, transact_df, on=["Problem Name", "extracted_qid"], how="right")
    
    merged_df.to_csv(out_path+'problem_data_all_w_explanation_qid_match.csv', index=False)
     
    print(f"Merged file saved as '{out_path+'problem_data_with_explanation_qid_match.csv'}'")

    return
    
    
def merge_problem_data_multiple_courses(problem_df, mapping_df, out_path):
    merged_dfs = []
    
    visited_course_version = []
    # Iterate over each row in the mapping DataFrame
    for i, row in mapping_df.iterrows():
        
        # dataset_name = row['dataset']
        course_version = row['Course_version']
        if course_version in visited_course_version:
            print('Visited')
            continue
        
        visited_course_version.append(course_version)
     
        
        # Load the corresponding child CSV
        source_short_ans_file = '/Users/machi/Desktop/OLI_data_w_quesID_{0}/OLI_pre_process/Output_Data/Output_Files_QUADL/short_answer.csv'.format(course_version)
        source_df = pd.read_csv(source_short_ans_file)
        
        # sometimes it has duplicated data with different unit anme and module name
        
        
        source_df = source_df.rename(columns={"Assessment ID": "Problem Name"})
        source_df = source_df.drop_duplicates(subset=['Problem Name','Step ID'])
        source_df['extracted_qid'] =  source_df['Step ID'].apply(extract_stepid)
        
        
        # Merge the source and child CSV based on the keys "problem name" and "Step ID"
        # merged_df = problem_df[problem_df['dataset'] == dataset_name].merge(
        #     source_df,
        #     on=["Problem Name", "extracted_qid"],
        #     how="left"
        # )
        
        merged_df = problem_df[problem_df['Course_version'] == course_version].merge(
            source_df,
            on=["Problem Name", "extracted_qid"],
            how="left"
        )
        
        # Append the merged DataFrame to the list
        merged_dfs.append(merged_df)
    
    final_df = pd.concat(merged_dfs, ignore_index=True)
    
    final_df.to_csv(out_path+'problem_data_all_w_explanation_qid_match_courseVersion.csv', index=False)
    
    return

def merge_by_extracted_qid_from_multiple_courses(transact_df, mapping_df, out_path):

    merged_dfs = []
    
    visited_course_version = []
    
    
    # Iterate over each row in the mapping DataFrame
    for i, row in mapping_df.iterrows():
        # if i == 1:break
        dataset_name = row['dataset']
        course_version = row['Course_version']
        
        if course_version in visited_course_version:
            print('Visited')
            continue
        
        visited_course_version.append(course_version)
        
        # Load the corresponding child CSV
        source_short_ans_file = '/Users/machi/Desktop/OLI_data_w_quesID_{0}/OLI_pre_process/Output_Data/Output_Files_QUADL/short_answer.csv'.format(course_version)
        source_df = pd.read_csv(source_short_ans_file)
        
        source_df = source_df.rename(columns={"Assessment ID": "Problem Name"})
        source_df = source_df.drop_duplicates(subset=['Problem Name','Step ID'])
        source_df['extracted_qid'] =  source_df['Step ID'].apply(extract_stepid)
        
        # Merge the source and child CSV based on the keys "problem name" and "Step ID"
        
        merged_df = transact_df[transact_df['Course_version'] == course_version].merge(
            source_df,
            on=["Problem Name", "extracted_qid"],
            how="left"
        )
        
        # Append the merged DataFrame to the list
        merged_dfs.append(merged_df)
    
    final_df = pd.concat(merged_dfs, ignore_index=True)
    
    final_df.to_csv(out_path+'short_ans_transact_w_explanation_qid_match.csv', index=False)
    
    return
     
     

def assign_step_index(source_df, out_path):
    # Sort the short_answer.csv by Problem.csv
    
    df_sorted = source_df.sort_values(by=['Assessment ID','Step ID'], ascending=[True, True]).reset_index(drop=True)
    

    df_sorted['step_index'] = df_sorted.groupby('Assessment ID').cumcount()
    
    df_sorted.to_csv(out_path+'short_answer_w_index.csv', index=False)
    


def extract_stepid(text):
    '''
    If there is no match just leave the step ID empty, (in this case it is very likely that Problem Name is sufficient to match)
    '''
    
    if pd.isna(text):  # Handle missing (NaN) values in "Input"
        return text
    else: 
        match = re.search(r'q[1-9]', text)
        if match:
            return match.group()
        else:
            if len(text) > 30:
                return text.split("_",1)[0]
            # return text
   
    
    
    

    
    
    

def unit_module_problems(df,out_path):
    selected_columns = ['Problem Name', 'Step Name', 'Selection', 'Level (Unit)', 'Level (Module)', 'CF (oli:resourceType)','CF (oli:purpose)']

    # Remove duplicates based on the selected columns
    # df_no_duplicates = df[selected_columns].drop_duplicates()
    
    df = df[selected_columns]
    df_no_duplicates = df.drop_duplicates(subset=['Problem Name','Step Name'])

    df_no_duplicates = df_no_duplicates.sort_values(by=['Problem Name','Step Name'], ascending=[True, True])
    # Save the resulting DataFrame to a new CSV file
    
    
    df_no_duplicates = df_no_duplicates.reset_index(drop=True)
    
    # Group by 'problem name' and assign an incremental index to each 'step name'
    df_no_duplicates['step_index'] = df_no_duplicates.groupby('Problem Name').cumcount()

    
    df_no_duplicates.to_csv(out_path+'problem_data_all.csv', index=False)

    return


def main(args_dict):
    
    '''
    # If Betshune data or OLI data where all semester uses the same version()
    '''
    # df = pd.read_csv(args_dict['input_file'])
    # unit_module_problems(df, args_dict['out_path'])
    
    # Run only once.
    source_df = pd.read_csv(args_dict['source_short_ans_file'])
    
    source_df['extracted_qid'] =  source_df['Step ID'].apply(extract_stepid)
    
    df = pd.read_csv(args_dict['problem_data'])
    df['extracted_qid'] = df['Selection'].apply(extract_stepid)
    
    merge_by_extracted_qid(source_df, df, args_dict['out_path'])
    # # df_source_w_index = pd.read_csv(args_dict['source_out_path']+'short_answer_w_index.csv')
    # df_transact_w_index = pd.read_csv(args_dict['out_path']+'problem_data_all.csv')


    '''
    Heartland Community College
    '''
 
    # df = pd.read_csv(args_dict['input_file'])
    # df['extracted_qid'] = df['Selection'].apply(extract_stepid)
    
    # mapping_df = pd.read_csv(args_dict['mapping_dataset_and_course_ver'])
    
    # Problem data + Expalantion
    # problem_df = pd.read_csv(args_dict['problem_data'])
    # problem_df['extracted_qid'] = problem_df['Selection'].apply(extract_stepid)
    # merge_problem_data_multiple_courses(problem_df, mapping_df,  args_dict['out_path'])
    
    # No longer needed or not tested
      # print(mapping_df)
    # Transaction data with Explanation
    # merge_by_extracted_qid_from_multiple_courses(df, mapping_df, args_dict['out_path'])
    
    
    
    
if __name__ == "__main__":
    
    args_dict = {
        # 'input_file':'/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/all_shortAns_concat/Betshune_ShortAns.csv',
        # 'input_file': '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/all_shortAns_concat/Heartland_ShortAns.csv',
        # 'input_file': '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/shortAns_filtered/Heartland_ShortAns_filtered.csv',
        'input_file': '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/all_shortAns_concat/OLI_ShortAns.csv',
        
        # 'out_path': '/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/',
        # 'out_path':'/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/',  
        'out_path': '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/',

        
        # 'problem_data': "/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/stat/problem_data_all.csv",
        # 'problem_data': '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/stat/problem_data_course_version.csv',
        'problem_data': "/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/stat/problem_data_all.csv",
        
     
        'source_short_ans_file': '/Users/machi/Desktop/OLI_data_w_quesID_to_csv/OLI_pre_process/Output_Data/Output_Files_QUADL/short_answer.csv',
        # 'source_out_path': '/Users/machi/Desktop/OLI_data/OLI_pre_process/Output_Data/Output_Files_QUADL/',
        
        'mapping_dataset_and_course_ver': '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/course_version_mapping_data.csv'

    }
    
    main(args_dict)
    
    
    