import pandas as pd
import re

'''
Remove
- .
- Single Word
-(blank)
- contains only number



Exceptions:
- For each (student and StepName), if there are multiple answeres, take the longer one?

'''


def is_float(value):
    try:
        return isinstance(float(value), float)
    except ValueError:
        return False



def filter_condition(text):
    # Remove leading/trailing whitespaces
    text = text.strip()

    # Check for blank text
    if not text:
        return False

    if len(text.split()) == 1:
        return False

    # Check if the text contains only non-word characters
    if re.match(r"^[^\w\s]+$", text):  # Checks if only non-word characters (no alphanumerics or spaces)
        return False


    return True

def remove_duplicate(df):
    
    # Sort the DataFrame by "StudentID", "StepName", and the length of "Input" in descending order
    df['InputLength'] = df['Input'].str.len()  # Create a column for length of "Input"
    
    df_sorted = df.sort_values(by=['dataset','Anon Student Id', 'Problem Name', 'Step Name', 'InputLength'], ascending=[True, True, True, True, False])

    # Drop duplicate rows based on "StudentID" and "StepName", keeping the first (longest "Input")
    df_deduplicated = df_sorted.drop_duplicates(subset=['Anon Student Id', 'Problem Name', 'Step Name'], keep='first')

    # Drop the "InputLength" column as it's no longer needed
    df_deduplicated = df_deduplicated.drop(columns=['InputLength'])
    
    #resort the data
    df_final_sorted = df_deduplicated.sort_values(by=['dataset', 'Row'], ascending=[True, True])

    return df_final_sorted

def main(args_dict):
    
    df = pd.read_csv(args_dict['input_file'])
    
    # Remove rows where the 'Input' column contains float values
    df_text= df[~df['Input'].apply(is_float)]
   
    df_filtered = df_text[df_text['Input'].apply(filter_condition)]
    
    df_final_sorted = remove_duplicate(df_filtered)

    df_final_sorted .to_csv(args_dict['output_file_path'], index=False)

    print(f"Filtered data saved to {args_dict['output_file_path']}")
    
    return


if __name__ == "__main__":
    
    args_dict = {
        # 'input_file': '/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/all_shortAns_concat/Betshune_ShortAns.csv',
        # 'input_file': '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/all_shortAns_concat/Heartland_ShortAns.csv',
        'input_file': '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/all_shortAns_concat/OLI_ShortAns.csv',
        
        # 'output_file_path':'/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/Betshune_ShortAns_filtered.csv'  # Define the path for output file,
        # 'output_file_path':'/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/Heartland_ShortAns_filtered.csv'  # Define the path for output file
        'output_file_path': '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/shortAns_filtered/OLI_ShortAns_filtered.csv',

    }
    
    main(args_dict)



