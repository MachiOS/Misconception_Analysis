# Splitting the text into list elements based on numbered points
import re
import pandas as pd


# Function to split text into list elements
def split_text(text):
    elements = re.split(r'\n\n\d+\. ', text)
    return [re.sub(r'^\d+\.\s*', '', element.strip()) for element in elements if element]

# Function to extract response and explanation
def extract_response_explanation(text):
    # response_match = re.search(r'\*\*Response:\*\*\s*"?(.*?)"?\s*(?:\n\s*-\s*)?\*(?:\*Explanation:\*\*|\*Why it\'s wrong:\*\*)', text, re.DOTALL)
    # explanation_match = re.search(r'\*(?:\*Explanation:\*\*|\*Why it\'s wrong:\*\*)\s*(.*)', text, re.DOTALL)
    # response_match = re.search(r'\*\*Response:?\*\*\s*"?(.*?)"?\s*(?:\n\s*-\s*)?\*(?:\*Explanation:\*\*|\*Why it\'s wrong:\*\*)', text, re.DOTALL)
    # explanation_match = re.search(r'\*(?:\*Explanation:\*\*|\*Why it\'s wrong:\*\*)\s*(.*)', text, re.DOTALL)
    
    # response = response_match.group(1) if response_match else ''
    # explanation = explanation_match.group(1) if explanation_match else ''
    # print(explanation.strip())
    match = re.search(
        r'(?i)\*\*Response[:\s]*\**["“”]?(.*?)[.?!]?\s*["“”]?\**\n\s*-\s*\*\*(Explanation|Why it is wrong)[:\s]*\**(.*)', 
        text, 
        re.IGNORECASE | re.DOTALL
    )
    if match:
        response = match.group(1).strip()
        explanation = match.group(3).strip()
        print(response, explanation)
        return response, explanation
    
    return None, None
    

# def extract_response_explanation(text):
#     response_match = re.search(r'\*\*Response:\*\*\s*"?(.*?)"?\s*(?:\n\s*-\s*)?\*(?:\*Explanation:\*\*|\*Why it\'s wrong:\*\*)', text, re.DOTALL)
#     explanation_match = re.search(r'\*(?:\*Explanation:\*\*|\*Why it\'s wrong:\*\*)\s*(.*)', text, re.DOTALL)
#     response = response_match.group(1).strip() if response_match else ''
#     explanation = explanation_match.group(1).strip() if explanation_match else ''
#     return response, explanation
    
    
# Expanding the data by creating a new row for each split output
df = pd.read_csv('/Users/machi/Documents/GitHub/Misconception_Analysis/OLI_TransactionData_Analyze/Generated_Data/Betsune/gpt-4o-mini/gen_wrong_ans/gpt-4o-mini_Betsune_gen_wrong_ans_thld_av_10_10.csv')
expanded_rows = []
for _, row in df.iterrows():
    split_data = split_text(row['gpt-4o-mini_gen_wrong_ans'])
    for item in split_data:
        new_row = row.copy()
        new_row['structured_gpt-4o-mini_gen_wrong_ans'] = item
        response, explanation =  extract_response_explanation(item)
        new_row['Response'] = response
        new_row['Why it is wrong'] = explanation
        expanded_rows.append(new_row)

# Creating new DataFrame
expanded_df = pd.DataFrame(expanded_rows)
output_csv = '/Users/machi/Documents/GitHub/Misconception_Analysis/OLI_TransactionData_Analyze/Generated_Data/Betsune/gpt-4o-mini/gen_wrong_ans/gpt-4o-mini_Betsune_gen_wrong_ans_thld_av_10_10_restructured.csv'
expanded_df.to_csv(output_csv)
