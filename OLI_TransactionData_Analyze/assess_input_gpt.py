import pandas as pd

from openai import OpenAI, AsyncOpenAI

import math, re
import nest_asyncio, asyncio
import pickle, time

# from datasets import load_dataset, concatenate_datasets
import json

import os
import collections

import sys


from dotenv import load_dotenv, dotenv_values 
load_dotenv(override=True)

'''
Access students responses to short answer questions.
'''

client_async = AsyncOpenAI(
    
    api_key=os.getenv("API_KEY")

)

task_instruction = '''
Given a question and answer, you need to assess a student's input to the question. It is one of the following:"Correct", "Partically Correct", "Incorrect". 
If it is not correct, specify which part of the student input is wrong.
'''

task_instruction_pprim = '''
P-prims are fundamental elements of intuitive knowledge that originate in the learner's intuitive interpretations experience. 
In Physics, for example, when experiencing physical phenomena, a student might develop P-prims like: 'Increased efforts generate greater results' or 'Force is exerted only on objects being pushed by an agent whether or not the object is moving.'
P-prims is basically a misconception but could be sometimes true depending on the context. Students learn correct concepts by refining their P-prims.
Given a question, gold answer, and a student's input, the goal is to identify what kind of P-prims the student might have regarding the topic of the question.
If the student's input is correct, just say "correct input". If it is partially correct or incorrect, list the P-prims.
'''

task_instruction_gen_wrong_response = '''
Given a question and answer, generate five wrong responses to the question that show misunderstandings the students could have. 
It doesn't always have to be directly related to the concept asked in the question. It can be about any related topic. 
For each response, explain why it is wrong.
'''

task_instruction_meta_knowledge = '''
Given a question about a concept(s), list 5 common misconceptions that students might have about various related concepts.
'''

async def get_async_gpt4(instruction,text, model):

    response = await client_async.chat.completions.create(
            model=model,
            messages=[
            
                {"role": "system", "content":  instruction},
                {"role": "user", "content": text}
            ]
            )
    # return response.choices[0].message.content
    return response.choices[0].message.content


def apply_async_get_embedding_gpt4(lst, model_name, version=None):
    loop = asyncio.get_event_loop()
    if version=='p-prim':
        tasks = [loop.create_task(get_async_gpt4(task_instruction_pprim, prompt, model_name)) for prompt in lst]
    
    elif version=='gen_wrong_ans':
        tasks = [loop.create_task(get_async_gpt4(task_instruction_gen_wrong_response, prompt, model_name)) for prompt in lst]
    elif version=='meta_knowledge':
        tasks = [loop.create_task(get_async_gpt4(task_instruction_meta_knowledge, prompt, model_name)) for prompt in lst]
    elif version=='assess':
        tasks = [loop.create_task(get_async_gpt4(task_instruction, prompt, model_name)) for prompt in lst]
    
    return loop.run_until_complete(asyncio.gather(*tasks))


def run_all_async(num_batches, batch_size, prompts_list, fname, model_name, path, max_tokens=None, version=None, ):
    all_text_set = []
    for i in range(math.ceil(num_batches)):
       
        gen_text_split =  prompts_list[batch_size*i:batch_size*i+batch_size]
        print("Starting_split: ", i)
        
        cur_gen_text = apply_async_get_embedding_gpt4(gen_text_split, model_name, version)
            
        print("Done_split: ", i)

        with open(path + fname+"_"+str(i)+".pkl", 'wb') as f:
            pickle.dump(cur_gen_text, f)
        print("File written: ", i)
        
        all_text_set.append(cur_gen_text)
        print("Sleeping..")
        time.sleep(70)
        print("Slept")

    return all_text_set


def format_input_output(question, answer, student_input, zero_shot=True):
    '''
    format should be same as few shot examples
    student_response_text = None when you don't use student response data
    '''

    prompt = f'''
    Question: {question}
    Answer: {answer}
    Student Input: {student_input}
    Correctness: [Correct or Partially Correct or Incorrect]
    Explanation: [Explain which part of the student input is wrong when it is not correct.]
    
    '''
    
    return prompt


def format_input_output_pprim(question, answer, student_input, zero_shot=True):
    '''
    format should be same as few shot examples
    student_response_text = None when you don't use student response data
    '''

    prompt = f'''
    Question: {question}
    Answer: {answer}
    Student Input: {student_input}
    P-prims:[write P-prims that the student might have when Student Input is partically Correct or incorrect. P-prims must be separated by ";", If the Student Input is correct, just say correct.]
    
    '''
    
    return prompt


def format_input_output_gen_wrong_ans(question, answer, zero_shot=True):
    
    prompt = f'''
    Question: {question}
    Answer: {answer}
    Expected incorrect responses: [Generate 5 responses that shows incorrect understanding on any related concept to the question, and describe why it is wrong.]
    '''
    
    return prompt

def format_input_output_meta_knowledge(question, zero_shot=True):
    
    prompt = f'''
    Question: {question}
    Common Misconceptions: [List 5 common misconceptions students might have about related concepts to the given question.]
    '''
    
    return prompt

def get_student_input(problem_name, step_name, df, course_version=None):
    
    if course_version == None:
        target_input_df = df[(df['Problem Name']==problem_name) & (df['Step Name']==step_name)].reset_index()
    else:
        target_input_df = df[(df['Problem Name']==problem_name) & (df['Step Name']==step_name) & (df['Course_version']==course_version)].reset_index()
        # target_input_df = df[(df['Problem Name']==problem_name) & (df['Step Name']==step_name)].reset_index()

   
    
    target_input_df['len_answer'] = target_input_df['Input'].apply(count_words)
    df_sorted = target_input_df.sort_values(by=['len_answer'],ascending=[False]).reset_index()
    
    input_list = df_sorted['Input'].tolist()
    row_idx_list = df_sorted['Row'].tolist()
    
    if course_version == None:
        return input_list, row_idx_list

    else:
        dataset_list = df_sorted['dataset'].tolist()
        return input_list, row_idx_list, dataset_list
        



def count_words(text):
    if pd.isna(text):  # Handle missing (NaN) values in "Input"
        return 0
    return len(text.split())

def sort_by_length_of_answer(df):
    
    df['len_answer'] =  df['Explanation'].apply(count_words)
    
    df_sorted = df.sort_values(by=['len_answer'], ascending=[False]).reset_index()
    
    
    return df_sorted

def sort_by_average_length_of_input(df, with_ds_version=False):
    
    #Calculate the length of each "Input" entry and store it in a new column
    df['Input Length'] = df['Input'].apply(count_words)

    # Calculate the average input length for each ("Problem Name", "Step Name") pair
    if with_ds_version:
        avg_length = df.groupby(['Problem Name', 'Step Name', 'dataset'])['Input Length'].mean().reset_index()
    else:
        avg_length = df.groupby(['Problem Name', 'Step Name'])['Input Length'].mean().reset_index()
        
        
        
    avg_length = avg_length.rename(columns={'Input Length': 'Average Input Length'})

    # Sort the unique pairs by average input length
    sorted_avg_length = avg_length.sort_values(by='Average Input Length', ascending=False)

    # Merge the sorted averages back with the original data to maintain original columns
    
    if with_ds_version:
        sorted_data = df.merge(sorted_avg_length, on=['Problem Name', 'Step Name','dataset']).sort_values(
        by=['Average Input Length', 'Problem Name', 'Step Name','dataset'], ascending=[False, True, True, True]
        )
    else:
        sorted_data = df.merge(sorted_avg_length, on=['Problem Name', 'Step Name']).sort_values(
        by=['Average Input Length', 'Problem Name', 'Step Name'], ascending=[False, True, True]
        )
        
    # Drop the helper columns used for sorting and export
    # sorted_data = sorted_data.drop(columns=['Input Length', 'Average Input Length'])

   
    
    return sorted_data
    
    

def mearge_input_and_sort_multi_course(problem_csv, student_input_csv, dataset_name):
    
    problem_df = pd.read_csv(problem_csv)
    input_df = pd.read_csv(student_input_csv)
        
    df_with_input_sort_by_len_ans = pd.DataFrame(columns=['Row','Unit', 'Module', 'Question','Explanation','Problem Name','Step Name', 'dataset', 'Course_version', 'Input'])
    
    problem_sorted_df = sort_by_length_of_answer(problem_df)

    print('len problem list', len(problem_sorted_df))
    

    for i, row in problem_sorted_df.iterrows():

        problem_name = row['Problem Name']
        step_name = row['Step Name']
        question = row['Question']
        answer = row['Explanation']
        unit = row['Unit']
        module = row['Module']
      
        course_version = row['Course_version']
        
        input_list, row_list, dataset_list = get_student_input(problem_name, step_name, input_df, course_version)
        

        prev_input = ""

        for j, each_input in enumerate(input_list):
            
            # Remove duplcated input
            if prev_input == each_input:
               continue
                
            row_indx = row_list[j]
            ds_version = dataset_list[j]
            
            new_row = {'Row':row_indx,'Unit':unit, 'Module':module, 'Question':question,'Explanation': answer,'Problem Name':problem_name,'Step Name':step_name, 'dataset':ds_version, 'Course_version':course_version, 'Input':each_input}
            
            df_with_input_sort_by_len_ans.loc[len(df_with_input_sort_by_len_ans)] = new_row
        
            prev_input = each_input
    
    
    
    
    sorted_data_path = './sorted_data/' + dataset_name +  '/' 
    if not os.path.exists(sorted_data_path):
        os.makedirs(sorted_data_path)
        
    
    sorted_by_ans_data_csv = sorted_data_path  + dataset_name  + "_sorted_by_ans.csv"
    
    df_with_input_sort_by_len_ans.to_csv(sorted_by_ans_data_csv, index=False)
    
    # sort the data by average number of student input lenght
    
    
    sort_by_input_df = sort_by_average_length_of_input(df_with_input_sort_by_len_ans, with_ds_version=True)
    
    sorted_by_input_data_csv = sorted_data_path  + dataset_name  + "_sorted_by_av_inputlen.csv"
    
    sort_by_input_df.to_csv(sorted_by_input_data_csv, index=False)
    
    
    return sorted_by_input_data_csv


def mearge_input_and_sort(problem_csv, student_input_csv, dataset_name):
    
    problem_df = pd.read_csv(problem_csv)
    input_df = pd.read_csv(student_input_csv)
    

    
    df_with_input_sort_by_len_ans = pd.DataFrame(columns=['Row','Unit', 'Module', 'Question','Explanation','Problem Name','Step Name', 'Input'])
    
    problem_sorted_df = sort_by_length_of_answer(problem_df)

    print('len problem list', len(problem_sorted_df))
    

    for i, row in problem_sorted_df.iterrows():
        problem_name = row['Problem Name']
        step_name = row['Step Name']
        question = row['Question']
        answer = row['Explanation']
        unit = row['Unit']
        module = row['Module']
        
        input_list, row_list = get_student_input(problem_name, step_name, input_df)
            
        prev_input = ""

        for j, each_input in enumerate(input_list):
            
            # Remove duplcated input
            if prev_input == each_input:
               continue
                
            row_indx = row_list[j]
            
            new_row = {'Row':row_indx,'Unit':unit, 'Module':module, 'Question':question,'Explanation': answer,'Problem Name':problem_name,'Step Name':step_name, 'Input':each_input}
            
            df_with_input_sort_by_len_ans.loc[len(df_with_input_sort_by_len_ans)] = new_row
        
            prev_input = each_input
    
    
    
    sorted_data_path = './sorted_data/' + dataset_name +  '/' 
    if not os.path.exists(sorted_data_path):
        os.makedirs(sorted_data_path)
        
    
    sorted_by_ans_data_csv = sorted_data_path  + dataset_name  + "_sorted_by_ans.csv"
    
    df_with_input_sort_by_len_ans.to_csv(sorted_by_ans_data_csv, index=False)
    
    # sort the data by average number of student input lenght
    
    
    sort_by_input_df = sort_by_average_length_of_input(df_with_input_sort_by_len_ans)
    
    sorted_by_input_data_csv = sorted_data_path  + dataset_name  + "_sorted_by_av_inputlen.csv"
    
    sort_by_input_df.to_csv(sorted_by_input_data_csv, index=False)
    
    
    return sorted_by_input_data_csv
    

def run_model(source_data_to_read,model_name, dataset_name, prompt_version, ques_list_to_skip = [],prob_names_skip = [], av_input_len_thld = 0, input_len_thld = 0,
                        max_tokens = None, min_num_words = 10, max_num_words = 300, max_num_words_in_all_responses = 100000, batch_size = 32, ):
    '''
    Run model
    '''
    
  
    df = pd.read_csv(source_data_to_read)
    
    output_df = pd.DataFrame(columns=df.columns)
    len_prompts_list = []
    prompts_list = []
    
    blank_pattern = r'^\s*$'

    for i, row in df.iterrows():
    
        # if i > 2:break
        problem_name = row['Problem Name']
        question = row['Question']
        answer = row['Explanation']
        
        if prompt_version == 'p-prim' or prompt_version == 'access':
            input = row['Input']
            '''
            Skip some problems
            '''
            if average_input_len <= av_input_len_thld:
                continue
        
            if input_len <= input_len_thld:
                continue
        
            average_input_len = row['Average Input Length']
            input_len = row['Input Length']
        
            
        if pd.isna(question) or re.match(blank_pattern, str(question)):
            continue
        
        '''
        Skip some problems
        '''    
        if question in ques_list_to_skip:
            # print('skipped')
            continue
        
        if problem_name in prob_names_skip:
            continue
        
        
        # output_df = output_df.append(row, ignore_index=True)

        if prompt_version == 'p-prim':
            prompt = format_input_output_pprim(question, answer, input)
        elif prompt_version == 'gen_wrong_ans':
            prompt = format_input_output_gen_wrong_ans(question, answer)
        elif prompt_version=='meta_knowledge':
            prompt = format_input_output_meta_knowledge(question)
        elif prompt_version=='access':
            prompt = format_input_output(question, answer, input)
        else:
            print("Please specfiy the pre-defined Prompt Version")
            sys.exit(0)
        # new_row = {'Row':row_indx,'Unit':unit, 'Module':module, 'Question':question,'Explanation': answer,'Problem Name':problem_name,'Step Name':step_name, 'Input':input, 'Input Length':input_length}
        
        output_df.loc[len(output_df)] = row
        
        len_prompts_list.append(len(prompt.split()))
    
        prompts_list.append(prompt)
        
            
    eval_path = './Generated_Data/' + dataset_name +  '/' + model_name + '/' + prompt_version + '/'
    if not os.path.exists(eval_path):
        os.makedirs(eval_path) 
        
    with open(eval_path + 'prompt_data.text','w') as f:
        for k, each_propmt in enumerate(prompts_list):
            f.write("ID:{0} ====================================\t{1}\n".format(k,each_propmt))
            # if k > 10: break 
    
    with open( eval_path + 'prompt_instruction.text','w') as f:
        if prompt_version=='p-prim':
            f.write("Role: {0}\n".format(task_instruction_pprim))
        elif prompt_version=='gen_wrong_ans':
            f.write("Role: {0}\n".format(task_instruction_gen_wrong_response))
        elif prompt_version=='meta_knowledge':
            f.write("Role: {0}\n".format(task_instruction_meta_knowledge))
        elif prompt_version=='assess':
            f.write("Role: {0}\n".format(task_instruction))
        
        
    '''
    Call API
    '''
    num_batches = len(prompts_list) / batch_size 
    
    fname = dataset_name + '-' + model_name + '-' + prompt_version + '-thld-av-' + str(av_input_len_thld) + '-' + str(input_len_thld)
    
    path  = './pkl_files/'+ dataset_name +  '/' + model_name + '/' + prompt_version + '/'
    
    if max_tokens is not None:
        path += 'max_token_' + str(max_tokens) + '/'

    if not os.path.exists(path):
        os.makedirs(path)
        
    nest_asyncio.apply()
    all_generated_text = run_all_async(num_batches, batch_size, prompts_list,fname, model_name, path, max_tokens, prompt_version)
    
    all_text_list = []
    for gen_text_list in all_generated_text:
        for text in gen_text_list:
            all_text_list.append(text)
            
    print(len(all_text_list))  
    
    outputs_col = model_name + "_" + prompt_version
    
    output_df[outputs_col] = all_text_list          
     
    '''
    Output to csv
    '''
    eval_file = eval_path + model_name + "_" + dataset_name + "_" + prompt_version + "_thld_av_" + str(av_input_len_thld) + '_' + str(input_len_thld)
    
    if max_tokens is not None:
        eval_file += 'max_token_' + str(max_tokens) 
        
    output_df.to_csv(eval_file +  ".csv", index=False)
    
   
    output_df.to_json(eval_file + '.json', orient='records', lines=True, indent=1)
    
    
    # show_output_nicely(eval_file +  ".csv",  eval_file + '.txt', outputs_col)
    
    return


def extract_info(row):
    # Extract 'Correctness'
    
    
    # Check if 'Explanation:' exists, and extract it if present
    correctness = ""
    
    if "Correctness:" in row:
        correctness = row.split("Correctness:")[1].split("Explanation:")[0].strip()
    
    elif "Correct" in row:
        correctness = "Correct"
        
    elif "Incorrect" in row:
        correctness = "Incorrect"
        
    elif "Partially Correct" in row:
        correctness = "Partially Correct"
    
  
    
    explanation = ""
    if "Explanation:" in row:
        explanation = row.split("Explanation:")[1].strip()
        
    return pd.Series([correctness, explanation])

def show_output_nicely(df, out_csv,model_name, prompt_version):
    
    # Apply function to the 'Input' column and create new columns
    df[['Correctness', 'Exp_for_correctness']] = df[model_name+'_'+ prompt_version].apply(extract_info)

    # Save the modified DataFrame with new columns back to a CSV file if needed
    df.to_csv(out_csv, index=False)   
    
    return



def add_coursedata_to_input_data(mapping_course_csv, student_input_csv, out_file_name):
    
    mapping_df = pd.read_csv(mapping_course_csv)
    student_input_df = pd.read_csv(student_input_csv)
    
    merged_df = pd.merge(student_input_df, mapping_df, on='dataset', how='left')
            
    merged_df.to_csv(out_file_name,index=False)
    
    return
    

def main():
    
    # mapping_course_csv = '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/course_version_mapping_data.csv'

    # problem_csv = '/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/stat/problem_data_all_w_explanation.csv'
    # problem_csv = '/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/stat/problem_data_all_w_explanation_qid_match.csv'
    # problem_csv = '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/stat/problem_data_all_w_explanation_qid_match_courseVersion.csv'
    # problem_csv = '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/stat/problem_data_all_w_explanation_qid_match.csv'
    
    # student_input_csv = '/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/shortAns_filtered/Betshune_ShortAns_filtered.csv'
    # student_input_csv = '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/shortAns_filtered/Heartland_ShortAns_filtered.csv'
    # student_input_csv = '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/shortAns_filtered/OLI_ShortAns_filtered.csv'

    # student_input_with_coursever_csv = '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/shortAns_filtered/Heartland_ShortAns_filtered_with_coursename.csv'
    # add_coursedata_to_input_data(mapping_course_csv, student_input_csv, student_input_with_coursever_csv)
    
    
    '''
    Get the student response for each "Problem Name" and "Step Name"
    '''
    
    model_name = "gpt-4o-mini"
    # model_name = "gpt-4"
    dataset_name = 'Betsune'
    # dataset_name = 'Heartland'
    # dataset_name = 'OLI Biology'
    prompt_version = 'meta_knowledge'
    # prompt_version = 'p-prim' if you want to generate p-prims for the topic or 'assess' if you want to access students responses, 'gen_wrong_ans' for generating wrong responses.
    
    
    # sorted_by_input_data_csv = mearge_input_and_sort(problem_csv, student_input_csv, dataset_name)
    # sorted_by_input_data_csv = mearge_input_and_sort_multi_course(problem_csv, student_input_with_coursever_csv, dataset_name)
    # source_data_to_read = sorted_by_input_data_csv
    
    # source_data_to_read = '/Users/machi/Desktop/OLI_TransactionData_Analyze/sorted_data/Betsune/Betsune_sorted_by_av_inputlen.csv'
    # source_data_to_read = '/Users/machi/Documents/GitHub/Misconception_Analysis/OLI_TransactionData_Analyze/sorted_data/Betsune/Betsune_qid_match_all_sorted_by_av_inputlen.csv'
    # source_data_to_read = '/Users/machi/Desktop/OLI_TransactionData_Analyze/sorted_data/Betsune/Betsune_sorted_by_av_inputlen_Multi_Steps.csv'
    # source_data_to_read = '/Users/machi/Documents/GitHub/Misconception_Analysis/OLI_TransactionData_Analyze/sorted_data/OLI Biology/OLI Biology_sorted_by_av_inputlen_noblank.csv'
    
    # for gen_wrong_ans /  for meta_knowledge
    source_data_to_read = '/Users/machi/Library/CloudStorage/GoogleDrive-machi.shimmei.e6@tohoku.ac.jp/My Drive/Misconception_Analysis/Misconception_OLI_Biology_Data/Student Transaction Data/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/stat/problem_data_all_w_explanation_qid_match.csv'
    

    
    ques_list_to_skip = ['List several questions you have about the study of biology. What concepts do you hope to cover in this course? What are you most excited to learn about?',
     'List three control variables other than age.']
    
    prob_names_skip = ['genetics_heredity_albino_digt', 'molecular_genes_mutation_lbd','intro_biology_intro_quiz','genetics_molecular_code_lbd']
    
    run_model(source_data_to_read,model_name, dataset_name, prompt_version, ques_list_to_skip,prob_names_skip,av_input_len_thld = 10, input_len_thld = 10,
                        max_tokens = None, min_num_words = 10, max_num_words = 300, max_num_words_in_all_responses = 100000, batch_size = 32)
    
    
    '''
    # Notes----------------
    # For OLI data
    * Run if the avarge input lenght of the problem > 20  the input length > 10, assess the input.
    
    '''
    
    # output_df = '/Users/machi/Desktop/OLI_TransactionData_Analyze/eval/Betsune/gpt-4o-mini/zero_shots/gpt-4o-mini_Betsune_zero_shots.csv'
    # output_df = '/Users/machi/Documents/GitHub/Misconception_Analysis/OLI_TransactionData_Analyze/eval/OLI Biology/gpt-4o-mini/zero_shots_qid_match/gpt-4o-mini_OLI Biology_zero_shots_qid_match_thld_av_20_10.csv'
    # # extracted_csv_name = '/Users/machi/Desktop/OLI_TransactionData_Analyze/eval/Betsune/gpt-4o-mini/zero_shots/gpt-4o-mini_Betsune_zero_shots_extracted.csv'
    # extracted_csv_name = '/Users/machi/Documents/GitHub/Misconception_Analysis/OLI_TransactionData_Analyze/eval/OLI Biology/gpt-4o-mini/zero_shots_qid_match/gpt-4o-mini_OLI Biology_zero_shots_qid_match_thld_av_20_10_extracted.csv'
    
    # df = pd.read_csv(output_df)
    
    # show_output_nicely(df,extracted_csv_name, model_name, prompt_version)
    

    
    

if __name__ == '__main__':
    
    main()
    # num_pickle_fil
    

