import pandas as pd

from openai import OpenAI, AsyncOpenAI

import math, re
import nest_asyncio, asyncio
import pickle, time

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

'''
Ask GPT to list misconceptions.
'''

few_shots = '''
Topic: Statistical Significance
Relevant Topics:[List 5 relevant topics to the given topic.]
Possible Misconceptions for each : [List 5 possible misconceptions for each relevant topics]
    
'''

task_inst_list_terms =   '''
Given a topic, list 5 related key terms, then list 5 possilbe misconceptions for each key term.
'''


task_inst_with_example =   '''
Given a topic, list 5 related key terms, then list 5 possilbe misconceptions for each key term.
Also, for each misconception, show an example of context where the misconception is likely to occur by providing a question and students'responses to it.
The question should not be too simple that directly ask about the misconception.
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
    if version=='common_misc':
        tasks = [loop.create_task(get_async_gpt4(task_inst_list_terms, prompt, model_name)) for prompt in lst]
    elif version=='common_misc_context':
        tasks = [loop.create_task(get_async_gpt4(task_inst_with_example, prompt, model_name)) for prompt in lst]
    
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

def format_input_output(topic, zero_shot=True):
   
    prompt = f'''
    Topic: {topic}
    Relevant Topics:[List 5 relevant topics to the given topic.]
    Possible Misconceptions for each : [List 5 possible misconceptions for each relevant topics]
    '''
    return prompt


def format_input_output_context(topic, zero_shot=True):
   
    prompt = f'''
    Topic: {topic}
    Relevant Topics:[List 5 relevant topics to the given topic.]
    Possible Misconceptions for each : [List 5 possible misconceptions for each relevant topics]
    Example Context: [For each misconception, provide an example of context where the misconception is likely to occur by providing a question and students'responses to it. The question should not be too simple that directly ask about the misconception.]
    '''
    return prompt

def run_model(source_data_to_read,model_name, dataset_name, prompt_version, av_input_len_thld = 0, input_len_thld = 0,
                        max_tokens = None, min_num_words = 10, max_num_words = 300, max_num_words_in_all_responses = 100000, batch_size = 32, ):
    '''
    Run model
    '''
    df = pd.read_csv(source_data_to_read)
    
    output_df = pd.DataFrame(columns=df.columns)
    len_prompts_list = []
    prompts_list = []
    
    blank_pattern = r'^\s*$'

    # Suppose the source data is a list of topic
    for i, row in df.iterrows():
    
        # if i > 2:break
        topic = row['topic']
        

        prompt = format_input_output(topic)
        
        if prompt_version == 'common_misc':
            prompt = format_input_output(topic)
            task_instruction = task_inst_list_terms

        elif prompt_version == 'common_misc_context':
            prompt = format_input_output_context(topic)
            task_instruction = task_inst_with_example
        else:
            print("Please specfiy the pre-defined Prompt Version")
            sys.exit(0)
        
        output_df.loc[len(output_df)] = row
        
        len_prompts_list.append(len(prompt.split()))
    
        prompts_list.append(prompt)
        
            
    eval_path = './Generated_Data/' + dataset_name +  '/' + model_name + '/' + prompt_version + '/'
    if not os.path.exists(eval_path):
        os.makedirs(eval_path) 
        
    with open(eval_path + 'prompt.text','w') as f:
        for k, each_propmt in enumerate(prompts_list):
            f.write("ID:{0} ====================================\t{1}\n".format(k,each_propmt))
            # if k > 10: break 
            
    with open( eval_path + 'prompt_instruction.text','w') as f:
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


def main():
    # model_name = "gpt-4o-mini"
    model_name = 'gpt-4o-2024-08-06'
    dataset_name = 'Statistics'
    # prompt_version = 'common_misc'
    
    prompt_version = 'common_misc_context'
    
    # source_data_to_read = '/Users/machi/Library/CloudStorage/GoogleDrive-machi.shimmei.e6@tohoku.ac.jp/My Drive/Misconception_Analysis/Statistics/'
    source_data_to_read = '/Users/machi/Documents/GitHub/Misconception_Analysis/OLI_TransactionData_Analyze/topics.csv'

    run_model(source_data_to_read, model_name, dataset_name, prompt_version, av_input_len_thld = 10, input_len_thld = 10,
                        max_tokens = None, min_num_words = 10, max_num_words = 300, max_num_words_in_all_responses = 100000, batch_size = 32)
    
    return



if __name__ == '__main__':
    
    main()


