import os
import re
import json
import PyPDF2
from PyPDF2 import PdfReader
import torch
from PIL import Image
from tqdm import tqdm
# from transformers import MllamaForConditionalGeneration, AutoProcessor
import transformers
import argparse
import openai
import time
# import tiktoken


# model_name_or_path = "m42-health/Llama3-Med42-70B"
# model_name_or_path = "ProbeMedicalYonseiMAILab/medllama3-v20"
# model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3-70B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-405B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"
# model_name_or_path = "meta-llama/Llama-3.3-70B-Instruct"

# model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_name_or_path,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

model_name_or_path = "o4-mini-2025-04-16"
openai.api_key = ""

def apply_chatgpt(messages, model=model_name_or_path, temperature=0.3, max_tokens=8192, top_p=1.0):
    if "o4-mini" in model_name_or_path: 
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
    elif "gpt" in model_name_or_path:
            response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    # return response.choices[0].message["content"]
    return response.choices[0].message.content

def apply_medllama3(messages, temperature=0.7, max_tokens=-1, top_k=150, top_p=0.75):
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if max_tokens != -1:
        outputs = pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt) :]
    else:
        outputs = pipeline(
            prompt,
            max_new_tokens=8192,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt) :]
    return response

def extract_json(response, pattern = r'\[.*\]'):
    # Regular expression pattern to match JSON content

    # Search for the pattern in the text
    # match = re.search(pattern, response, re.DOTALL)
    matches = re.findall(pattern, response, re.DOTALL)

    if not matches:
        print("No JSON object found in the text.")
        print(response)
        # print(json_data)
        return None, None

    json_data = matches[0] if len(matches) == 1 else matches[-1]
    
    try:
        # Load the JSON data
        data = json.loads(json_data)
        return None, data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        # print(response)
        # print(json_data)
        return e, json_data

def handleError(messages, next_response):
    error, next_response_dict = extract_json(next_response)
    print(error)
    print(next_response)
    ################################ no json file, regenerate ################################
    cnt = 1
    while error == None and next_response_dict == None and next_response:
        print(f"No json, repeat generating for the {cnt} time")
        raw_prompt = messages[0]["content"]
        prompt = "Plseas return the result in the defined json. " + raw_prompt
        messages[0]["content"] = prompt
        if "o4-mini" not in model_name_or_path and "gpt" not in model_name_or_path:
            next_response = apply_medllama3(messages, temperature=0.3)
        else:
            next_response = apply_chatgpt(messages, temperature=0.3)
        error, next_response_dict = extract_json(next_response)
        cnt += 1

    ################################ json file incorrect ################################
    cnt = 1
    while error and cnt < 10:
        print(f"fix error for the {cnt} time")
        prompt = f"""There is an Error decoding JSON: {error} in the following json data
        {next_response_dict}, Can you fix the error and return the correct json format. Directly return the json without explanation.
        """
        messages = [{"role": "user", "content": prompt}]
        if "o4-mini" not in model_name_or_path and "gpt" not in model_name_or_path:
            new_response = apply_medllama3(messages, temperature=0.3)
        else:
            new_response = apply_chatgpt(messages, temperature=0.3)
        print(new_response)
        error, next_response_dict = extract_json(new_response)
        cnt += 1
    
    if error:
        prompt = f"""There is an Error decoding JSON: {error} in the following json data
        {next_response_dict}, Can you fix the error and return the correct json format. Make sure it can be loaded using python (json.loads()). Directly return the json without explanation.
        """
        messages = [{"role": "user", "content": prompt}]
        if "o4-mini" not in model_name_or_path and "gpt" not in model_name_or_path:
            new_response = apply_medllama3(messages, temperature=0.3)
        else:
            new_response = apply_chatgpt(messages, temperature=0.3)
        next_response_dict = json.loads(new_response)
    return next_response_dict

def call_llm(raw_text):
#     prompt = f"""Well organize the textbook. The overall text is all about {sec}. 
#     1. Ignore the figure and its captions.
#     2. If you think there are subtitles in the text, well organize it like "subtitle": "paragraph". But the paragraph must be the exact raw text. no summarization.
#     3. Return a json format {{"your content"}}
# """
    prompt = f"""You are an expert EMS educator. Categorize the following multiple-choice question into the following NREMT-aligned categories, note that one question might be multiple categories:

    1. Airway/Respiration/Ventilation
        Includes
        - Ventilation
        - Capnography
        - Oxygenation
        - Airway and Ventilation
        - Airway Management
        - Respiratory Emergencies
        - Supplemental Oxygen
        Airway Management

    2. Cardiovascular
        Includes
        - Resuscitation
        - Post-Resuscitation Care
        - Ventricular Assist Devices
        - Stroke
        - Cardiac Arrest
        - Pediatric Cardiac Arrest
        - Congestive Heart Failure
        - Acute Coronary Syndrome
        - EKG monitoring

    3. Operations
        Includes
        - At-Risk Populations
        - Ambulance Safety
        - Field Triage—Disasters/MCIs
        - EMS Provider Hygiene, Safety, and Vaccine
        - EMS Culture of Safety
        - Pediatric Transport
        - Crew Resource Management
        - EMS Research
        - Evidence Based Guidelines
        - Ethics/Legal
        - Documentation
        
    4. Medical
        Includes
        - Special Healthcare Needs
        - OB/GYN Emergencies
        - Infectious Diseases
        - Medication Delivery (Medicines)
        - Pain Management
        - Psychiatric and Behavioral Emergencies
        - Toxicological Emergencies - Opioids
        - Neurological Emergencies - Seizures
        - Endocrine Emergencies - Diabetes
        - Immunological Emergencies
        - Poisoning
        - Environmental Emergencies
        - Behavioral Emergencies

    5. Trauma
        Includes
        - Trauma Triage
        - Central Nervous System (CNS) Injury
        - Hemorrhage Control
        - Fluid Resuscitation
        - Shock
        - Soft Tissue Injuries
        - Injuries to the Chest, Abdomen, and Genitalia
        - Injuries to Muscles, Bones, and Joints
        - Injuries to Head, Neck, and Spine
        - Burns

    6. Assessment 
        Patient Assessment Includes 
        - Scene Size-Up
        - Primary Assessment
        - History Taking and Secondary Assessment
        
    7. Uncategorized 
        Use only if none of the above categories apply. For example, anatomy.
    
Consider the underlying concept or skill the question is assessing. Do not base your decision on keywords alone; use clinical reasoning to determine the core domain.

Return your answer as a list with a single number, e.g., ["1"].

Here is the question to categorize:
{raw_text}"""

    messages = [{"role": "user", "content": prompt}]

    if "o4-mini" not in model_name_or_path and "gpt" not in model_name_or_path:
        response = apply_medllama3(messages, temperature=0.3)
    else:
        response = apply_chatgpt(messages, temperature=0.3)
    # print(response)
    error, jsonfile = extract_json(response, pattern = r'\[.*\]')
    
    if error:
        jsonfile = handleError(messages, response)
        if not jsonfile:
            raise Exception("after handling error, there is still no json file")
    
    if not jsonfile:
        raise Exception("no json, rerun the code")

    return jsonfile





def generate_label(file_name):
    map = {
        "1": "airway_respiration_and_ventilation",
        "2": "cardiology_and_resuscitation",
        "3": "ems_operations",
        "4": "medical_and_obstetrics_gynecology",
        "5": "trauma",
        "6": "assessment",
        "7": "uncategorized"
    }

    with open(f'./{file_name}.json', 'r') as f:
        data = json.load(f)

    pseudo_labels = []

    if not os.path.exists("./log"):
        os.makedirs("./log")

    if f"{file_name}.json" in os.listdir("./log"):
        with open(f"./log/{file_name}.json", "r") as f:
            log = json.load(f)
    else:
        log = {}

    
    for each in tqdm(data):
        q = each["question"]
        choices = [c for c in each["choices"]]
        text = f"{q}\n{choices}"

        if text in log:
            pseudo_labels.append(log[text])
            continue

        # list
        preds = call_llm(text)
        final_pred = []
        for pred in preds:
            pred_category = str(pred)

            if any(char.isdigit() for char in pred_category):
                if pred_category not in map:
                    pred_category = pred_category[0]
            else:
                secondary_map = {
                    "airway/respiration/ventilation": "1",
                    "cardiovascular": "2",
                    "operations": "3",
                    "medical": "4",
                    "trauma": "5",
                    "assessment": "6",
                    "uncategorized": "7",
                }
                if pred_category.lower() in secondary_map:
                    pred_category = secondary_map[pred_category.lower()]
                else:
                    raise Exception("rerun the code")

            pred_category = map[pred_category]
            final_pred.append(pred_category)

        cur_dct = each
        cur_dct["pred category"] = final_pred
        log[text] = cur_dct
        with open(f"./log/{file_name}.json", "w") as f:
            json.dump(log, f, indent=4)
        pseudo_labels.append(cur_dct)
    
    with open(f'./{file_name}_llm.json', 'w') as f:
        json.dump(pseudo_labels, f, indent=4)



def pick_wrong_label(file_name):
    with open(f"./{file_name}.json", "r") as f:
        content = json.load(f)
    
    save_file = []
    for each in content:
        gt = each["category"]
        pred = each["pred category"]
        if gt != pred:
            save_file.append(each)
    
    with open(f"./{file_name}-check.json", "w") as f:
        json.dump(save_file, f, indent=4)

    print(len(save_file))


if __name__ == "__main__":
    generate_label("unlabel_close")
    # pick_wrong_label("label_open_gpt4o")