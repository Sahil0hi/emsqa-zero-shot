import argparse
import json
import re 
import os
import time
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCasualLM, pipeline

# just a small comand line interface to run the model, it was not nessacry but I thought it would be nice to have
parser = argparse.ArgumentParser()
parser.add_argument("--input",  required=True, help="Path to EMRJSON files ")
parser.add_argument("--output", required=True, help="output JSON file for predics")
parser.add_argument("--model",  default="Qwen/Qwen2.5-7B-Chat",
                    help="HF model ")
parser.add_argument("--max_new_tokens", type=int, default=8,
                    help="How many tokens to generate for each answer")
args = parser.parse_args()

##############################################################################
#. Load Qwen‑2.5‑7B‑Chat ───────────────────────────────────────────────────
print("Loading model …" )

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)#model from arguemt, defualt is Qwen/Qwen2.5-7B-Chat
model     = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True
            )
gen = pipeline(
        "text-generation", 
        model = model, 
        tokenizer=tokenizer, 
        device_map="auto", 
        batch_size=1.
        batach_size =1
        )
# ##############################################################################
# prompt and parse

answer_regex = re.compile(r"([A-D])", re.I) # matches a single letter A to D case-insensitively


def qwen_ask(question: str, choices: list[str]) -> str:
    prompt = ( "You are an expert EMR taking a certification practice test. \n\n"
              "You are given a question and four possible answers. Your task is to select the best answer from the four options.  \n\n"
              
            f"Question:\n{question}\n\n"
                f"Choices:\n"
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}\n\n"
                "Answer (just one letter A, B, C, or D):"
            )
    completion = gen(prompt, max_new_tokens = args.max_new_tokens, do_sample = False, temperature = 0.0, eos_token_id=tokenizer.eos_token_id)[0]["generated_text"]

    end = completion[len(prompt):]
    match = answer_regex.search(end)
    return match.group(1).upper() if match else "?"

##############################################################################
# Run inference over the dataset ──────────────────────────────────────────

with open(args.input) as file:
    emr = json.load(file)

predictions = []
start = time.time()
for item in tqdm(emr, desc="Inferring"):
    pred_letter = ask_qwen(item["Question"], item["Choices"])
    predictions.append({
        "question":        item["Question"],
        "choices":         item["Choices"],
        "true_answer":     item["Answer"],
        "model_answer":    pred_letter
    })
elapsed = time.time() - start
print(f"\nFinished {len(predictions)} questions in {elapsed/60:.1f} min")

##############################################################################
# output JSON ───────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as file:
    json.dump(predictions, file, indent=2)
print(f"Wrote predictions → {args.output}")