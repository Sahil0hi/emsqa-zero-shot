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

