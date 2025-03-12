import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, AutoTokenizer, AutoConfig
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr

def wa5(logits):
    logprobs = np.array([logits["Excellent"], logits["Good"], logits["Fair"], logits["Poor"], logits["Bad"]])
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    return np.inner(probs, np.array([1, 0.75, 0.5, 0.25, 0]))

# fill up here
model_id = "path-to-weight"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)


processor = AutoProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Define rating tokens
toks = ["Excellent", "Good", "Fair", "Poor", "Bad"]
ids_ = [id_[0] for id_ in tokenizer(toks)["input_ids"]]
print("Rating token IDs:", ids_)

# Fixed prompt template (user part only)
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Assume you are an image quality evaluator. \nYour rating should be chosen from the following five categories: Excellent, Good, Fair, Poor, and Bad (from high to low). \nHow would you rate the quality of this image?"},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

def predict(image_file):
    # Load image
    raw_image = Image.open(image_file)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    # Manually append the assistant prefix "The quality of this image is "
    prefix_text = "The quality of this image is "
    prefix_ids = tokenizer(prefix_text, return_tensors="pt")["input_ids"].to(0)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_ids], dim=-1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])  # Update attention mask

    # Generate exactly one token (the rating)
    output = model.generate(
        **inputs,
        max_new_tokens=1,  # Generate only the rating token
        output_logits=True,
        return_dict_in_generate=True,
    )

    # Extract logits for the generated rating token
    last_logits = output.logits[-1][0]  # Shape: [vocab_size]
    logits_dict = {tok: last_logits[id_].item() for tok, id_ in zip(toks, ids_)}
    weighted_score = wa5(logits_dict)
    return logits_dict,weighted_score

import json


image_paths = [

            ["/fs-computility/mllm1/zhangzicheng/qa_data_unzip/dataset/",
            ],
            ] * 2

json_prefix = "/fs-computility/mllm1/zhangzicheng/qa_data_unzip/dataset/jsons/"
jsons = [
    json_prefix + "test_spaq.json",
    json_prefix + "test_koniq.json",
]

os.makedirs(f"iqa_results/{model_id}/", exist_ok=True)

all_pred = []
all_gt = []
model_pred = []
model_gt = []

for image_path, json_ in zip(image_paths,jsons):

    with open(json_) as f:
        iqadata = json.load(f)
        prs, gts = [], []
        image_tensors = []
        batch_data = []

        for i, llddata in enumerate(tqdm(iqadata, desc="Evaluating [{}]".format(json_.split("/")[-1]))):
            try:
                filename = llddata["image"]
            except:
                filename = llddata["img_path"]
            llddata["logits"] = defaultdict(float)

            if isinstance(image_path, list):
                for p in image_path:
                    if os.path.exists(p + filename):
                        image = p + filename
            else:
                image = image_path + filename


            batch_data.append(llddata)

            if True or i == len(iqadata):
                with torch.inference_mode():
                    logits_dict,weighted_score = predict(image)

                for j, xllddata in enumerate(batch_data):

                    xllddata["logits"] = logits_dict
                    xllddata["score"] = weighted_score
                    prs.append(float(xllddata["score"]))
                    if "gt_score" in xllddata.keys():
                        gts.append(float(xllddata["gt_score"]))
                    elif "id" in xllddata.keys():
                        gts.append(float(xllddata["id"].split('->')[-1]))
                    else:
                        print("No GT score found.")
                    # print(llddata)
                    json_ = json_.replace("combined/", "combined-")
                    with open(f"iqa_results/{model_id}/intance-{json_.split('/')[-1]}", "a") as wf:
                        json.dump(xllddata, wf)

                batch_data = []

        all_pred += prs
        all_gt += gts
        model_gt.append(np.mean(gts))

        print("Spearmanr", spearmanr(prs,gts)[0], "Pearson", pearsonr(prs,gts)[0], "Mean", np.mean(prs), (np.mean(gts)-1)/4)
        result_data = {'srcc':spearmanr(prs,gts)[0],'plcc':pearsonr(prs,gts)[0]}
        with open(f"iqa_results/{model_id}/{json_.split('/')[-1]}", "a") as wf:
            json.dump(result_data, wf)
