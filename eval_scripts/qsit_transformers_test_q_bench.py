import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, AutoTokenizer
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr

def wa5(logits):
    logprobs = np.array([logits["Excellent"], logits["Good"], logits["Fair"], logits["Poor"], logits["Bad"]])
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    return np.inner(probs, np.array([1, 0.75, 0.5, 0.25, 0]))

model_id = "path-to-the-weight"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

processor = AutoProcessor.from_pretrained(model_id)


def predict(image_file, llddata):
    # Load image
    message = llddata["question"] + "\n"
    for choice, ans in zip(["A.", "B.", "C.", "D."], llddata["candidates"]):
        message += f"{choice} {ans}\n"
    conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": message},
            {"type": "image"},
        ],
    },
    ]
    raw_image = Image.open(image_file)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    text = processor.decode(output[0][2:], skip_special_tokens=True)
    return text.split("assistant")[-1]

import json


image_paths = [

                ["/fs-computility/mllm1/zhangzicheng/qa_data_unzip/dataset/qbench/images/dev/"
                ],
                ] * 1

json_prefix = "/fs-computility/mllm1/zhangzicheng/qa_data_unzip/dataset/qbench/"
jsons = [
    json_prefix + "llvisionqa_dev.json"
]

os.makedirs(f"results/qbench/{model_id}", exist_ok=True)

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
        correct = 0

        for i, llddata in enumerate(tqdm(iqadata, desc="Evaluating [{}]".format(json_.split("/")[-1]))):
            try:
                filename = llddata["image"]
            except:
                filename = llddata["img_path"]

            if isinstance(image_path, list):
                for p in image_path:
                    if os.path.exists(p + filename):
                        image = p + filename
            else:
                image = image_path + filename

            batch_data.append(llddata)

            if True or i == len(iqadata):
                with torch.inference_mode():
                    outputs = predict(image, llddata)
                llddata["correct_choice"] = chr(65 + llddata['candidates'].index(llddata["correct_ans"]))
                if llddata["correct_choice"] == outputs or llddata["correct_choice"]+"." in outputs or llddata["correct_ans"] == outputs:
                    correct += 1
                elif llddata["correct_choice"] in outputs.split("\n"):
                    correct += 1
                llddata["response"] = outputs
                print("[Running Accuracy]: {:.4f}".format(correct/(i+1)))

                with open(f"results/qbench/{model_id}/{json_.split('/')[-1][:-5]}_results.jsonl", "a") as wf:
                                        json.dump(llddata, wf)
                                        wf.write("\n")
        with open(f"results/qbench/{model_id}/{json_.split('/')[-1][:-5]}_results.jsonl", "a") as wf:
            json.dump({"ACC:":correct/(i+1)}, wf)
            wf.write("\n")
