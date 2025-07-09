from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

app = FastAPI()

# 加载tokenizer和onnx模型
tokenizer = AutoTokenizer.from_pretrained("./model/converted")
session = ort.InferenceSession("./model/converted/model.onnx", providers=["CPUExecutionProvider"])

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64

class InferenceResponse(BaseModel):
    output: str

@app.post("/inference", response_model=InferenceResponse)
def inference(req: PromptRequest):
    inputs = tokenizer(req.prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    generated = input_ids.copy()
    for _ in range(req.max_new_tokens):
        position_ids = np.arange(generated.shape[1], dtype=np.int64)
        position_ids = np.expand_dims(position_ids, axis=0)
        ort_inputs = {
            "input_ids": generated,
            "attention_mask": np.ones_like(generated),
            "position_ids": position_ids
        }
        ort_outs = session.run(None, ort_inputs)
        logits = ort_outs[0]
        next_token_id = int(np.argmax(logits[0, -1]))
        if next_token_id == tokenizer.eos_token_id:
            break
        generated = np.concatenate([generated, [[next_token_id]]], axis=1)

    # 只解码新生成部分
    output = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    return InferenceResponse(output=output)