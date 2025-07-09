from optimum.onnxruntime import ORTModelForCausalLM, ORTOptimizer, ORTQuantizer, ORTConfig

def export_onnx():
    model_id = "microsoft/phi-1_5"
    save_path = "./model/converted"
    
    model = ORTModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        provider="CPUExecutionProvider",
        export_options={"opset": 17}  # 指定更高的opset
    )
    model.save_pretrained(save_path)
    print("ONNX 模型导出完成，路径:", save_path)

if __name__ == "__main__":
    export_onnx()


"""
optimum-cli export onnx --model microsoft/phi-1_5 ./model/converted --task causal-lm --opset 17
"""