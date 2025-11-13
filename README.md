# DCFA: Dual-view Causal Attribution for Failure Reasoning in Agentic AI Systems


## Datasets

We use the datasets in the [Who\&When](https://github.com/mingyin1/Agents_Failure_Attribution) benchmark, where the datasets are publicated on [Hugging Face](https://huggingface.co/datasets/Kevin355/Who_and_When)

### Requirements

To install requirements:

```
pip install -r requirements.txt
```

### Inference
Please ensure that you specify the AutoFA method (--method) in the corresponding sections of the code before executing it.

- Models
DCFA supports two inference modes:
- Cloud API mode for large models (e.g., DeepSeek-R1-671B, GPT-5)
- vLLM or Local deployment for smaller models (e.g., Qwen3-Coder-30B)

#### Run
Used for models that can be loaded locally:
```
python Automated_FA/DCFA_failure_attribution.py --model #MODEL --api_key #API_KEY --base_url #BASE_URL --local_llm_type "local" --local_model_path #LOCAL_MODEL_PATH 
```
Used when the model is hosted via a vLLM inference server:
```
python Automated_FA/DCFA_failure_attribution.py --model #MODEL --api_key #API_KEY --base_url #BASE_URL --local_llm_type "vllm" --vllm_api_key #VLLM_API_KEY --vllm_base_url #VLLM_BASE_URL
```