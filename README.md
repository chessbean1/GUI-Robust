# GUI-Robust: A Comprehensive Dataset for Testing GUI Agent Robustness in Real-World Anomalies

[![PDF GUI-Robust](https://img.shields.io/badge/PDF-GUI--Robust-red)](#)
[![Dataset GUI-Robust](https://img.shields.io/badge/Dataset-GUI--Robust-brightgreen)](https://huggingface.co/datasets/kuangtie/GUI-Robust)



## Evaluation Script

Run the toolkit via:

```bash
python evaluation.py \
  --model_name <YourModel> \
  --eval_type step|task \
  --task_type normal|abnormal \
  --data_path <path_to_data_folder>
```

The evaluation script supports two modes:

- `--eval_type` `step`: Evaluates per-step grounding accuracy (Action acc. and Coord. acc.)

- `--eval_type` `task`: Evaluates full task execution (Action acc., Coord. acc., and Task Success)

We provide the Qwen2.5-VL model as an example to run the evaluation script (Need to fill in your own API key).

## Integrate Your Own Model

### Model Interface Specification

To integrate custom models into the evaluation, each model should implement the following interface:

```python
class YourModel:
    def __init__()
    
    def pred_step_loc(self, step_description: str, screenshot_base64: str) \
        -> dict:
        # Return a prediction for the current step
        return {
            "ele_loc": {"x": 100, "y": 200},
            "ele_type": "text",
            "action": {"type": "click", "content": "Search"}
        }

    def pred_task_full(self, task_description: str, \
        screenshot_list_base64: List[str]) -> List[dict]:
        # Return a list of predictions for the full task
        return [ ... ]  # One entry per step
```

These two methods respectively handle single-step prediction and full-task multi-step prediction. The evaluation pipeline will automatically invoke these methods, compare predictions against ground truth, and report metrics such as action accuracy, coordinate accuracy, and task success rate.

### Prediction Output Format 

Each prediction should contain:
- `Element coordinates` (x, y)
- `Element type` (icon, text, box, none)
- `Action type` and `content` (click, input, get\_info, open, wait, human)

For full-task predictions, return a list of such dictionaries (one per step).

### Prompt Templates

A Chinese version template can be seen in ./model/prompt

## Citation
Please consider citing if you find our work useful:
```
@inproceedings{
    yang2025guirobust,
    title={GUI-Robust: A Comprehensive Dataset for Testing GUI Agent Robustness in Real-World Anomalies},
    author={Jingqi Yang and Zhilong Song and Jiawei Chen and Mingli Song and Sheng Zhou and Linjun Sun and Xiaogang Ouyang and Chun Chen and Can Wang},
    booktitle={NeurIPS Datasets and Benchmarks Track},
    year={2025},
    url={https://openreview.net/forum?id=22gw3kITCd},
}
```