from openai import OpenAI
import os, base64, json
from tqdm import tqdm

def get_image_format(base64_image):
    header = base64_image[:20]  # 更长前缀更安全
    if header.startswith("iVBOR"):
        return "png"
    elif header.startswith("/9j/"):
        return "jpeg"
    elif header.startswith("R0lG"):
        return "gif"
    else:
        return "unknown"

def generate_data_uri(base64_image):
    img_format = get_image_format(base64_image)
    if img_format == "unknown":
        raise ValueError("Unsupported or unknown image format")

    return f"data:image/{img_format};base64,{base64_image}"

def extract_pred(response):
    """
    处理模型输出 response，提取为标准 pred 格式，包含必要字段结构及容错
    """
    # 如果是字符串，先尝试解析为字典
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            print("× response JSON 解析失败，原始内容:", response)
            return {
                "ele_loc": {"x": -1, "y": -1},
                "ele_type": "",
                "action": {"type": "", "content": ""}
            }

    # 如果解析后不是字典，直接返回空结构
    if not isinstance(response, dict):
        print("× response 格式异常，非 dict：", response)
        return {
            "ele_loc": {"x": -1, "y": -1},
            "ele_type": "",
            "action": {"type": "", "content": ""}
        }

    # 提取字段，带默认值回退
    ele_loc = response.get("ele_loc", {})
    ele_type = response.get("ele_type", "")
    action = response.get("action", {})

    pred = {
        "ele_loc": {
            "x": ele_loc.get("x", -1),
            "y": ele_loc.get("y", -1)
        },
        "ele_type": ele_type if isinstance(ele_type, str) else "",
        "action": {
            "type": action.get("type", "") if isinstance(action, dict) else "",
            "content": action.get("content", "") if isinstance(action, dict) else ""
        }
    }

    return pred

class Qwen2_5_VL:

    def __init__(
        self, 
        api_key="sk-xxx",   # TODO：替换为你自己的api key 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        prompt_step_file = "prompt_step.txt",
        prompt_task_file = "prompt_task.txt",
        prompt_task_abn_file = "prompt_task_abnormal.txt"
    ):
        self.api_key = api_key
        self.base_url = base_url
        prompt_step_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), prompt_step_file)
        prompt_task_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), prompt_task_file)
        prompt_task_abn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), prompt_task_abn_file)

        with open(prompt_step_path, 'r', encoding='utf-8') as f:
            prompt_step = f.read()
        with open(prompt_task_path, 'r', encoding='utf-8') as f:
            prompt_task = f.read()
        with open(prompt_task_abn_path, 'r', encoding='utf-8') as f:
            prompt_task_abn = f.read()
        self.prompt_step = prompt_step
        self.prompt_task = prompt_task
        self.prompt_task_abn = prompt_task_abn
        self.client = OpenAI(
            base_url=self.base_url, 
            api_key=self.api_key
        )

    def pred_step_loc(self, step_description, base64_image):
        """
        pred = {
            "ele_loc":{
                "x": ,
                "y":
            },
            "ele_type":
            "action":{
                "type": ,
                "content":
            }
        }
        """
        data_uri = generate_data_uri(base64_image)
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": self.prompt_step}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": step_description},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    }
                ]
            }
        ]
        # print(messages)
        chat_completion = self.client.chat.completions.create(
                model="qwen2.5-vl-72b-instruct",
                messages=messages
            )

        data = json.loads(chat_completion.model_dump_json())
        response = data['choices'][0]['message']['content']

        pred = extract_pred(response)

        return pred

    def pred_task_full(self, task_description, base64_image_list):
        """
        完整的任务预测函数，处理多张base64图像的输入并与模型交互，
        每次将历史信息添加到messages中
        """
        # 初始化messages
        
        # 保存历史记录
        history = []
        preds = []
        
        # 遍历所有base64图像并逐步推理
        for base64_image in base64_image_list:
            # 在当前消息中添加新的图片并向模型请求预测
            # print(base64_image)
            data_uri = generate_data_uri(base64_image)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.prompt_task_abn}]
                }] + history + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_description},
                        {"type": "image_url", "image_url": {"url": data_uri}} 
                    ]
                }
            ]
            

            # 调用模型获取输出
            chat_completion = self.client.chat.completions.create(
                model="qwen2.5-vl-72b-instruct",
                messages=messages
            )
            
            # 从模型的响应中提取内容
            data = json.loads(chat_completion.model_dump_json())
            response = data['choices'][0]['message']['content']

            assistant_message = {
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            }
            history.append(assistant_message)
            
            pred = extract_pred(response)

            # 输出模型返回内容，作为下一轮输入的历史记录
            preds.append(pred)
        
        return preds


