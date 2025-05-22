from tqdm import tqdm
import os, json, base64, sys, argparse
from openai import OpenAI

# from ui_tars_model import UI_TARS
# from qwen2_vl_model import Qwen2_VL
from models.ui_tars_model import UI_TARS
from models.qwen2_5_vl_model import Qwen2_5_VL
from models.gpt4o_model import GPT4o
from models.gemini2_5_model import Gemini2_5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--eval_type', type=str, required=True)
    parser.add_argument('--task_type', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)

    args = parser.parse_args()
    return args

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

is_abnormal = "False"

def convert_field_add_to_encoded(input_json_path, image_folder):
    """
    input_json_path: str - 路径指向 field_add.json
    image_folder: str - 存放 image_0.png 等图像的文件夹路径
    """
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            steps = json.load(f)

        output = []
        task_description = steps[0]["task_description"]

        for i, step in enumerate(steps[1:]):
            img_filename = step["img_path"]
            image_path = os.path.join(image_folder, img_filename)
            encoded_image = encode_image(image_path)

            if is_abnormal == "True":
                converted_step = {
                    "task_description": task_description,
                    "step_id": f"step_{i}",
                    "step_description": step["step_description"],
                    "screenshot_base64": encoded_image,
                    "action": step["action"],
                    "action_human": step["action_human"],
                    "ele_type": step["ele_type"],
                    "ele_loc": step["ele_loc"]
                }
            else:
                converted_step = {
                    "task_description": task_description,
                    "step_id": f"step_{i}",
                    "step_description": step["step_description"],
                    "screenshot_base64": encoded_image,
                    "action": step["action"],
                    "ele_type": step["ele_type"],
                    "ele_loc": step["ele_loc"]
                }
            output.append(converted_step)
    except Exception as e:
        print(input_json_path)
        with open("./wrong_data.txt", 'a', encoding='utf-8') as f:
            f.write(f"[{task_description}] 模型调用失败: {e}")
            f.write("\n")
        print(f"[{task_description}] 模型调用失败: {e}")
        return None

    return output

def is_point_inside_box(point, box):
    """判断点是否在框内"""
    x, y = point
    # if(x==None):
        # print("WTF")
    box_x, box_y, width, height = box['x'], box['y'], box['width'], box['height']
    if None in (box_x, box_y, width, height, x, y):
        print("None Point")
        # print(point)
        return False
    return (box_x <= x <= box_x + width) and (box_y <= y <= box_y + height)

def eval_step_loc(results):
    """
    输入: results 列表，每一项是 {"step_id", "step_description", "prediction", "ground_truth"}
    输出: 打印 action 和 ele_loc 的准确率
    """
    total = 0
    correct_action = 0
    correct_loc = 0

    total_icon = 0
    total_text = 0
    total_box = 0
    correct_loc_icon = 0
    correct_loc_text = 0
    correct_loc_box = 0

    for res in results:
        pred = res.get("prediction", {})
        gt = res.get("ground_truth", {})

        ele_type = gt.get("ele_type", {})

        total += 1
        if ele_type == "icon":
            total_icon += 1
        elif ele_type == "text":
            total_text += 1
        elif ele_type == "box":
            total_box += 1

        if pred is None:
            print(f"× 跳过 step_id={res.get('step_id')}，prediction 为 None")
            continue

        # --- Action 评估 ---
        action_pred = pred.get("action", {})
        action_gt = gt.get("action", {})
        type_match = action_pred.get("type") == action_gt.get("type")
        content_match = action_pred.get("content") == action_gt.get("content")
        if type_match and content_match:
            correct_action += 1

        # --- ele_loc 评估 ---
        loc_pred = pred.get("ele_loc", {})
        point = (loc_pred.get("x", -1), loc_pred.get("y", -1))
        box = gt.get("ele_loc", {})
        if all(k in box for k in ("x", "y", "width", "height")) and is_point_inside_box(point, box):
            correct_loc += 1
            if ele_type == "icon":
                correct_loc_icon += 1
            elif ele_type == "text":
                correct_loc_text += 1
            elif ele_type == "box":
                correct_loc_box += 1


    action_acc = correct_action / total if total else 0
    loc_acc = correct_loc / total if total else 0

    loc_acc_icon = correct_loc_icon / total_icon if total_icon else 0
    loc_acc_text = correct_loc_text / total_text if total_text else 0
    loc_acc_box = correct_loc_box / total_box if total_box else 0

    print(f"√ 评估结果：")
    print(f"- Action 准确率: {correct_action}/{total} = {action_acc:.2%}")
    print(f"- Ele_Loc 位置命中率: {correct_loc}/{total} = {loc_acc:.2%}")

    print(f"- Ele_Loc_icon 位置命中率: {correct_loc_icon}/{total_icon} = {loc_acc_icon:.2%}")
    print(f"- Ele_Loc_text 位置命中率: {correct_loc_text}/{total_text} = {loc_acc_text:.2%}")
    print(f"- Ele_Loc_box 位置命中率: {correct_loc_box}/{total_box} = {loc_acc_box:.2%}")

    return {
        "action_accuracy": action_acc,
        "loc_accuracy": loc_acc,
        "total": total
    }


def eval_task_full(results):
    """
    评测函数，计算任务的整体准确率，包括每一步的action准确率，ele_loc准确率和任务成功率
    输入: results，包含每个任务的预测和实际值
    输出: 计算的各项准确率并打印结果
    """
    total_tasks = len(results)
    total_action_correct = 0
    total_loc_correct = 0
    total_task_successful = 0

    total_steps_action = 0
    total_steps_loc = 0

    # 遍历每个任务
    for task_result in results:
        task_description = task_result["task_description"]
        pred = task_result["prediction"]
        ground_truth = task_result["ground_truth"]

        if pred is None:
            print(f"× 跳过 task={task_description}，prediction 为 None")
            continue

        task_steps = len(pred)
        total_steps_action += task_steps
        
        task_action_correct = 0
        task_loc_correct = 0
        task_successful = True  # 假设任务成功

        # 遍历任务中的每一步
        for pred_step, gt_step in zip(pred, ground_truth):
            # --- Action 准确率 ---
            pred_action = pred_step["action"]
            gt_action = gt_step["action"]
            pred_action_type = pred_action["type"]
            if pred_action_type == "wait":                          # 动作评测时wait和human要特殊处理
                action_correct = ( gt_action["type"] == "wait" )
            elif pred_action_type == "human":
                if is_abnormal == False:
                    action_correct = False
                elif gt_step["action_human"]["content"] == "":
                    action_correct = False
                else:
                    action_correct = True
            else:
                action_correct = (pred_action["type"] == gt_action["type"]) and (pred_action["content"] == gt_action["content"])

            # --- ele_loc 准确率 ---
            # get_info, open 无坐标
            action_type = gt_action["type"]
            element_type = gt_step["ele_type"]
            if element_type != "none":                  #元素类型为none的都不用比较坐标
                total_steps_loc += 1

                pred_loc = pred_step["ele_loc"]
                gt_loc = gt_step["ele_loc"]
                point = (pred_loc["x"], pred_loc["y"])
                box = {
                    "x": gt_loc["x"],
                    "y": gt_loc["y"],
                    "width": gt_loc["width"],
                    "height": gt_loc["height"]
                }
                loc_correct = is_point_inside_box(point, box)

                # 定位累计正确数
                if loc_correct:
                    task_loc_correct += 1
            else:
                loc_correct = True

            # 动作累计正确数
            if action_correct:
                task_action_correct += 1


            # 如果某一步失败，整个任务失败
            if not action_correct or not loc_correct:
                task_successful = False

        # 累加任务的总成绩
        total_action_correct += task_action_correct
        total_loc_correct += task_loc_correct
        if task_successful:
            total_task_successful += 1

    # 计算总体准确率
    action_accuracy = total_action_correct / total_steps_action if total_steps_action > 0 else 0
    loc_accuracy = total_loc_correct / total_steps_loc if total_steps_loc > 0 else 0
    task_success_rate = total_task_successful / total_tasks if total_tasks > 0 else 0

    # 输出评测结果
    print(f"√ 任务评测结果：")
    print(f"- Action 准确率: {total_action_correct}/{total_steps_action} = {action_accuracy:.2%}")
    print(f"- ele_loc 准确率: {total_loc_correct}/{total_steps_loc} = {loc_accuracy:.2%}")
    print(f"- 整体任务成功率: {total_task_successful}/{total_tasks} = {task_success_rate:.2%}")

    return {
        "action_accuracy": action_accuracy,
        "loc_accuracy": loc_accuracy,
        "task_success_rate": task_success_rate
    }



def main(args):

    # directory = './data_ele_loc/'
    directory = args.data_path
    eval_type = args.eval_type
    
    global is_abnormal
    if args.task_type == "abnormal":
        is_abnormal = "True"
    else:
        is_abnormal = "False"
    model_name = args.model_name

    # print(is_abnormal)
    model = None

    if model_name == "Qwen2.5-VL":
        model = Qwen2_5_VL()
    elif model_name == "Gemini2.5-flash-preview":
        model = Gemini2_5()
    elif model_name == "GPT4o":
        model = GPT4o()
    elif model_name == "UI-TARS":
        model = UI_TARS()

    model.__init__()

    task_to_run = []
    for root, dirs, files in tqdm(os.walk(directory)):
        if not dirs:        
            input_json_path = os.path.join(root, "field_add.json")
            tasks = convert_field_add_to_encoded(input_json_path, root)
            if(tasks==None):
                continue
            if(eval_type == "step"):    #Step Location Acc
                task_to_run = task_to_run + tasks
            else:
                task_to_run.append(tasks)

    results = []

    if(eval_type == "step"):
        for step in tqdm(task_to_run):
            step_id = step["step_id"]
            step_description = step["step_description"]
            base64_image = step["screenshot_base64"]

            # 解码图像
            # image = decode_base64_image(base64_img)

            # 调用模型
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
            try:
                pred = model.pred_step_loc(step_description, base64_image) #TODO: model.pred_step_loc
            except Exception as e:
                print(f"[{step_id}] 模型调用失败: {e}")
                pred = None

            # 记录结果（也可加入 ground truth）
            results.append({
                "step_id": step_id,
                "step_description": step_description,
                "prediction": pred,
                "ground_truth": {
                    "action": step["action"],
                    "ele_type": step["ele_type"],
                    "ele_loc": step["ele_loc"]
                }
            })

        # print(results)
        # with open(output_file, 'a', encoding='utf-8') as f:
        output_file = "./result_step_" + model_name +".json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        eval_step_loc(results)

    else:
        for task in tqdm(task_to_run):
            task_description = task[0]["task_description"]
            base64_image_list = []
            i = 0
            for step in task:
                base64_image = step["screenshot_base64"]
                # print(base64_image)
                # print(i)
                i = i+1
                base64_image_list.append(base64_image)


            # 解码图像
            # image = decode_base64_image(base64_img)

            # 调用模型
            """
            pred = [
                {
                    "ele_loc":{
                        "x": ,
                        "y":
                    },
                    "ele_type":
                    "action":{
                        "type": ,
                        "content":
                    }
                },
                ......
            ]

            """
            try:
                pred = model.pred_task_full(task_description, base64_image_list) #TODO: model.pred_task_full
                # print("hi")
            except Exception as e:
                print(f"[{task_description}] 模型调用失败: {e}")
                pred = None

            # 记录结果（也可加入 ground truth）
            if(pred!=None):
                results.append({
                    "task_description": task_description,
                    "prediction": pred,
                    "ground_truth": task
                })

        # print(results)
        # with open(output_file, 'a', encoding='utf-8') as f:
        output_file = "./result_task_" + model_name + ".json"
        output_data = [
            {
                "task_description": item["task_description"],
                "prediction": item["prediction"]
            }
            for item in results
        ]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        # with open(output_file, 'w', encoding='utf-8') as f:
        #     json.dump(results, f, ensure_ascii=False, indent=4)

        eval_task_full(results)

if __name__ == "__main__":
    main(parse_args())        

    # for sample in task_to_run:
    #     print(sample["task_description"])
    #     print(sample["step_id"])
    #     print(sample["step_description"])
    #     print("\n")


