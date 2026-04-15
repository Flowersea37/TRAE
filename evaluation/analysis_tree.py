import json
import re

# datapath = "/workspace/mnt/lxb_work/dlx_work/dynamic_reflection/model_eval/outputs/eval-1-3-qwen3-8b-grpo-clip-higher-step75-AceMath-7B-RM-temp0.6-topp0.95-max-token5000/aime24/eval.json"
reflect_times = 3
raw_path = "/workspace/mnt/lxb_work/xgq_work/TRAE_upload/evaluation/evaluation/outputs/eval-3-qwen3-8b-tree-best-config-10k-epoch1-dis05-tsp1-tokenloss-while-style-releasetest-step117-max-token5000/"

def extract_tags(text):
    """
    提取字符串中最后一个 <response></response> 和 <critique></critique> 标签内容。
    支持内容包含换行符。
    """
    result = {}

    # 找到所有 <response>...</response>
    responses = re.findall(r"<response>([\s\S]*?)</response>", text, re.MULTILINE)
    if responses == []:
        return False

    # 找到所有 <critique>...</critique>
    critiques = re.findall(r"<critique>([\s\S]*?)</critique>", text, re.MULTILINE)
    if critiques == []:
        return False

    return True

benchmarks = ['gsm8k','olympiad','minervamath','math-500','amc23','aime24','aime25']
# benchmarks = ['gsm8k','olympiad','math-500','amc23','aime24','aime25']
# benchmarks = ['math-500']
# benchmarks = ['mmlupro']
# benchmarks = ['gsm8k','olympiad']
for benchmark_name in benchmarks:
    datapath = raw_path + benchmark_name + "/eval.json"
    pass_1 = 0
    pass_8 = 0
    best_of_8 = 0
    total = 0
    reflect_preference = [0 for _ in range(reflect_times + 1)]

    # === 新增统计 ===
    correct_to_wrong_01 = 0
    wrong_to_correct_01 = 0

    correct_to_wrong_reasoning = 0
    wrong_to_correct_reasoning = 0
    reflect_error = 0
    print('-------------'+benchmark_name+'-----------')

    with open(datapath, "r") as f:
        for j, line in enumerate(f):
            data = json.loads(line)
            total += 1

            initial = data["judges_0"][0]

            # judge@1
            if initial is True:
                pass_1 += 1

            # judge@8
            if True in data["judges_0"]:
                pass_8 += 1


            # 每次反思正确率
            for i in range(reflect_times + 1):
                if data[f"judges_{i}"][0] is True:
                    reflect_preference[i] += 1
                if i == 0:
                    continue
                if extract_tags(data[f"reflection_{i}"][0]) == False:
                    # print(j+1)
                    reflect_error += 1

    # ======================
    # 输出统计结果
    # ======================

    print("Total:", total)
    print("Pass@1:", pass_1)
    for i in range(reflect_times + 1):
        print(f"第{i}次反思正确率: {reflect_preference[i] / total}")
    print('end:-------------'+benchmark_name+'-----------\n')

