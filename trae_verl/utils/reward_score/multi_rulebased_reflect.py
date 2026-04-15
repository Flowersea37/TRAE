import re
import math
import numpy as np
# from math_verify.errors import TimeoutException
# from math_verify.metric import math_metric
# from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\\\%", "")
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2

def pred_extract_answer(model_output: str):
    if "\\boxed{" not in model_output:
        return None
    
    idx = model_output.rfind("\\boxed{")
    extract_answer = "\\boxed{"

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(model_output):
        if model_output[i] == "{":
            num_left_braces_open += 1
        if model_output[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else model_output[idx + len(extract_answer) : right_brace_idx]

    return retval


def compute_score(solution_str, ground_truth, extra_info=None) -> dict:
    #model_output: str, ground_truth: str, first_reward_score: float = 0, second_reward_score: float = 0, first_judge: bool = False
    model_output = solution_str
    ground_truth = ground_truth["ground_truth"]
    judge_0 = extra_info.get("judge_0",None)
    # rm_scores = responses.get("rm_scores",None)
    responses = extra_info.get("all_responses",[])
    answers = ["" for _ in responses]
    judges = [False for _ in responses]
    print("----------------------------")
    # breakpoint()
    judge_scores = []
    for response in responses:
        if response == "":
            judge_scores.append(-1)
        else:
            judge_scores.append(0)
   
    for idx in range(len(answers)):
        answer = pred_extract_answer(responses[idx])
        if answer == None:
            answer = responses[idx]
        judges[idx] = is_equiv(answer, ground_truth)
        answers[idx] = answer

    for i in range(len(judges)):
        if judges[i] == True:
            judge_scores[i] = 1

    ret_score = sum(judge_scores)
    nparr_judges_score = np.array(judge_scores)

    dict_result = dict()
    dict_result['score'] = ret_score
    dict_result['step_reward'] = nparr_judges_score
    ret_score = sum(judge_scores)
    # print(f"all_responses={responses}")
    print(f"answers={answers},ground_truth={ground_truth}")
    print(f'judge_score = {judge_scores}')
    # print(f'answer={answer},ground_truth={ground_truth}')
    return dict_result