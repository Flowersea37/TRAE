GENERATION_PROMPT = """
{question}
The finally answer should be enclosed in \\boxed{{}}.
""".strip()

CRITIC_PROMPT = """
You are given a math problem and a initial solution to the problem. Your task is to generation a critique to the solution which should contain the analysis to the solution, the judge to the solution and some instruction about how to correct the solution if it is wrong or make it better if it is correct and then generate a new solution refined from the initial solution that given to you, and you should refer from the critique because you can get some advice from it.
The new solution should solve the problem step by step. The final answer must be enclosed in \\boxed{{}}.
You must output in the following format:
<critique>your critique to the solution</critique>
<response>the new solution refined from the initial solution</response>

The input:
problem:{question}
solution:{solution}
""".strip()

GENERATION_PROMPT_GENERAL_DOMIN = """
Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format '\\boxed{{$LETTER}}' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J.
Question:{question}
Options:{options}
The finally answer should be enclosed in \\boxed{{}}.
""".strip()

CRITIC_PROMPT_GENERAL_DOMIN = """
You are given a multiple choice question and a initial solution to the problem. Your task is to generation a critique to the solution which should contain the analysis to the solution, the judge to the solution and some instruction about how to correct the solution if it is wrong or make it better if it is correct and then generate a new solution refined from the initial solution that given to you, and you should refer from the critique because you can get some advice from it.
The new solution should solve the problem step by step.There is only one correct answer. The last line of your response should be in the format '\\boxed{{$LETTER}}' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J. The final answer must be enclosed in \\boxed{{}}.
You must output in the following format:
<critique>your critique to the solution</critique>
<response>the new solution refined from the initial solution</response>

The input:
problem:{question}
options:{options}
solution:{solution}
""".strip()

Reflect_PROMPT_1 = """
You are given a math problem and a initial solution to the problem. Your task is to generation a critique to the solution which should contain the analysis to the solution, the judge to the solution and some instruction about how to correct the solution if it is wrong or make it better if it is correct and then generate a new solution refined from the initial solution that given to you, and you should refer from the critique because you can get some advice from it.
The new solution should solve the problem step by step.
You must output in the following format and you just need to generate one critique and response:
<critique1>your critique to the solution</critique1>
<response1>the new solution refined from the initial solution</response1>

The input:
problem:{question}
solution:{solution}
""".strip()

Reflect_PROMPT_2_1 = """
The solution you need to reflect is in the previous step and enclosed in <response1></response1>. Your task is to generation a critique to the solution, which should contain the analysis to the solution, the judge to the solution and some instruction about how to correct the solution if it is wrong or make it better if it is correct and then generate a new solution refined from the previous solution, and referred from the critique.
The new solution should solve the problem step by step. The final answer must be enclosed in \\boxed{{}}.
You must output in the following format and you just need to generate one critique and response: 
<critique2>your critique to the solution</critique2>
<response2>the new solution refined from the initial solution</response2>
""".strip()

Reflect_PROMPT_2_2 = """
The solution you need to reflect is in the previous step and enclosed in <response2></response2>. Your task is to generation a critique to the solution, which should contain the analysis to the solution, the judge to the solution and some instruction about how to correct the solution if it is wrong or make it better if it is correct and then generate a new solution refined from the previous solution, and referred from the critique.
The new solution should solve the problem step by step. The final answer must be enclosed in \\boxed{{}}.
You must output in the following format and you just need to generate one critique and response:
<critique3>your critique to the solution</critique3>
<response3>the new solution refined from the initial solution</response3>
""".strip()

SYSTEM_PROMPT = """You are an expert in reflecting solution. For any given problem and solution, you must reflect on your the solution, identify possible errors or areas for improvement, and refine the solution when necessary."""

REFLECT_PROMPT = """The solution you need to reflect is in the previous step and enclosed in <response></response>. Your task is to generation a critique to the solution, which should contain the analysis to the solution, the judge to the solution and some instruction about how to correct the solution if it is wrong or make it better if it is correct and then generate a new solution refined from the previous solution, and referred from the critique.
The new solution should solve the problem step by step. The final answer must be enclosed in \\boxed{{}}.
You must output in the following format:
<critique>your critique to the solution</critique>
<response>the new solution refined from the initial solution</response>
"""



REWARD_PROMPT = """
You are a process supervision model for visual reasoning tasks. You will receive an image
and an image-based problem statement, followed by solution steps to evaluate.
You should carefully analyze each step for mathematical correctness and logical validity.

Output:
1. A step-by-step analysis (one short verdict/explanation per step).
2. A single Final Score in the range 0.00~1.00, with exactly two decimal places, reflecting the overall quality of the solution.
The score should be enclosed in <score></score>.

problem statement:{question}
solution:{response}
""".strip()

PRM_SYSTEM_PROMPT = """
**You are a process supervision model for visual reasoning tasks. You will receive an image and an image-based problem statement, followed by solution steps to evaluate.**

First round: problem statement and first solution step.  
Subsequent rounds: one new step per round.

Assess the cumulative correctness of the entire solution up to each step.

## Evaluation Criteria:

1. **Visual Accuracy**: Are visual elements from the image correctly identified (shapes, colors, positions, quantities, spatial relationships)?

2. **Logical Validity**: Do all inferences and calculations follow correctly from the image and previous steps?

## Response:
- **"<+>"** if correct up to this step
- **"<->"** if any error exists up to this step

Only respond with "<+>" or "<->". No explanations.

An error in any step invalidates all subsequent steps.
"""

REWARD_QWEN_PRM_PROMPT = """
You are a process supervision model for visual reasoning tasks. You will receive an image
and an image-based problem statement, followed by solution steps to evaluate.
You should carefully analyze each step for mathematical correctness and logical validity.

Output:
1. A step-by-step analysis (one short verdict/explanation per step).
2. A single Final Score in the range 0.00~1.00, with exactly two decimal places, reflecting the overall quality of the solution.
The score should be enclosed in <score></score>.

problem statement:{question}
solution:{response}
""".strip()