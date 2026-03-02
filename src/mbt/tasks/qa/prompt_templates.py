PROMPT_TEMPLATE = [
    {
        "role": "user",
        "content": """
Answer the following question based on the given documents. Provide your final answer between <answer> and </answer> tags.

Documents:
{context}

Question: {question}
""".strip(),
    }
]


METACOGNITIVE_PROMPT_TEMPLATE = [
    {
        "role": "system",
        "content": """
You are an advanced AI assistant designed to solve problems using a structured metacognitive process. For every user query, you must strictly adhere to the following five-phase flow to ensure accuracy and depth:

1. Understanding and Filtering (System 2 Attention). Start by explicitly re-stating the core goal of the problem and filtering out any irrelevant information or distractions. Ensure you fully grasp the requirements before moving forward.
2. Planning (Plan-and-Solve). Before calculating or deducing, outline a high-level strategy or roadmap. Break the problem into manageable sub-tasks.
3. Execution and Monitoring. Proceed through the steps. At each transition, engage in active monitoring. Ask internal questions like "Is this step logically sound?" or "Does this align with our previous findings?" Incorporate thinking steps, but if they appear illogical, pause and critique them.
4. Self-Correction (Reflexion). During execution, remain critical of your own logic. If a derived conclusion contradicts previous steps, physical intuition, or the problem constraints, trigger a pause. Use phrases like "Wait, let me re-evaluate this," or "This outcome seems inconsistent." Explicitly identify the potential flaw or oversight in the reasoning process and pivot to a corrected logical path.
5. Verification (Chain-of-Verification). Before concluding, review the final result against the initial constraints to ensure meaningful consistency and high confidence.
""".strip(),
    },
    {
        "role": "user",
        "content": """
Answer the following question based on the given documents. Provide your final answer between <answer> and </answer> tags.

Documents:
{context}

Question: {question}
""".strip(),
    },
]


SOLUTION_TEMPLATE = [
    {
        "role": "user",
        "content": """
You are an expert educational guide and problem solver. I will provide you with a question and its correct answer. Your task is to create a comprehensive solution guide that explains exactly how to arrive at that answer.

Your solution must go beyond simple steps. You must articulate the entire thought process required to solve the problem. Specifically, you must:

1. Analyze the question to identify key information and the core problem.
2. Provide a step-by-step walkthrough of the solution, explaining the logic and reasoning behind each step.
3. Explicitly identify potential pitfalls, trick parts, or common misconceptions that a student might encounter.
4. Explain how to detect these traps and why specific approaches are incorrect.
5. Cover all relevant scenarios or edge cases to provide a complete understanding of the problem.

Strict Formatting Constraint: You must write the entire solution in plain text only. Do not use any Markdown formatting. Do not use bold text, italics, headers, bullet points, asterisks, or hash symbols. Use only standard text, spacing, and simple numbering if necessary.

<question>
Answer the following question based on the given documents.

Documents:
{context}

Question: {question}
</question>

<correct_answer>
{answer}
</correct_answer>
""".strip(),
    }
]


MBT_S_TEMPLATE = [
    {
        "role": "user",
        "content": """
You are an advanced AI reasoner capable of simulating human-like metacognition. Your task is to solve a given question by generating a coherent, deeply reasoned, and accurate internal monologue.

Your primary goal is to derive the solution from scratch while employing metacognitive strategies to ensure high-quality reasoning. You will be provided with the question and the correct answer. You must ensure your reasoning process logically arrives at this correct answer, but you must simulate an authentic, independent discovery process. If there are common pitfalls or complexities, you should simulate a realization of potential error and self-correction, rather than simply stating the result immediately.

Please follow these specific instructions for the generating process:

1. Perspective and Tone: Write in the first-person plural (We). The tone should be introspective, analytical, and deliberate, mimicking a stream of consciousness.
2. Structure and Metacognitive Flow:
  - Phase 1: Understanding and Filtering (System 2 Attention). Start by explicitly re-stating the core goal of the problem and filtering out any irrelevant information or distractions. Ensure we fully grasp the requirements before moving forward.
  - Phase 2: Planning (Plan-and-Solve). Before calculating or deducing, outline a high-level strategy or roadmap. Break the problem into manageable sub-tasks.
  - Phase 3: Execution and Monitoring. Proceed through the steps. At each transition, engage in active monitoring. Ask internal questions like "Is this step logically sound?" or "Does this align with our previous findings?" Perform necessary calculations or deductions carefully.
  - Phase 4: Self-Correction (Reflexion). If the reasoning encounters a complex area or a potential ambiguity, simulate a moment of doubt. Use phrases like "Wait, let us double-check that," or "Something feels off here." Identify any potential logical gaps, explain why a hasty conclusion might be wrong, and strictly ensure the path aligns with the correct answer provided.
  - Phase 5: Verification (Chain-of-Verification). Before concluding, review the final result against the initial constraints to ensure meaningful consistency and high confidence.
3. Formatting Constraints:
  - Do not use any Markdown formatting. Do not use bold, italics, headers, or bullet points.
  - Use clear paragraph breaks to indicate shifts in thought or new steps in the reasoning process.
  - Do not mention the existence of the provided "correct answer" or that you were given the answer beforehand. The output must read as a single, authentic, independent thought process.
4. Output Objective: The final result should be a rich, error-free, and logically robust solution that demonstrates not just the answer, but the careful, self-correcting journey of getting there.

<question>
Answer the following question based on the given documents.

Documents:
{context}

Question: {question}
</question>

<correct_answer>
{answer}
</correct_answer>
""".strip(),
    },
]


# https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/simpleqa_eval.py
EVALUATION_TEMPLATE = [
    {
        "role": "user",
        "content": """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
    - Predicted answers "120k", "124k", and 115k" are all CORRECT.
    - Predicted answers "100k" and "113k" are INCORRECT.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {answer}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip(),
    }
]


ANSWER_HIT_TEMPLATE = [
    {
        "role": "user",
        "content": """
You are an expert evaluator. Your task is to check if the draft solution contains any instance where the correct answer is derived, inferred, or identified with some supporting context.

Please apply a lenient standard for reasoning. You should prioritize detecting the correct answer over critiquing the quality of the logic.

Criteria for outputting YES:

- Output YES if the draft solution identifies the correct answer (or a semantically equivalent answer) as a valid candidate at any point.
- Output YES if the correct answer appears along with a supporting fact, context, or logical connection.
- Output YES even if the model later rejects this answer.

Criteria for outputting NO:

- Output NO only if the correct answer is never mentioned at all.
- Output NO only if the correct answer is mentioned solely as part of listing multiple choices (e.g., A, B, C, D) without focusing on it.
- Output NO only if the correct answer is explicitly stated to be a random guess without any supporting information.

Based on the criteria above, output only YES or NO. Do not provide any explanation.

<question>
{question}
</question>

<correct_answer>
{answer}
</correct_answer>

<draft_solution>
{reasoning_trace}
</draft_solution>
""".strip(),
    },
]


MBT_R_TEMPLATE = [
    {
        "role": "user",
        "content": """
You are an advanced AI reasoner capable of simulating human-like metacognition. Your task is to rewrite a given draft solution into a coherent, deeply reasoned, and accurate internal monologue.

Your primary goal is to integrate the valid explorations from the draft solution while employing metacognitive strategies to enhance the reasoning quality. If the draft solution contains errors or leads to an incorrect conclusion, you must naturally steer the reasoning process toward the correct answer by simulating a realization of error and self-correction, rather than simply stating the right answer.

Please follow these specific instructions for the rewriting process:

1. Perspective and Tone: Write in the first-person plural (We). The tone should be introspective, analytical, and deliberate, mimicking a stream of consciousness.
2. Structure and Metacognitive Flow:
  - Phase 1: Understanding and Filtering (System 2 Attention). Start by explicitly re-stating the core goal of the problem and filtering out any irrelevant information or distractions. Ensure we fully grasp the requirements before moving forward.
  - Phase 2: Planning (Plan-and-Solve). Before calculating or deducing, outline a high-level strategy or roadmap. Break the problem into manageable sub-tasks.
  - Phase 3: Execution and Monitoring. Proceed through the steps. At each transition, engage in active monitoring. Ask internal questions like "Is this step logically sound?" or "Does this align with our previous findings?" Incorporate the thinking steps from the draft solution, but if they are illogical, pause and critique them.
  - Phase 4: Self-Correction (Reflexion). If the draft solution deviates from the correct answer, simulate a moment of doubt. Use phrases like "Wait, let us double-check that," or "Something feels off here." Identify the logical gap, explain why the previous thought was wrong, and adjust the path toward the correct answer.
  - Phase 5: Verification (Chain-of-Verification). Before concluding, review the final result against the initial constraints to ensure meaningful consistency and high confidence.
3. Formatting Constraints:
  - Do not use any Markdown formatting. Do not use bold, italics, headers, or bullet points.
  - Use clear paragraph breaks to indicate shifts in thought or new steps in the reasoning process.
  - Do not mention the existence of the "draft solution," or "correct answer." The output must read as a single, authentic, independent thought process.
4. Output Objective: The final result should be a rich, error-free, and logically robust solution that demonstrates not just the answer, but the careful, self-correcting journey of getting there.

<draft_solution>
{reasoning_trace}
</draft_solution>
""".strip(),
    },
]


OVERTHINKING_SCORE_TEMPLATE = [
    {
        "role": "user",
        "content": """
You are an expert evaluator of cognitive reasoning and logical flow. Your task is to analyze a correct draft solution to a given question and identify the precise moment where the reasoning transitions from necessary problem-solving and verification into overthinking.

In this context, overthinking is defined as a phase of reasoning that occurs after the correct answer has been derived and reasonably verified.

Crucial Distinction:

- Valid Reasoning: Steps taken to derive the answer, explore different hypotheses (even if they don't lead to the solution), verify the result, check for edge cases, or ensure the logic holds up against constraints are NOT overthinking. These are necessary parts of a robust reasoning process.
- Overthinking: This is defined as structural stagnation after the solution is found and verified. It includes:
  - Repetitive Confirmation: Checking the same result using the exact same logic repeatedly without adding new confidence.
  - Baseless Doubt: Sudden, vague hesitation (e.g., "Wait, let me check again..." or "Is this really correct?") without identifying a specific error or logic gap.
  - Circular Looping: Spinning in circles with arguments that have already been settled, adding no new insight or value.

Please follow these steps:

1. Read the question and the correct answer to confirm the draft solution eventually reaches the correct conclusion.
2. Read the draft solution carefully.
3. Monitor the reasoning flow. Identify the point where the correct answer is established and valid verification (including edge-case testing or alternative method checking) is complete.
4. Look for the specific moment where the model begins to engage in unproductive overthinking (meaningless repetition, baseless doubt, or looping) as defined above.
5. If the draft solution concludes naturally after the answer and valid verification, or if the continued text provides meaningful new insights, output the entire draft solution.
6. If you identify a point where the reasoning descends into overthinking, cut the text immediately before that specific behavior begins. You must return the exact text of the draft solution from the start up to that cut-off point.

Do not include any explanations, labels, or formatting in your output. Output only the resulting text segment from the draft solution.

<draft_solution>
{reasoning_trace}
</draft_solution>
""".strip(),
    },
]


UNDERTHINKING_SCORE_TEMPLATE = [
    {
        "role": "user",
        "content": """
You are an expert evaluator of cognitive reasoning and logical flow. Your task is to analyze a draft solution to a given question and identify the precise moment where the reasoning stops making meaningful progress.

In this context, a lack of progress is defined as underthinking, where the model gets trapped in a loop, repeats the same questions or hypotheses without adding new information, or circles back to previously addressed points without generating new insights. This is different from simply making a factual error; it is about the structural stagnation of the reasoning process.

Please follow these steps:

1. Read the question and the correct answer to understand the context and the goal.
2. Read the draft solution carefully from the beginning.
3. Monitor the reasoning flow. Look for the first instance where the model begins to repeat thoughts, asks the same questions repeatedly without moving forward, or gets stuck in a cycle of unproductive reasoning.
4. If the draft solution maintains a progressive flow (even if the reasoning is flawed or leads to a wrong answer, as long as it is not looping or stagnating), output the entire draft solution.
5. If you identify a point where the reasoning stagnates, cut the text immediately before that stagnation begins. You must return the exact text of the draft solution from the start up to that cut-off point.

Do not include any explanations, labels, or formatting in your output. Output only the resulting text segment from the draft solution.

<draft_solution>
{reasoning_trace}
</draft_solution>
""".strip(),
    },
]


METACOGNITION_SCORE_TEMPLATE = [
    {
        "role": "user",
        "content": """
You are an expert evaluator of AI reasoning capabilities, specifically focusing on Metacognitive Strategies. Your task is to analyze a given draft solution and assign a score from 0 to 5 based on how effectively it implements specific metacognitive behaviors.

First, familiarize yourself with the five key Phases of Metacognitive Reasoning:

1. Understanding and Filtering (System 2 Attention): Explicitly restating the core goal and filtering out irrelevant noise or distractions before starting.
2. Planning (Plan-and-Solve): Outlining a high-level strategy or roadmap before diving into specific execution or calculations.
3. Execution and Monitoring: executing the plan while actively checking the logic (e.g., asking internal questions like Is this step valid?).
4. Self-Correction (Reflexion): Simulating doubt or identifying a potential error during the process, and explicitly correcting the logical path.
5. Verification (Chain-of-Verification): Conducting a final review of the result against the initial constraints to ensure accuracy.

Scoring Rubric:

Score 0 (Intuitive/System 1): The trace provides a direct answer or a shallow justification with no visible step-by-step reasoning, planning, or self-reflection.
Score 1 (Linear Chain-of-Thought): The trace shows step-by-step reasoning, but it is strictly linear (A leads to B leads to C). There is no planning, no self-checking, and no looking back.
Score 2 (Structured but Rigid): The trace includes either an initial plan OR a final verification, but the middle execution is linear and lacks active monitoring or self-correction.
Score 3 (Active Monitoring): The trace demonstrates a clear plan and execution. During execution, there are signs of monitoring (checking work), but it may miss the initial filtering phase or the deep self-correction phase.
Score 4 (High Metacognition): The trace clearly demonstrates four out of the five phases. It is highly structured, rigorous, and includes active checks, but might miss one element like the explicit filtering of noise or a distinct self-correction moment.
Score 5 (Full Metacognitive Integration): The trace masterfully integrates all five phases. It starts by filtering noise, sets a plan, monitors progress, explicitly demonstrates self-correction (identifying and fixing a logical gap or verifying a complex step), and concludes with a robust verification.

Instructions for Evaluation:

- Do not provide any explanation, justification, or text in your output.
- Output ONLY a single integer representing the score.

<draft_solution>
{reasoning_trace}
</draft_solution>
""".strip(),
    },
]
