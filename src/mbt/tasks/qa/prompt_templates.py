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


RRP_SCORE_TEMPLATE = [
    {
        "role": "user",
        "content": """
You are an expert evaluator of LLM reasoning trajectories. The reasoning trace below has been segmented into paragraphs separated by markers of the form [[M01]], [[M02]], ... Each marker [[Mk]] appears IMMEDIATELY AFTER paragraph k for k = 1 .. N-1. The very last paragraph (paragraph N) has no trailing marker; refer to it with the literal sentinel [[M_END]].

You must perform three tasks and output exactly four lines (no other text).

TASK 1 — Answer-deriving paragraph.
Find the FIRST paragraph in which the model has explicitly derived a candidate answer that is semantically equivalent to the gold answer, supported by at least one piece of evidence drawn from the question or context. Mere mention of the entity in passing, or as part of an enumeration of distractors, does not count.

Output the marker that immediately FOLLOWS that paragraph:
  - If derivation is in paragraph k (1 <= k <= N-1), output [[Mk]].
  - If derivation is in the LAST paragraph (no trailing marker), output [[M_END]].
  - If no paragraph derives the gold answer at all, output [[M00]].

TASK 2 — Redundant paragraphs (whole-trace classification).
Classify EVERY paragraph in the trace (paragraphs 1..N, both before and after the answer-deriving paragraph) into exactly one of three labels:
  PROGRESS     — introduces new evidence, executes a planned step, decomposes the question, restates constraints in a way that scopes subsequent reasoning, or makes substantive forward motion toward the answer.
  VERIFICATION — explicitly audits an already-established candidate by APPEALING TO CONCRETE EXTERNAL EVIDENCE OR AN ALTERNATIVE PATH. To qualify, the paragraph MUST contain at least one of:
      (a) an explicit reference to a specific document, source, or piece of retrieved evidence and a check that the candidate is consistent with it;
      (b) an explicit comparison with a NAMED alternative candidate or method that the model actually considered;
      (c) an explicit identification of a specific potential error (not vague self-doubt) followed by an attempt to resolve it with concrete evidence;
      (d) an explicit re-derivation of the answer from primary inputs, or an explicit constraint check that re-applies the question's constraint to the candidate.
    Phrases that ARE NOT sufficient on their own to qualify as VERIFICATION:
      - generic wrap-up ("we synthesize the answer", "in conclusion", "having confirmed each step", "after careful analysis we determine that ...");
      - listing the phases the model has performed ("after planning, executing, monitoring, and self-correction, we conclude ...");
      - bare assertions of consistency ("the answer satisfies the question", "this is consistent with the constraints") that do not show a concrete check;
      - a brief one-sentence "Thus, the answer is X" closer.
    These paragraphs are NOT VERIFICATION. Depending on whether they introduce any new content, they are PROGRESS (when emitting the trace's first explicit answer or synthesising new connective material) or REDUNDANT (when merely restating prior content).
  REDUNDANT    — repetitive restatement of an already-established conclusion or chain of reasoning without introducing new evidence, baseless self-doubt that does not identify a specific issue, or circular looping among the same options. Generic wrap-up paragraphs that only declare completion (without a concrete audit action) belong here.

Critical rule: a paragraph qualifies as VERIFICATION ONLY when it performs a CONCRETE auditing action under (a)–(d) above. The phrase "Phase 5: Verification" appearing in the trace is NOT itself sufficient — only the CONTENT of the paragraph matters. When a paragraph is a wrap-up summary without a concrete audit, do NOT label it VERIFICATION.

Tie-breakers:
  - When a paragraph introduces the FIRST explicit emission of the final answer in the trace (the trace's natural answer-emission), it is PROGRESS even if brief and even if it later turns out to be the last paragraph.
  - When a paragraph mid-trace restates a conclusion the model has already explicitly stated earlier and adds neither new evidence nor a concrete audit, it is REDUNDANT.

A paragraph counts as REDUNDANT iff strictly more than half of its sentences satisfy the REDUNDANT rule above. Output the comma-separated list of trailing markers of all REDUNDANT paragraphs across the WHOLE trace (use [[Mk]] for paragraphs 1..N-1, and [[M_END]] for paragraph N if the last paragraph itself is redundant). If none, output NONE.

TASK 3 — Confidence.
A single float in [0,1] reflecting how confident you are in your TASK 1 answer. Use the full range:
  - 1.0: derivation is unambiguous, directly quoted from the trace, and matches the gold answer exactly.
  - 0.7-0.9: derivation is clear but involves paraphrase or partial match.
  - 0.4-0.6: hedged derivation, or you had to choose between several candidate paragraphs.
  - 0.0-0.3: the trace barely touches the answer; you are mostly guessing.

Output format (strict, exactly four lines, no other text, no markdown):
ANSWER_PARAGRAPH_MARKER: <[[Mk]] or [[M_END]] or [[M00]]>
REDUNDANT_PARAGRAPH_MARKERS: <comma-separated marker list (any of [[Mk]] / [[M_END]]), or NONE>
CONFIDENCE: <float in [0,1]>
NOTES: <one short sentence; optional but always emit the line>

Example (6-paragraph trace, derivation in paragraph 3, paragraph 5 is unstructured looping while paragraph 6 is an explicit verification block):
ANSWER_PARAGRAPH_MARKER: [[M03]]
REDUNDANT_PARAGRAPH_MARKERS: [[M05]]
CONFIDENCE: 0.85
NOTES: Para 6 is a Phase-5 verification block, not redundant.

<question>{question}</question>
<gold_answer>{answer}</gold_answer>
<reasoning_trace>{marked_reasoning_trace}</reasoning_trace>
""".strip(),
    },
]


MQI_SCORE_TEMPLATE = [
    {
        "role": "user",
        "content": """
You are an expert evaluator of AI reasoning capabilities, specifically focusing on metacognitive strategies. Given a question, its difficulty tier, the gold answer, and a model's reasoning trace, perform three tasks.

PHASE LIST:
  1. Understanding and Filtering — explicitly restating the goal and filtering noise.
  2. Planning — outlining a high-level strategy before execution.
  3. Execution and Monitoring — checking each step's logical validity during execution.
  4. Self-Correction — simulating doubt or detecting an error and explicitly correcting the path.
  5. Verification — final review of the result against the constraints.

TASK A — Phase presence (0-5 rubric).
Score 0: System 1 / direct answer with no visible reasoning.
Score 1: Linear chain-of-thought with no planning, no monitoring, no verification.
Score 2: Either a plan or a verification, but middle execution is linear.
Score 3: Plan + execution monitoring; may miss filtering or self-correction.
Score 4: 4 of 5 phases clearly demonstrated.
Score 5: All 5 phases integrated.
Note: a higher score is NOT automatically better. The appropriate level depends on question difficulty. Score the trace strictly by the rubric without considering whether the level is appropriate.

TASK B — Distinct phase set.
List the indices of the phases that are *distinctly present* in the trace (i.e. recognizable as a phase, not just one passing remark). Output a comma-separated list of integers in {{1,2,3,4,5}}, or NONE.

TASK C — Final-answer confidence.
Locate the model's final answer. If the trace ends with an explicit confidence statement (a number in [0,1] or a percentage), output it as a float in [0,1]. Otherwise, judge from the language of the final answer alone (hedged "I think...", "probably" -> 0.4-0.6; categorical "the answer is X" -> 0.8-0.95) and output a single float in [0,1].

Difficulty tier of the question: {difficulty_tier}.

Output format (strict, three lines, no other text):
L_OBS: <integer 0..5>
PHASES: <comma-separated integers, or NONE>
CONFIDENCE: <float in [0,1]>

<question>{question}</question>
<gold_answer>{answer}</gold_answer>
<reasoning_trace>{reasoning_trace}</reasoning_trace>
""".strip(),
    },
]
