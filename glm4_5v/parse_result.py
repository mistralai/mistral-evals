import json
import re

# Load the data from the JSON file
with open('./answers_mmmu.json', 'r') as f:
    data = json.load(f)

def is_float(element: any) -> bool:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def is_fraction(s):
    if s.count('/') == 1:
        parts = s.split('/')
        if len(parts) == 2:
            numerator, denominator = parts[0], parts[1]
            try:
                int(numerator)
                int(denominator)
                if int(denominator) == 0:
                    return False
                return True
            except ValueError:
                return False
    return False

def check_equal(float_1: str, float_2: str, rounded: bool = False) -> bool:
    return round(float(float_1)) == round(float(float_2)) if rounded else float(float_1) == float(float_2)

def extract_final_answer(model_answer: str) -> str:
    """Extracts the final answer from the model_answer string."""
    text = model_answer.strip()
    if '<|begin_of_box|>' in text and '<|end_of_box|>' in text:
        begin_idx = text.rfind('<|begin_of_box|>')
        end_idx = text.find('<|end_of_box|>', begin_idx)
        if begin_idx != -1 and end_idx != -1:
            content = text[begin_idx + len('<|begin_of_box|>') : end_idx]
            content = content.strip()
            if "Final Answer:" in content or "Finally Answer:" in content:
                fa_idx = (content.rfind("Final Answer:") 
                          if content.rfind("Final Answer:") != -1 
                          else content.rfind("Finally Answer:"))
                if fa_idx != -1:
                    content = content[fa_idx + len("Final Answer:"):].strip()
            if content.lower().startswith("final answer"):
                colon_pos = content.lower().find("answer:")
                if colon_pos != -1:
                    content = content[colon_pos + len("answer:"):].strip()
            content = content.rstrip('.').strip()
            return content

    if '<|end_of_box|>' in text and '<|begin_of_box|>' not in text:
        idx = text.rfind("Final Answer")
        if idx == -1:
            idx = text.rfind("Finally Answer")
        if idx != -1:
            end_idx = text.find('<|end_of_box|>', idx)
            if end_idx != -1:
                if "Final Answer:" in text[idx:]:
                    content = text[idx + len("Final Answer:"): end_idx]
                elif "Final Answer\n" in text[idx:]:
                    content = text[idx + len("Final Answer\n"): end_idx]
                else:
                    content = text[idx + len("Finally Answer"): end_idx]
                return content.strip().rstrip('.').strip()

    for phrase in ["Final Answer:", "Finally Answer:", "Final Answer\n"]:
        idx = text.rfind(phrase)
        if idx != -1:
            content = text[idx + len(phrase):].strip()
            return content.rstrip('.').strip()
    idx = text.lower().rfind("final answer is")
    if idx != -1:
        content = text[idx + len("final answer is"):].strip()
        content = content.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
        return content.rstrip('.').strip()

    return None

total_questions = len(data)
correct_count = 0

for qid, qa in data.items():
    model_ans_str = qa["model_answer"]
    ref_answers = qa["reference_answer"]
    final_ans = extract_final_answer(model_ans_str)
    if final_ans is None:
        print(f"qid: {qid}, model answer: {repr(model_ans_str[-50:])}, ref answer: {ref_answers}")
        continue
    final_ans_norm = final_ans.strip().lower()
    ref_norms = [ref.strip().lower() for ref in ref_answers]
    if final_ans_norm in ref_norms:
        correct_count += 1

    elif is_float(final_ans_norm) or is_fraction(final_ans_norm):
        if is_fraction(final_ans_norm):
            parts = final_ans_norm.split("/")
            numerator = int(parts[0])
            denominator = int(parts[1])
            final_ans_norm = float(numerator) / float(denominator)
        for ref_ans in ref_norms:
            # NOTE: specify rounded=False for an exact match.
            if check_equal(final_ans_norm, ref_ans, rounded=True):
                correct_count += 1
    else:
        # mismatch - take last 50 chars of model answer to get the final answer
        print(f"qid: {qid}, model answer: {repr(model_ans_str[-50:])}, ref answer: {ref_answers}")

print(f"Total questions: {total_questions}")
print(f"Correctly answered: {correct_count}")
print(f"Accuracy: {correct_count/total_questions:.2%}")