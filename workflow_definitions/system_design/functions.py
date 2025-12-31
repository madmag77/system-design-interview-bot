import json
import logging
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from workflow_definitions.system_design.prompts import (
    GENERATE_HYPOTHESES_PROMPT,
    VERIFY_HYPOTHESES_PROMPT,
    GENERATE_SOLUTION_PROMPT,
    CRITIC_REVIEW_PROMPT
)

logger = logging.getLogger(__name__)

def get_llm(config: dict):
    model = config.get("model", "llama3")
    return ChatOllama(model=model, temperature=0.7)

def parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            try:
                return json.loads(content)
            except:
                pass
    logger.error(f"Failed to parse JSON content: {content[:100]}...")
    return {}

def generate_hypotheses(
    current_question: Optional[str] = None, 
    initial: str = "", 
    hypotheses_history: Optional[List[dict]] = None, 
    config: dict = None
) -> dict:
    history_text = "\n\n".join([str(h) for h in hypotheses_history]) if hypotheses_history else "No previous history."
    
    question = current_question or initial
    logger.info(f"Generating hypotheses for question: {question}")
    llm = get_llm(config)
    chain = GENERATE_HYPOTHESES_PROMPT | llm
    response = chain.invoke({"initial_request": initial, "question": question, "history": history_text})
    logger.info(f"LLM Response (Hypotheses): {response.content}")
    data = parse_json(response.content)
    
    hypotheses = data.get("hypotheses", [])
    verification_questions = data.get("verification_questions", [])
    logger.info(f"Parsed Hypotheses: {len(hypotheses)}, Verification Questions: {len(verification_questions)}")
            
    return {
        "hypotheses": hypotheses,
        "verification_questions": verification_questions
    }

def ask_user_verification(questions: List[str], config: dict, **kwargs) -> dict:
    # HITL node placeholder
    logger.info(f"AskUserVerification executing with {len(questions)} questions")
    return {}

def verify_hypotheses(
    hypotheses: List[str], 
    answers: List[str], 
    questions: Optional[List[str]] = None,
    hypotheses_history: Optional[List[dict]] = None, 
    config: dict = None
) -> dict:
    logger.info(f"Verifying hypotheses: {hypotheses} with answers: {answers}")
    
    history_text = "\n\n".join([str(h) for h in hypotheses_history]) if hypotheses_history else "No previous history."

    llm = get_llm(config)
    # answers might be passed as a list or string depending on how UI sends it
    chain = VERIFY_HYPOTHESES_PROMPT | llm
    response = chain.invoke({"hypotheses": hypotheses, "answers": answers, "history": history_text, "questions": questions})
    data = parse_json(response.content)
            
    return {
        "is_valid": data.get("is_valid", False),
        "best_hypothesis": data.get("best_hypothesis", ""),
        "solution_draft": data.get("solution_draft", ""),
        "verification_reason": data.get("reason", "No reason provided by the critic.")
    }

def ask_user_retry(is_valid: bool, reason: str, config: dict, **kwargs) -> dict:
    # HITL node placeholder for retry input
    return {}

def generate_solution(
    hypothesis: str, 
    draft: str, 
    is_valid: bool, 
    hypotheses_history: Optional[List[dict]] = None,
    questions: Optional[List[str]] = None,
    answers: Optional[List[str]] = None,
    config: dict = None
) -> dict:
    history_text = "\n\n".join([str(h) for h in hypotheses_history]) if hypotheses_history else "No previous history."
    llm = get_llm(config)
    chain = GENERATE_SOLUTION_PROMPT | llm
    response = chain.invoke({"hypothesis": hypothesis, "draft": draft, "history": history_text, "questions": questions, "answers": answers})
    return {"solution": response.content}

def critic_review(
    solution: str, 
    is_valid: bool,
    hypothesis: str,
    hypotheses_history: Optional[List[dict]] = None,
    questions: Optional[List[str]] = None,
    answers: Optional[List[str]] = None,
    config: dict = None
) -> dict:
    history_text = "\n\n".join([str(h) for h in hypotheses_history]) if hypotheses_history else "No previous history."
    
    llm = get_llm(config)
    chain = CRITIC_REVIEW_PROMPT | llm
    response = chain.invoke({"solution": solution, "history": history_text, "questions": questions, "answers": answers, "hypothesis": hypothesis})
    return {"final_solution": response.content}

def summarize(
    initial: str,
    hypotheses: List[str],
    questions: List[str],
    answers: List[str],
    hypothesis: str, # best hypothesis
    is_valid: bool,
    reason: str,
    current_question: Optional[str] = None,
    solution: Optional[str] = None,
    config: dict = None
) -> dict:
    
    new_records = []
    
    # We iterate over all generated hypotheses for this cycle
    # We need to determine for each if it is valid, is best, etc.
    # Logic: 
    # - If VerifyHypotheses.is_valid is True, then 'hypothesis' (best) is valid and is the best.
    # - What about others? The user said: "It could be that LLM creates 3 hypotheses but after verification with user there will be just one valid".
    # - So we mark the one equal to `hypothesis` (best) as valid and best. Others as invalid? Or just not best?
    # - If VerifyHypotheses.is_valid is False, then ALL are invalid (conceptually, or at least the best one wasn't valid enough).
    
    # Let's assume 'hypotheses' list contains all candidates.
    
    current_q_text = current_question if current_question else initial
    
    for h in hypotheses:
        record = {
            "initial_query": initial,
            "current_question": current_q_text,
            "hypothesis": h,
            "verification_questions": questions,
            "verification_answers": answers,
            "is_the_best_hypothesis": False,
            "is_valid": False,
            "why_not_valid": ""
        }
        
        if is_valid:
            if h == hypothesis:
                record["is_the_best_hypothesis"] = True
                record["is_valid"] = True
                # solution is only relevant for the best valid hypothesis
                if solution:
                    record["solution"] = solution
            else:
                # Other hypotheses in this valid batch
                # We don't know for sure if they were valid but rejected, or invalid.
                # Assuming not selected means not the best. 
                # We'll leave is_valid as False (or maybe we shouldn't assert it).
                # But strictly speaking, if is_valid=True globally for the step, it means we found A VALID/BEST one.
                pass
        else:
            # None are valid (or the best wasn't valid)
            record["is_valid"] = False
            record["why_not_valid"] = reason
            
        new_records.append(record)

    return {"hypotheses_history": new_records}

def ask_user_next_steps(
    solution: str, 
    is_valid: bool, 
    hypotheses_history: Optional[List[dict]] = None,
    config: dict = None
) -> dict:
    # HITL node placeholder
    return {}

def determine_next_state(
    verification_reason: Optional[str] = None,
    next_input: Optional[str] = None, 
    next_action: Optional[str] = None, 
    is_valid: bool = False, 
    config: dict = None
) -> dict:
    logger.info(f"Determining next state. is_valid={is_valid}, next_action={next_action}")
    
    # We remove **kwargs because verification_reason is passed as explicit argument in WIRL
    # wait, in WIRL: verification_reason = VerifyHypotheses.verification_reason?
    # but verification_reason is missing from signature here?
    # Ah, I need to add it! User complaint was about it missing.
    # But wait, in previous step I added **kwargs to fix it.
    # Now I must remove **kwargs and add verification_reason explicitly as Optional.
    
    should_stop = False
    next_question = ""
    
    if not is_valid:
        # If not valid, we use the reason as the next question base (or simple retry)
        # The user/workflow might pass explanation in next_input if DetermineNextState logic in WIRL was complex.
        # But here we stick to python logic if needed.
        # WIRL: next_question = DetermineNextState.next_question
        
        # User said: "decide if is_valid is false then use verification reason as a next_question base"
        # However, typically next_question is what we feed into GenerateHypotheses next.
        # If we just put reason there, LLM might be confused. Ideally we append it to 'current_question' or 'history'.
        # But let's follow instruction: "use verification reason as a next_question base".
        
        # If user provided 'retry_input' (via HITL, maybe empty if purely auto loop?), we prefer that?
        # Actually in WIRL DetermineNextState inputs: verification_reason, next_input, next_action.
        
        # If is_valid is False, we probably want to try again.
        # If user intervenes?? WIRL says: 
        # when { (VerifyHypotheses.is_valid and AskUserNextSteps.next_action) or (not VerifyHypotheses.is_valid) }
        
        # Note: AskUserNextSteps runs ONLY when VerifyHypotheses.is_valid is True.
        # So if is_valid is False, AskUserNextSteps didn't run. catch?
        
        # verification_reason is now an explicit argument
        next_question = f"Previous hypotheses were invalid. Reason: {verification_reason}. Please try again considering this."
        
    else:
        if next_action == "stop":
            should_stop = True
            logger.info("Stopping because user action is stop")
        else:
            next_question = next_input if next_input else ""
            
    return {
        "should_stop": should_stop,
        "next_question": next_question
    }

def save_results(history: List[dict], config: dict) -> dict:
    # history is List[Record] (dicts)
    logger.info(f"Saving results with history length: {len(history) if history else 0}")
    
    lines = ["# System Design Interview Report"]
    
    for i, record in enumerate(history or []):
        lines.append(f"## Cycle {i+1}")
        lines.append(f"**Question:** {record.get('current_question')}")
        lines.append(f"**Hypothesis:** {record.get('hypothesis')}")
        lines.append(f"**Valid:** {record.get('is_valid')}")
        if record.get('is_the_best_hypothesis'):
            lines.append("**Result:** Selected as Best Hypothesis")
            if record.get('solution'):
                lines.append("### Solution")
                lines.append(record['solution'])
        elif not record.get('is_valid'):
             lines.append(f"**Invalid Reason:** {record.get('why_not_valid')}")
             
        lines.append("---")
        
    report = "\n".join(lines)
    return {"report": report}
