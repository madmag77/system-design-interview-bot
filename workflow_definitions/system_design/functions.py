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
from pydantic import BaseModel, Field

# Define Pydantic Models for Structured Output

class HypothesesList(BaseModel):
    hypotheses: List[str] = Field(description="List of distinct hypotheses regarding potential bottlenecks/risks")
    verification_questions: List[str] = Field(description="List of specific verification questions to ask the interviewer")

class HypothesisVerification(BaseModel):
    hypothesis: str = Field(description="The text of the hypothesis being verified")
    is_valid: bool = Field(description="True if this specific hypothesis is valid")
    reason: Optional[str] = Field(description="Reason provided if invalid, or other comments")
    is_best: bool = Field(description="True if this is selected as the best interesting hypothesis")

class VerificationResult(BaseModel):
    hypotheses_feedback: List[HypothesisVerification] = Field(description="List of verification details for each hypothesis")
    solution_draft: Optional[str] = Field(description="Brief solution draft for the best hypothesis")

logger = logging.getLogger(__name__)

def get_llm(config: dict):
    model = config.get("model", "llama3")
    return ChatOllama(model=model, temperature=0.7)

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
    structured_llm = llm.with_structured_output(HypothesesList, method="json_schema")
    chain = GENERATE_HYPOTHESES_PROMPT | structured_llm
    
    # response will be an instance of HypothesesList
    response = chain.invoke({"initial_request": initial, "question": question, "history": history_text})
    
    hypotheses = response.hypotheses
    verification_questions = response.verification_questions
    logger.info(f"Generated Hypotheses: {len(hypotheses)}, Verification Questions: {len(verification_questions)}")
            
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
    structured_llm = llm.with_structured_output(VerificationResult, method="json_schema")
    # answers might be passed as a list or string depending on how UI sends it
    chain = VERIFY_HYPOTHESES_PROMPT | structured_llm
    
    # response is VerificationResult instance
    response = chain.invoke({"hypotheses": hypotheses, "answers": answers, "history": history_text, "questions": questions})
    
    # Determine global validity and best hypothesis from detailed feedback
    feedback = response.hypotheses_feedback
    is_valid_global = any(h.is_valid for h in feedback)
    
    best_h = next((h.hypothesis for h in feedback if h.is_best), "")
    if not best_h and is_valid_global:
         # Fallback: pick first valid
         best_h = next((h.hypothesis for h in feedback if h.is_valid), "")

    # Construct global reason from invalid hypotheses if global is invalid
    global_reason = ""
    if not is_valid_global:
        reasons = [f"{h.hypothesis}: {h.reason}" for h in feedback if h.reason]
        global_reason = "; ".join(reasons) or "No valid hypotheses found."

    logger.info(f"Verification Result: Valid={is_valid_global}, Best={best_h}")
            
    return {
        "is_valid": is_valid_global,
        "best_hypothesis": best_h,
        "solution_draft": response.solution_draft or "",
        "verification_reason": global_reason,
        "verification_details": [h.model_dump() for h in feedback]
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
    verification_details: Optional[List[dict]] = None,
    config: dict = None
) -> dict:
    
    new_records = []
    current_q_text = current_question if current_question else initial

    if verification_details:
        # We have detailed per-hypothesis feedback
        for item in verification_details:
             # item is dict from HypothesisVerification
             h_text = item.get("hypothesis")
             h_valid = item.get("is_valid", False)
             h_reason = item.get("reason", "")
             h_is_best = item.get("is_best", False)
             
             record = {
                "initial_query": initial,
                "current_question": current_q_text,
                "hypothesis": h_text,
                "verification_questions": questions,
                "verification_answers": answers,
                "is_the_best_hypothesis": h_is_best,
                "is_valid": h_valid,
                "why_not_valid": h_reason
            }
             
             if h_is_best and solution:
                 record["solution"] = solution
                 
             new_records.append(record)
    else:
        # Fallback for old behavior (should not happen if WIRL updated)
        # We iterate over all generated hypotheses for this cycle
    
        if is_valid:
            if h == hypothesis:
                record["is_the_best_hypothesis"] = True
                record["is_valid"] = True
                # solution is only relevant for the best valid hypothesis
                if solution:
                    record["solution"] = solution
            else:
                pass
        else:
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
        lines.append(f"### Hypothesis {i+1}")
        lines.append(f"**Question:** {record.get('current_question')}\n")
        lines.append(f"**Hypothesis:** {record.get('hypothesis')}\n")
        
        # Verification Q&A
        questions = record.get('verification_questions', [])
        answers = record.get('verification_answers', [])
        if questions and answers:
            lines.append("**Verification:**")
            for q, a in zip(questions, answers):
                lines.append(f"- **Q:** {q}")
                lines.append(f"  **A:** {a}")
            lines.append("") # Empty line

        is_valid = record.get('is_valid')
        if is_valid:
            lines.append("**Status:** Valid")
            if record.get('is_the_best_hypothesis'):
                lines.append("**Result:** Selected as Best Hypothesis")
                if record.get('solution'):
                    lines.append("\n#### Solution")
                    lines.append(record['solution'])
        else:
            lines.append("**Status:** Invalid")
            reason = record.get('why_not_valid') or "No reason provided."
            lines.append(f"**Invalid Reason:** {reason}")
             
        lines.append("\n---\n")
        
    report = "\n".join(lines)
    return {"report": report}
