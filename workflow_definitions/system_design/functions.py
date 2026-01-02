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

class VerificationResult(BaseModel):
    is_valid: bool = Field(description="True if any hypothesis is valid/viable risk that needs solving")
    best_hypothesis: str = Field(description="The text of the best valid hypothesis, or empty if none valid")
    solution_draft: str = Field(description="Brief solution draft or direction, if valid")
    reason: str = Field(description="Reason why hypotheses are invalid, or empty if valid")

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
            
    return {
        "is_valid": response.is_valid,
        "best_hypothesis": response.best_hypothesis,
        "solution_draft": response.solution_draft,
        "verification_reason": response.reason
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
