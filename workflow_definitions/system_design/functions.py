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

def generate_hypotheses(current_question: str, initial: str, config: dict, **kwargs) -> dict:
    # History comes from kwargs.get("history") which is List[str] or None
    history = kwargs.get("history") or []
    history_text = "\n\n".join(history) if history else "No previous history."
    
    question = current_question or initial
    logger.info(f"Generating hypotheses for question: {question}")
    llm = get_llm(config)
    chain = GENERATE_HYPOTHESES_PROMPT | llm
    response = chain.invoke({"question": question, "history": history_text})
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

def verify_hypotheses(hypotheses: List[str], answers: List[str], config: dict) -> dict:
    logger.info(f"Verifying hypotheses: {hypotheses} with answers: {answers}")
    llm = get_llm(config)
    # answers might be passed as a list or string depending on how UI sends it
    chain = VERIFY_HYPOTHESES_PROMPT | llm
    response = chain.invoke({"hypotheses": hypotheses, "answers": answers})
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

def generate_solution(hypothesis: str, draft: str, config: dict, **kwargs) -> dict:
    llm = get_llm(config)
    chain = GENERATE_SOLUTION_PROMPT | llm
    response = chain.invoke({"hypothesis": hypothesis, "draft": draft})
    return {"solution": response.content}

def critic_review(solution: str, config: dict, **kwargs) -> dict:
    llm = get_llm(config)
    chain = CRITIC_REVIEW_PROMPT | llm
    response = chain.invoke({"solution": solution})
    return {"final_solution": response.content}

def ask_user_next_steps(solution: str, config: dict, **kwargs) -> dict:
    # HITL node placeholder
    return {}

def determine_next_state(
    retry_input: Optional[str], 
    next_input: Optional[str], 
    next_action: Optional[str], 
    is_valid: bool, 
    config: dict
) -> dict:
    logger.info(f"Determining next state. is_valid={is_valid}, next_action={next_action}")
    should_stop = False
    next_question = ""
    
    if not is_valid:
        # User provided new input to retry
        next_question = retry_input if retry_input else ""
        if not next_question:
            should_stop = True # Or handle as error
            logger.info("Stopping because not valid and no retry input")
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

def save_results(history: List[str], config: dict) -> dict:
    # In a real app, we might save to file here, but the user said "don't use persistance" 
    # (meaning intermediate results), but "safe results to a file" at the end.
    # We'll return it and let the UI handle download or save.
    logger.info(f"Saving results with history length: {len(history) if history else 0}")
    # Join history with separators for the report
    report = "\n\n" + "="*80 + "\n\n".join(history or [])
    return {"report": report}
