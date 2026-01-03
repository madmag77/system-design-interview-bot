import csv
import sys
import datetime
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langgraph.errors import GraphInterrupt
from wirl_pregel_runner.pregel_graph_builder import build_pregel_graph
from evaluation.simulated_interviewer import SimulatedInterviewer

# Import REAL functions
from workflow_definitions.system_design.functions import (
    generate_hypotheses,
    verify_hypotheses,
    generate_solution,
    critic_review,
    ask_user_next_steps,
    save_results,
    summarize,
    determine_next_state,
    ask_user_verification,
    ask_user_retry
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tasks(file_path):
    tasks = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)
    return tasks

def run_evaluation_loop():
    # Setup LLM for simulated interviewer
    eval_llm = ChatOllama(model="gpt-oss:20b", temperature=0.0, reasoning="high")
    interviewer = SimulatedInterviewer(eval_llm)
    
    # Setup Functions Map for WIRL
    functions_map = {
        "generate_hypotheses": generate_hypotheses,
        "ask_user_verification": ask_user_verification,
        "verify_hypotheses": verify_hypotheses,
        "ask_user_retry": ask_user_retry,
        "generate_solution": generate_solution,
        "critic_review": critic_review,
        "summarize": summarize,
        "ask_user_next_steps": ask_user_next_steps,
        "determine_next_state": determine_next_state,
        "save_results": save_results
    }
    
    workflow_path = "workflow_definitions/system_design/workflow.wirl"
    
    tasks = load_tasks("evaluation/task_1.csv")
    results = []
    
    for task in tasks:
        task_id = task['task_id']
        logger.info(f"Starting Task {task_id}: {task['initial_prompt']}")
        
        # Build Graph per task to ensure fresh state (though we use unique thread_id)
        app = build_pregel_graph(workflow_path, functions_map, checkpointer=MemorySaver())
        thread_id = f"eval_task_{task_id}_{datetime.datetime.now().timestamp()}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Phase 1
        try:
            logger.info("Phase 1: Initial Request")
            app.invoke({"initial_question": task['initial_prompt']}, config)
        except GraphInterrupt:
            # We are likely at AskUserVerification (or Answers input)
            # The workflow interrupts BEFORE the node that needs input? 
            # Actually, WIRL interrupts usually mean "I need input to proceed".
            # Interrupt 1: AskUserVerification -> we need to provide answers.
            pass
            
        # Check state to see what invocation happened.
        # But simply, we know the flow: Interrupt -> We provide Answers.
        
        # We need to know WHAT questions were asked.
        # Typically we inspect the state or the snapshot.
        snapshot = app.get_state(config)
        # The snapshot values should contain the questions if they were in the state.
        # WIRL state structure depends on the definition.
        # Assuming the state has 'start.verification_questions' or similar?
        # Actually proper way: inspect the last message or state keys.
        # For this prototype, I assume we are at the point needing answers.
        
        # Let's extract questions from state.
        # Based on functions.py, generate_hypotheses returns {verification_questions: ...}
        # The 'InterviewLoop' likely holds the state.
        state_values = snapshot.values
        # Need to allow loose finding of questions
        questions = []
        if isinstance(state_values, dict):
             # Try to find recent questions
             # This is tricky without knowing exact WIRL state schema perfectly, 
             # but let's assume we can find list of strings that look like questions
             # Or we blindly ask the interviewer to "answer pending questions"
             pass
        
        # For specific logic:
        # The 'ask_user_verification' node returns empty dict, but 'verify_hypotheses' needs 'answers'.
        # The interruption happens. We need to resume with Command(resume={"answers": [...]})
        
        # We need to find specific questions to ask the interviewer LLM.
        # Let's assume they are stored in the state under 'GenerateHypotheses'.
        # Or we can just ask the interviewer to "generate generic answers based on context" if we can't find them easily.
        # But let's try to get them.
        
        latest_questions = state_values.get("GenerateHypotheses", {}).get("verification_questions", [])
        if not latest_questions:
             # Look deeper?
             # For now fallback to generic
             latest_questions = ["Clarify scale", "Clarify strictness"]
             
        answers = interviewer.answer_verification(latest_questions, task['context_phase_1'])
        logger.info(f"Generated Answers Phase 1: {answers}")
        
        try:
            app.invoke(Command(resume=answers), config)
        except GraphInterrupt:
            # Interrupt 2: AskUserNextSteps (Solution generated, now what?)
            pass
            
        # We assume we reached phase 2 boundary
        
        # Phase 2: Inject Challenge
        logger.info("Phase 2: Injecting Challenge")
        challenge = interviewer.generate_challenge(task['context_phase_2'])
        logger.info(f"Challenge: {challenge}")
        
        # Resume with next_action="continue" and new_input=challenge
        try:
             # NextSteps expects a dict or object?
             # streamlit_app.py L183: st.session_state.resume_value = {"next_action": next_action, "new_input": new_input}
             # So this IS a dict.
            app.invoke(Command(resume={"next_action": "continue", "new_input": challenge}), config)
        except GraphInterrupt:
             # Interrupt 3: Verification for Phase 2
             pass
        
        # Again, get questions
        snapshot = app.get_state(config)
        state_values = snapshot.values
        # We need the NEW questions.
        latest_questions = state_values.get("GenerateHypotheses", {}).get("verification_questions", [])
        
        answers_p2 = interviewer.answer_verification(latest_questions, task['context_phase_2'])
        logger.info(f"Generated Answers Phase 2: {answers_p2}")
        
        try:
            app.invoke(Command(resume=answers_p2), config)
        except GraphInterrupt:
            # Interrupt 4: Next Steps Phase 2
            pass
            
        # Finish
        app.invoke(Command(resume={"next_action": "stop", "new_input": ""}), config)
        
        # Get Final Report
        snapshot = app.get_state(config)
        # The return value of start is in 'values'
        # If SaveResults was the last node, its output should remain in the state or we check the message history.
        # But 'values' contains the latest state. 'SaveResults' returns a dict which updates the state keys.
        # So 'report' should be a top-level key in values if the graph schema has it.
        # Let's inspect 'values' directly or try to find it.
        
        final_report = snapshot.values.get("SaveResults.report")
        if not final_report:
            # Fallback: check SaveResults specific node output if persisted specifically (unlikely in basic StateGraph unless partitioned)
            # Or iterate messages?
            # Actually, `save_results` returns `{"report": report}`. 
            # If the State has a `report` key, it will be updated.
            # We need to verify the State schema in `functions.py` or wherever the graph is built.
            # Assuming typical LangGraph where functions return dicts of updates.
            logger.info(f"Snapshot values keys: {snapshot.values.keys()}")
            pass
        
        final_report = final_report or "No Report Found"
        
        # Score
        score_data = interviewer.score_report(final_report, task['ideal_outcome'])
        logger.info(f"Score: {score_data}")
        
        results.append({
            "task_id": task_id,
            "score": score_data['score'],
            "reasoning": score_data['reasoning'],
            "final_report": final_report
        })

    # Save Results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"eval_reports/results_{timestamp}.csv"
    with open(out_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "score", "reasoning", "final_report"])
        writer.writeheader()
        writer.writerows(results)
        
    logger.info(f"Evaluation complete. Results saved to {out_file}")

if __name__ == "__main__":
    run_evaluation_loop()
