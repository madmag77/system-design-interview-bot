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

import argparse

def run_evaluation_loop(tasks_file="evaluation/tasks.csv"):
    # Setup LLM for simulated interviewer
    eval_llm = ChatOllama(model="gpt-oss:20b", temperature=0.0, reasoning="medium")
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
    
    logger.info(f"Loading tasks from {tasks_file}")
    tasks = load_tasks(tasks_file)
    results = []
    
    for task in tasks:
        task_id = task['task_id']
        logger.info(f"Starting Task {task_id}: {task['initial_prompt']}")
        
        # Build Graph per task to ensure fresh state
        app = build_pregel_graph(workflow_path, functions_map, checkpointer=MemorySaver())
        thread_id = f"eval_task_{task_id}_{datetime.datetime.now().timestamp()}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Phase 1
        try:
            logger.info("Phase 1: Initial Request")
            app.invoke({"initial_question": task['initial_prompt']}, config)
        except GraphInterrupt:
            # Expected interrupt for verification questions
            pass
            
        # Handle Phase 1 Verification Loop (handles retries if hypotheses invalid)
        logger.info("Entering Phase 1 Verification Loop")
        handle_verification_loop(app, config, interviewer, task['context_phase_1'])
        
        # Phase 2: Inject Challenge
        logger.info("Phase 2: Injecting Challenge")
        challenge = interviewer.generate_challenge(task['context_phase_2'])
        logger.info(f"Challenge: {challenge[:100]}...")
        
        try:
            app.invoke(Command(resume={"next_action": "continue", "new_input": challenge}), config)
        except GraphInterrupt:
             pass
        
        # Handle Phase 2 Verification Loop
        logger.info("Entering Phase 2 Verification Loop")
        handle_verification_loop(app, config, interviewer, task['context_phase_2'])
            
        # Finish
        app.invoke(Command(resume={"next_action": "stop", "new_input": ""}), config)

        # Get Final Report
        snapshot = app.get_state(config)
        final_report = snapshot.values.get("SaveResults.report")
        if not final_report:
            logger.error(f"Report NOT FOUND. Available keys in state: {list(snapshot.values.keys())}")
        final_report = final_report or "No Report Found"
        
        # Score
        score_data = interviewer.score_report(final_report, task['ideal_outcome'])
        logger.info(f"Score: {score_data['score']}")
        
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

def handle_verification_loop(app, config, interviewer, context):
    """
    Continues to provide answers to verification questions until the workflow
    reaches the 'AskUserNextSteps' node (indicating a valid solution) or stops.
    """
    while True:
        snapshot = app.get_state(config)
        if not snapshot.next:
            break
            
        next_step = snapshot.next[0]
        
        if next_step == "AskUserNextSteps":
            # Valid solution reached, ready for next phase/stop
            break
            
        if next_step == "AskUserVerification":
            # Workflow needs answers (either first attempt or retry after invalid)
            state_values = snapshot.values
            latest_questions = state_values.get("GenerateHypotheses.verification_questions", [])
            
            if not latest_questions:
                 logger.error("No verification questions found in state")
            
            logger.info(f"Generating Loop Answers (count: {len(latest_questions)})")     
            answers = interviewer.answer_verification(latest_questions, context)
            
            try:
                app.invoke(Command(resume=answers), config)
            except GraphInterrupt:
                continue
        else:
             # Unexpected state, maybe log and break? 
             # For now assume if it's not NextSteps, it's something we should try to resume or let follow-through
             # But strictly we only handle Verification here.
             logger.warning(f"Unexpected next step in verification loop: {next_step}")
             break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run system design interview evaluation.")
    parser.add_argument("tasks_file", nargs="?", default="evaluation/task_1.csv", help="Path to the tasks CSV file")
    args = parser.parse_args()
    
    run_evaluation_loop(args.tasks_file)
