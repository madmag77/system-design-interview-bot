import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wirl_pregel_runner.pregel_graph_builder import build_pregel_graph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langgraph.errors import GraphInterrupt

@pytest.fixture
def workflow_path():
    return "workflow_definitions/system_design/workflow.wirl"

def test_system_design_workflow_e2e(workflow_path):
    """End-to-end test of the system design workflow with mocked functions."""
    
    # Mock functions
    # Using simple functions as in the example for clarity and control
    
    def mock_generate_hypotheses(current_question, initial, config, **kwargs):
        return {
            "hypotheses": ["H1", "H2"], 
            "verification_questions": ["Q1"]
        }

    def mock_verify_hypotheses(hypotheses, answers, config, **kwargs):
        assert "questions" in kwargs, "Missing 'questions' input in VerifyHypotheses"
        assert "hypotheses_history" in kwargs, "Missing 'hypotheses_history' input in VerifyHypotheses"
        return {
            "is_valid": True, 
            "best_hypothesis": "H1", 
            "solution_draft": "Draft",
            "verification_reason": "Valid"
        }

    def mock_generate_solution(hypothesis, draft, is_valid, config, **kwargs):
        assert "questions" in kwargs, "Missing 'questions' input in GenerateSolution"
        assert "answers" in kwargs, "Missing 'answers' input in GenerateSolution"
        assert "hypotheses_history" in kwargs, "Missing 'hypotheses_history' input in GenerateSolution"
        return {"solution": "Solution"}

    def mock_critic_review(solution, is_valid, config, **kwargs):
        assert "questions" in kwargs, "Missing 'questions' input in CriticReview"
        assert "answers" in kwargs, "Missing 'answers' input in CriticReview"
        assert "hypothesis" in kwargs, "Missing 'hypothesis' input in CriticReview"
        assert "hypotheses_history" in kwargs, "Missing 'hypotheses_history' input in CriticReview"
        return {"final_solution": "Final Solution"}
        
    def mock_summarize(initial, current_question, hypotheses, questions, answers, hypothesis, is_valid, reason, config, **kwargs):
        return {"hypotheses_history": [{"hypothesis": "H1", "is_valid": True}]}
    
    def mock_determine_next_state(verification_reason, next_input, next_action, is_valid, config):
        return {
            "should_stop": True, 
            "next_question": ""
        }

    def mock_save_results(history, config):
        return {"report": "Report"}

    # Mock HITL functions that return empty dicts as they are interrupted before output in this flow?
    # Actually in the original test they return empty dicts.
    
    mock_functions = {
        "generate_hypotheses": mock_generate_hypotheses,
        "ask_user_verification": lambda questions, config, **kwargs: {},
        "verify_hypotheses": mock_verify_hypotheses,
        "ask_user_retry": lambda config, **kwargs: {},
        "generate_solution": mock_generate_solution,
        "critic_review": mock_critic_review,
        "summarize": mock_summarize,
        "ask_user_next_steps": lambda solution, is_valid, config, **kwargs: {},
        "determine_next_state": mock_determine_next_state,
        "save_results": mock_save_results
    }

    app = build_pregel_graph(workflow_path, mock_functions, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test_e2e"}}
    
    # Start Workflow
    print("Starting workflow...")
    try:
        res = app.invoke({"initial_question": "Test"}, config)
        if res and "__interrupt__" in res:
             print("Interrupted at AskUserVerification (returned)")
        else:
             print(f"Workflow finished without interrupt. Result: {res}")
             assert False, "Should have interrupted at AskUserVerification"
    except GraphInterrupt:
        print("Interrupted at AskUserVerification (exception)")
    
    # Verify interrupt state
    state = app.get_state(config)
    assert state.tasks[0].interrupts
    
    # Resume with answers (simulating AskUserVerification input)
    print("Resuming with answers...")
    try:
        res = app.invoke(Command(resume=["Answer1"]), config)
        if res and "__interrupt__" in res:
             print("Interrupted at AskUserNextSteps (returned)")
        else:
             print(f"Workflow finished without interrupt. Result: {res}")
             assert False, "Should have interrupted at AskUserNextSteps"
    except GraphInterrupt:
        print("Interrupted at AskUserNextSteps (exception)")

    # Verify interrupt state
    state = app.get_state(config)
    assert any(task.interrupts for task in state.tasks)
    
    # Resume with stop (simulating AskUserNextSteps input)
    print("Resuming with stop...")
    res = app.invoke(Command(resume={"next_action": "stop", "new_input": ""}), config)
    
    # Final assertions
    print(f"Workflow finished. Result keys: {res.keys()}")
    assert "SaveResults.report" in res
    assert res["SaveResults.report"] == "Report"
