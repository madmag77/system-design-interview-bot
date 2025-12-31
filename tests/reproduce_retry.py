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

def test_workflow_retry_interrupt(workflow_path):
    """Test that the workflow interrupts at AskUserRetry when hypotheses are invalid."""
    
    # Mock functions
    # VerifyHypotheses returns is_valid=False
    mock_functions = {
        "generate_hypotheses": lambda current_question, initial, config, **kwargs: {"hypotheses": ["H1"], "verification_questions": ["Q1"]},
        "ask_user_verification": lambda questions, config, **kwargs: {}, # HITL
        "verify_hypotheses": lambda hypotheses, answers, config: {
            "is_valid": False, 
            "best_hypothesis": "", 
            "solution_draft": "",
            "verification_reason": "Invalid"
        },
        "ask_user_retry": lambda is_valid, config, **kwargs: {}, # HITL
        "generate_solution": lambda hypothesis, draft, config, **kwargs: {"solution": "Sol"}, # Should NOT run
        "critic_review": lambda solution, config, **kwargs: {"final_solution": "FinalSol"}, # Should NOT run
        "ask_user_next_steps": lambda solution, config, **kwargs: {}, # Should NOT run
        "determine_next_state": lambda retry_input, next_input, next_action, is_valid, config: {
            "should_stop": True, # Should stop if retry input missing
            "next_question": ""
        },
        "save_results": lambda history, config: {"report": "Report"}
    }

    app = build_pregel_graph(workflow_path, mock_functions, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test_retry"}}
    
    print("Starting workflow...")
    # 1. Initial run -> Interrupt at AskUserVerification
    app.invoke({"initial_question": "Test"}, config)
    
    # 2. Resume with verification answer -> Verify returns False -> Interrupt at AskUserRetry
    print("Resuming with verification answers (expecting invalid)...")
    
    try:
        res = app.invoke(Command(resume=["Answer"]), config)
        
        # If we are here, it means NO interrupt was raised (which is the bug)
        # OR it returned the interrupt object (if not raising exception)
        
        if res and "__interrupt__" in res:
             print("PASS: Interrupted at AskUserRetry (returned)")
             # Check interrupt details to be sure it is AskUserRetry
             assert res["__interrupt__"][0].value
        else:
             print(f"FAIL: Workflow finished without interrupt. Result: {res}")
             state = app.get_state(config)
             print(f"Current state tasks: {state.tasks}")
             pytest.fail("Should have interrupted at AskUserRetry")
             
    except GraphInterrupt:
        print("PASS: Interrupted at AskUserRetry (exception)")

if __name__ == "__main__":
    pytest.main([__file__])
