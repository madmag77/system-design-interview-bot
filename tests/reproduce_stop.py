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

def test_workflow_stop_logic(workflow_path):
    """Test that the workflow actually stops when 'next_action' is 'stop'."""
    
    # Mock all functions to be predictable
    mock_functions = {
        "generate_hypotheses": lambda current_question, initial, config, **kwargs: {"hypotheses": ["H1"], "verification_questions": ["Q1"]},
        "ask_user_verification": lambda questions, config, **kwargs: {}, # HITL
        "verify_hypotheses": lambda hypotheses, answers, config: {"is_valid": True, "best_hypothesis": "H1", "solution_draft": "Draft", "verification_reason": "Valid"},
        "ask_user_retry": lambda is_valid, config, **kwargs: {}, # HITL
        "generate_solution": lambda hypothesis, draft, config, **kwargs: {"solution": "Sol"},
        "critic_review": lambda solution, config, **kwargs: {"final_solution": "FinalSol"},
        "ask_user_next_steps": lambda solution, config, **kwargs: {}, # HITL
        "determine_next_state": lambda retry_input, next_input, next_action, is_valid, config: {
            "should_stop": (next_action == "stop"),
            "next_question": "NextQ" if next_action == "loop" else ""
        },
        "save_results": lambda history, config: {"report": "Report"}
    }

    app = build_pregel_graph(workflow_path, mock_functions, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test_stop"}}
    
    print("Starting workflow...")
    # 1. Initial run -> Interrupt at AskUserVerification
    app.invoke({"initial_question": "Test"}, config)
    
    # 2. Resume with verification answer -> Interrupt at AskUserNextSteps
    # The workflow goes: Verify -> GenerateSolution -> CriticReview -> AskUserNextSteps
    print("Resuming with verification answers...")
    app.invoke(Command(resume=["Answer"]), config)
    
    # 3. Resume with STOP action
    print("Resuming with STOP action...")
    res = app.invoke(Command(resume={"next_action": "stop", "new_input": ""}), config)
    
    print(f"Result keys: {res.keys()}")
    
    if "SaveResults.report" in res:
        print("PASS: Workflow stopped and produced report.")
    else:
        state = app.get_state(config)
        print(f"FAIL: Workflow did NOT produce report. Current state tasks: {state.tasks}")
        pytest.fail("Workflow did not stop as expected.")

if __name__ == "__main__":
    pytest.main([__file__])
