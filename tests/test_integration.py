import pytest
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wirl_pregel_runner.pregel_graph_builder import build_pregel_graph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langgraph.errors import GraphInterrupt
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from workflow_definitions.system_design.functions import HypothesesList, VerificationResult

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

@pytest.fixture
def workflow_path():
    return "workflow_definitions/system_design/workflow.wirl"

@patch("workflow_definitions.system_design.functions.get_llm")
def test_system_design_integration(mock_get_llm, workflow_path):
    """
    Integration test using REAL functions (mocks only LLM).
    This ensures that WIRL passes arguments that match the Python signatures.
    """
    
    # Setup Mock LLM responses
    mock_llm_instance = MagicMock()
    # verify_hypotheses calls llm.invoke? No, it uses chain | llm.
    # If llm is a mock, chain treats it as runnable if it has invoke? 
    # Or as callable? 
    # Safest is to make it both callable and have invoke returning the same.
    
    mock_get_llm.return_value = mock_llm_instance
    
    # We need to simulate different responses based on the prompt/context, 
    # or just return a generic valid JSON that works for all steps.
    
    # 1. Generate Hypotheses (Structured)
    hypotheses_obj = HypothesesList(
        hypotheses=["H1", "H2"],
        verification_questions=["Q1"]
    )
    
    # 2. Verify Hypotheses (Structured)
    verify_obj = VerificationResult(
        is_valid=True,
        best_hypothesis="H1",
        solution_draft="Draft",
        reason="Valid"
    )
    
    # 3. Generate Solution (String - Normal Invoke)
    solution_resp = "Solution content"
    
    # 4. Critic Review (String - Normal Invoke)
    critic_resp = "Final Solution"
    
    # Config: 
    # generate_hypotheses: get_llm -> structured_llm -> invoke -> HypothesesList
    # verify_hypotheses: get_llm -> structured_llm -> invoke -> VerificationResult
    # generate_solution: get_llm -> invoke -> AIMessage
    # critic_review: get_llm -> invoke -> AIMessage
    
    # We need to distinguish calls. 
    # mock_get_llm returns mock_llm_instance
    # mock_llm_instance.with_structured_output returns ??? (maybe same mock or new one)
    
    mock_structured_llm = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    
    # Set side effects
    
    # structured_llm.invoke calls:
    # When using pipe syntax, the runnable is often invoked directly
    mock_structured_llm.side_effect = [hypotheses_obj, verify_obj]
    mock_structured_llm.invoke.side_effect = [hypotheses_obj, verify_obj]
    
    # normal llm.invoke calls (for solution and critic)
    msg_sol = AIMessage(content=solution_resp)
    msg_cri = AIMessage(content=critic_resp)
    mock_llm_instance.side_effect = [msg_sol, msg_cri]
    mock_llm_instance.invoke.side_effect = [msg_sol, msg_cri]

    fn_map = {
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
    
    app = build_pregel_graph(workflow_path, fn_map, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test_integration"}}
    
    print("Starting integration workflow...")
    
    # 1. Start
    try:
        res = app.invoke({"initial_question": "Test Question"}, config)
        # Should interrupt at AskUserVerification
        assert res and "__interrupt__" in res
    except GraphInterrupt:
        pass
        
    # Check state
    state = app.get_state(config)
    assert any(task.interrupts for task in state.tasks)
    
    # 2. Provide Answers
    print("Resuming with verification answers...")
    try:
        res = app.invoke(Command(resume=["Answer1"]), config)
        # Should interrupt at AskUserNextSteps
        assert res and "__interrupt__" in res
    except GraphInterrupt:
        pass
        
    state = app.get_state(config)
    assert any(task.interrupts for task in state.tasks)
    
    # 3. Stop
    print("Resuming with stop...")
    # WIRL inputs for AskUserNextSteps output: next_action, new_input
    # DetermineNextState inputs: next_input, next_action, is_valid
    res = app.invoke(Command(resume={"next_action": "stop", "new_input": ""}), config)
    
    assert "SaveResults.report" in res
    print("Integration test finished successfully.")
