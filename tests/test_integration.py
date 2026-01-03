import pytest
import sys
import json
import itertools
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
from workflow_definitions.system_design.functions import HypothesesList, VerificationResult, HypothesisVerification

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
    
    def run_workflow(is_valid_scenarion: bool):
        # Setup Mock LLM responses
        mock_llm_instance = MagicMock()
        mock_get_llm.return_value = mock_llm_instance
        mock_structured_llm = MagicMock()
        mock_llm_instance.with_structured_output.return_value = mock_structured_llm

        # 1. Generate Hypotheses (Structured)
        hypotheses_obj = HypothesesList(
            hypotheses=["H1", "H2"],
            verification_questions=["Q1"]
        )

        # 2. Verify Hypotheses (Structured)
        # Using new HypothesisVerification schema
        h1_verify = HypothesisVerification(
            hypothesis="H1",
            is_valid=is_valid_scenarion,
            reason="Good" if is_valid_scenarion else "Bad",
            is_best=is_valid_scenarion
        )
        verify_obj = VerificationResult(
            hypotheses_feedback=[h1_verify],
            solution_draft="Draft"
        )
        
        # 3. Generate Solution (String - Normal Invoke)
        # Only runs if valid
        solution_resp = "Solution content"
        
        # 4. Critic Review (String - Normal Invoke)
        # Only runs if valid
        critic_resp = "Final Solution"
        
        # Configure side effects
        mock_structured_llm.side_effect = itertools.cycle([hypotheses_obj, verify_obj])
        mock_structured_llm.invoke.side_effect = itertools.cycle([hypotheses_obj, verify_obj])
        
        msg_sol = AIMessage(content=solution_resp)
        msg_cri = AIMessage(content=critic_resp)
        msg_analysis = AIMessage(content="Analysis Result")
        
        # mock_llm_instance is used for generate_solution and critic_review
        mock_llm_instance.side_effect = itertools.cycle([msg_sol, msg_cri])
        mock_llm_instance.invoke.side_effect = itertools.cycle([msg_sol, msg_cri])
        
        # mock_bound_llm is used for verify_hypotheses (agent loop)
        mock_bound_llm = MagicMock()
        mock_bound_llm.invoke.return_value = msg_analysis
        mock_llm_instance.bind_tools.return_value = mock_bound_llm


        # SPIES
        spies = {
            "generate_hypotheses": MagicMock(wraps=generate_hypotheses),
            "ask_user_verification": MagicMock(wraps=ask_user_verification),
            "verify_hypotheses": MagicMock(wraps=verify_hypotheses),
            "ask_user_retry": MagicMock(wraps=ask_user_retry),
            "generate_solution": MagicMock(wraps=generate_solution),
            "critic_review": MagicMock(wraps=critic_review),
            "summarize": MagicMock(wraps=summarize),
            "ask_user_next_steps": MagicMock(wraps=ask_user_next_steps),
            "determine_next_state": MagicMock(wraps=determine_next_state),
            "save_results": MagicMock(wraps=save_results)
        }

        app = build_pregel_graph(workflow_path, spies, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": f"test_integration_{is_valid_scenarion}"}}
        
        # Start
        try:
            app.invoke({"initial_question": "Test Question"}, config)
        except GraphInterrupt:
            # Resume 1: Answers
            pass
            
        try:
             app.invoke(Command(resume={"answers": ["A1"]}), config)
        except GraphInterrupt:
            # Resume 2: Next Steps (only happens if valid)
            pass
            
        if is_valid_scenarion:
             # If valid, we hit AskUserNextSteps interrupt
             app.invoke(Command(resume={"next_action": "stop", "new_input": ""}), config)
        else:
             # If invalid, logic continues without interrupt at NextSteps?
             # Actually, if invalid -> DetermineNextState -> Stop (if we assume next_action is stop? or we assume retry?)
             # DetermineNextState: next_action comes from AskUserNextSteps (which didn't run).
             # So next_action is None?
             # determine_next_state(next_action=None) => stop=False, next_question="Previous hypotheses were invalid..."
             # Workflow loops back to GenerateHypotheses.
             # We can't easily stop the loop unless we inject stop or max_iterations hit.
             # But we can assert what ran so far.
             pass
        
        return spies

    # Case 1: Valid
    print("Testing Valid Scenario...")
    spies_valid = run_workflow(is_valid_scenarion=True)
    assert spies_valid["generate_hypotheses"].called
    assert spies_valid["verify_hypotheses"].called
    assert spies_valid["generate_solution"].called
    assert spies_valid["critic_review"].called
    # This assertion catches the bug:
    assert spies_valid["summarize"].called, "Summarizer should run in valid path"
    
    # Case 2: Invalid
    print("Testing Invalid Scenario...")
    spies_invalid = run_workflow(is_valid_scenarion=False)
    assert spies_invalid["generate_hypotheses"].called
    assert spies_invalid["verify_hypotheses"].called
    assert not spies_invalid["generate_solution"].called
    assert not spies_invalid["critic_review"].called
    assert spies_invalid["summarize"].called, "Summarizer should run in invalid path"
