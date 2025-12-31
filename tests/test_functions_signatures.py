import pytest
from unittest.mock import MagicMock, patch
from workflow_definitions.system_design.functions import (
    generate_hypotheses,
    verify_hypotheses,
    generate_solution,
    critic_review,
    ask_user_retry,
    ask_user_next_steps, 
    ask_user_verification
)

@pytest.fixture
def mock_config():
    return {"configurable": {"thread_id": "test_thread"}, "model": "test_model"}

@patch("workflow_definitions.system_design.functions.get_llm")
def test_generate_solution_signature(mock_get_llm, mock_config):
    """Test that generate_solution accepts extra kwargs without error."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value.content = "Mock Solution"
    mock_get_llm.return_value.__or__.return_value = mock_chain
    
    # helper for mocking LLM pipeline
    
    # Call with expected AND unexpected args
    try:
        generate_solution(
            hypothesis="H1", 
            draft="Draft", 
            config=mock_config,
            is_valid=True, # Unexpected arg that caused crash
            random_arg="foo"
        )
    except TypeError as e:
        pytest.fail(f"generate_solution raised TypeError with extra args: {e}")

@patch("workflow_definitions.system_design.functions.get_llm")
def test_critic_review_signature(mock_get_llm, mock_config):
    """Test that critic_review accepts extra kwargs."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value.content = "Critique"
    mock_get_llm.return_value.__or__.return_value = mock_chain
    
    try:
        critic_review(
            solution="Sol", 
            config=mock_config, 
            is_valid=True,
            extra="arg"
        )
    except TypeError as e:
        pytest.fail(f"critic_review raised TypeError: {e}")

def test_ask_user_retry_signature(mock_config):
    """Test ask_user_retry signature."""
    try:
        ask_user_retry(
            is_valid=False, 
            reason="Invalid",
            config=mock_config,
            extra="stuff"
        )
    except TypeError as e:
         pytest.fail(f"ask_user_retry raised TypeError: {e}")

def test_ask_user_next_steps_signature(mock_config):
    """Test ask_user_next_steps signature."""
    try:
        ask_user_next_steps(
            solution="Finished", 
            config=mock_config,
            is_valid=True,
            random="val"
        )
    except TypeError as e:
         pytest.fail(f"ask_user_next_steps raised TypeError: {e}")
