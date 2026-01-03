import pytest
import sys
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import os

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import run_evaluation_loop
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

@patch("evaluation.evaluator.ChatOllama")
@patch("evaluation.evaluator.SimulatedInterviewer")
@patch("evaluation.evaluator.build_pregel_graph")
@patch("evaluation.evaluator.load_tasks")
@patch("builtins.open", new_callable=MagicMock)
def test_evaluation_loop(mock_open, mock_load_tasks, mock_build_graph, mock_interviewer_cls, mock_chat_ollama):
    # Setup Mocks
    mock_load_tasks.return_value = [
        {
            "task_id": "1",
            "initial_prompt": "Prompt",
            "context_phase_1": "C1",
            "context_phase_2": "C2",
            "ideal_outcome": "Outcome"
        }
    ]
    
    # Mock Interviewer instance
    mock_interviewer = MagicMock()
    mock_interviewer_cls.return_value = mock_interviewer
    mock_interviewer.answer_verification.return_value = ["A1"]
    mock_interviewer.generate_challenge.return_value = "Challenge"
    mock_interviewer.score_report.return_value = {"score": 8, "reasoning": "Good"}
    
    # Mock App (Workflow)
    mock_app = MagicMock()
    mock_build_graph.return_value = mock_app
    
    # Configure App behavior (Interrupts and State)
    # We need to simulate the sequence of invokes:
    # 1. Initial invoke -> Interrupt (AskUserVerification)
    # 2. Resume Answers -> Interrupt (AskUserNextSteps)
    # 3. Resume Challenge -> Interrupt (AskUserVerification P2)
    # 4. Resume Answers P2 -> Interrupt (AskUserNextSteps P2)
    # 5. Resume Stop -> Finish
    
    # We use side_effect to raise Interrupts then return None
    # actually invoke does not return anything useful usually in this loop, we assume interrupts
    
    # Sequence of invokes:
    # 1. invoke(dict) -> raise Interrupt
    # 2. invoke(Command) -> raise Interrupt
    # 3. invoke(Command) -> raise Interrupt
    # 4. invoke(Command) -> raise Interrupt
    # 5. invoke(Command) -> returns final state (mock)
    
    mock_app.invoke.side_effect = [
        GraphInterrupt(), # 1
        GraphInterrupt(), # 2
        GraphInterrupt(), # 3
        GraphInterrupt(), # 4
        {"messages": []}  # 5 Success
    ]
    
    # Mock State and its evolution
    # The loop calls get_state() repeatedly.
    # We need to simulate the state "next" field to control the loop:
    # Phase 1:
    #  1. Initial invoke -> Interrupt.
    #  2. get_state -> next=["AskUserVerification"] (Loop starts, needs answers)
    #  3. invoke(answers) -> Interrupt.
    #  4. get_state -> next=["AskUserNextSteps"] (Loop ends, valid)
    
    # Phase 2:
    #  1. invoke(challenge) -> Interrupt.
    #  2. get_state -> next=["AskUserVerification"] (Loop starts, needs answers)
    #  3. invoke(answers) -> Interrupt.
    #  4. get_state -> next=["AskUserNextSteps"] (Loop ends, valid)
    
    # Final:
    #  1. invoke(stop) -> Success
    #  2. get_state -> final report
    
    # So get_state is called:
    # 1. Inside loop P1 (start)
    # 2. Inside loop P1 (check after invoke)
    # 3. Inside loop P2 (start)
    # 4. Inside loop P2 (check after invoke)
    # 5. After finish to get report
    
    state_p1_start = MagicMock()
    state_p1_start.next = ["AskUserVerification"]
    state_p1_start.values = {"GenerateHypotheses": {"verification_questions": ["Q1"]}}
    
    state_p1_end = MagicMock()
    state_p1_end.next = ["AskUserNextSteps"]
    
    state_p2_start = MagicMock()
    state_p2_start.next = ["AskUserVerification"]
    state_p2_start.values = {"GenerateHypotheses": {"verification_questions": ["Q2"]}}
    
    state_p2_end = MagicMock()
    state_p2_end.next = ["AskUserNextSteps"]
    
    state_final = MagicMock()
    state_final.next = []
    state_final.values = {
        "SaveResults": {"report": "Final Report Used For Scoring"},
        "SaveResults.report": "Final Report Used For Scoring"
    }
    
    mock_app.get_state.side_effect = [
        state_p1_start, # Phase 1 Loop Start
        state_p1_end,   # Phase 1 Loop End (after answer invoke)
        state_p2_start, # Phase 2 Loop Start
        state_p2_end,   # Phase 2 Loop End
        state_final     # Final Report
    ]

    # Run loop
    try:
        # We need to cleanup any created results file after
        run_evaluation_loop()
    except Exception as e:
        pytest.fail(f"Evaluation loop failed: {e}")
        
    # Assertions
    assert mock_interviewer.answer_verification.call_count == 2
    assert mock_interviewer.generate_challenge.call_count == 1
    assert mock_interviewer.score_report.call_count == 1
    
    # Verify report was passed to scorer
    mock_interviewer.score_report.assert_called_with("Final Report Used For Scoring", "Outcome")
    
    # Verify report was passed to scorer
    mock_interviewer.score_report.assert_called_with("Final Report Used For Scoring", "Outcome")
    
    # Verify file write occurred (optional, but good to check)
    mock_open.assert_called()
