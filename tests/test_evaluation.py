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
def test_evaluation_loop(mock_load_tasks, mock_build_graph, mock_interviewer_cls, mock_chat_ollama):
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
    
    # Mock State
    mock_snapshot = MagicMock()
    mock_snapshot.values = {
        "GenerateHypotheses": {"verification_questions": ["Q1"]},
        "SaveResults": {"report": "Final Report Used For Scoring"}
    }
    mock_app.get_state.return_value = mock_snapshot

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
    
    # Cleanup csv
    # The script writes to evaluation/results_...csv
    # We can just ignore it or find and delete
    # simple cleanup:
    eval_dir = Path("evaluation")
    for f in eval_dir.glob("results_*.csv"):
        try:
           f.unlink()
        except:
            pass
