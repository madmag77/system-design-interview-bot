# Agent Documentation

## Running Tests

To run the tests for this project, use the following command from the root directory:

```bash
.venv/bin/python -m pytest tests/test_functions_signatures.py tests/test_workflow.py tests/test_integration.py
```

Arguments explanation:
- `.venv/bin/python`: Uses the python interpreter from the virtual environment.
- `-m pytest`: Runs pytest module.
- `tests/...`: Specifies the test files to run.

## Workflow Details

The system design workflow uses `wirl` and is backed by Python functions in `workflow_definitions/system_design/functions.py`.
- **Hypotheses History**: We track the history of generated hypotheses to avoid repetition and provide context.
- **Interrupts**: The workflow loops and interrupts for user input (`AskUserVerification`, `AskUserNextSteps`).
- **Summarizer**: A custom reducer function that aggregates results into the history.

## Common Issues

- **Signature Matching**: Ensure that the functions in `functions.py` match the signatures expected by the WIRL definitions and the test mocks.
- **Indentation**: Be careful with python indentation when editing files.
- **Report Rendering**: The `save_results` function formats the history into a markdown report.
