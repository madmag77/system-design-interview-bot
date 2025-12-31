# System Design Interview Bot

This application simulates a System Design Interview using LLMs and WIRL workflows.

## Features
- Interactive interview simulation.
- Generates engineering hypotheses and verification questions.
- Validates user answers.
- Generates and critiques solutions.
- HITL (Human-in-the-Loop) interaction.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Note: You might need to install `wirl-lang` and `wirl-pregel-runner` from the `reference_wirl` directory if they are not available on PyPI.

2. Run the application:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Testing
Run the workflow tests:
```bash
pytest tests/test_workflow.py
```
