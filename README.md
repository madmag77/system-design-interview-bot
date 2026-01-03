# System Design Interview Bot

A dedicated agentic workflow for practicing System Design Interviews using LLMs. This project simulates an interview environment where you act as an interviewer and LLM acts as the candidate (in Streamlit).

## 🚀 Features
- **Interactive Interview Simulation** (Streamlit App)
- **Role-playing Agents**: Interviewer, Candidate, and Critic LLMs.
- **Workflow Steps**:
  - Hypothesis Generation
  - Verification & Constraints Checking
  - Solution Design
  - Review & Critique
- **Automated Evaluation Loop** for testing bot performance against diverse tasks.

## 📖 UI Walkthrough & Usage

This workflow simulates a system design interview where the **Bot acts as the Candidate** and **You act as the Interviewer**.

1.  **Scenario Setup**
    
    You (Interviewer) start by providing a system design problem (e.g., "Design a URL Shortener").
    
    ![Initial Screen](images/initial_screen.png)

2.  **Engineering Challenges (Hypotheses)**
    
    The Candidate generates 2-3 potential engineering challenges (hypotheses) for the problem. It then asks you clarifying questions to validate these hypotheses and decide which one to tackle.
    
    ![Verification Questions](images/verifying_questions.png)

3.  **Verification & Refinement**
    
    You answer the questions to guide the Candidate. If the hypotheses are not what you intended, your answers help the Candidate refine its understanding and choose the correct direction.
    
    ![Invalid Hypotheses](images/not_valid_hypotheses.png)

4.  **Solution & Deep Dive**
    
    The Candidate provides a solution for the chosen hypothesis. It then asks if you want to:
    *   **Deep Dive**: Ask follow-up questions (e.g., "What if QPS increases 100x?"). The context (previous hypotheses) is retained, extending the current solution.
    *   **Finish**: End the interview and generate the final report.

    ![Final Report](images/report1.png)

## 🛠️ Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have `ollama` installed and running with the required models (e.g., `gpt-oss:20b` or `gemma3:27b`).*

2. **Run the Interactive App**:
   Start the Streamlit interface to practice an interview yourself:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## 📊 Evaluation

The project includes an automatic evaluation script to benchmark the bot's performance. It uses an **LLM as a Judge** to evaluate the quality of the generated solutions.

### How it works
The evaluator reads tasks, context, and expected main points from a CSV file (e.g., `evaluation/tasks.csv`). It runs the bot against these scenarios and uses an LLM to grade the generated solutions on a scale of **0 to 5**.

**Current Baseline**: The architecture currently achieves an average score of **4.3/5**.

### Running Evaluation

1. **Run Evaluator**:
   By default, this runs tasks defined in `evaluation/tasks.csv` (10 diverse scenarios):
   ```bash
   python evaluation/evaluator.py [path/to/tasks.csv]
   ```
   
2. **View Reports**:
   Results, including scores and generated design reports, are saved to:
   `eval_reports/results_YYYYMMDD_HHMMSS.csv`

## 🧪 Testing

Run the integration tests to verify the workflow logic:
```bash
pytest tests/test_evaluation.py
```
## 🏗️ Internal Design

The bot is architected using a modular workflow definition and a Streamlit-based orchestrator.

### 1. WIRL Workflow
The core logic is defined in `workflow_definitions/system_design/workflow.wirl`. This file specifies the `SystemDesignInterview` workflow, including:
*   **Cycles**: The `InterviewLoop` handles the iterative interaction (Hypothesis -> Verification -> Solution).
*   **Typed Inputs/Outputs**: Data passed between nodes is typed for readability and convenience, ensuring clear interfaces.

### 2. Logic & Agents
The implementation of workflow nodes resides in `workflow_definitions/system_design/functions.py`.
*   **Pure Functions**: Nodes are implemented as pure functions, making them easy to debug and test in isolation.
*   **Agentic Verification**: The `verify_hypotheses` function utilizes a specialized LangGraph agent (`workflow_definitions/system_design/agent.py`).
*   **Tools**: This agent is equipped with a **Python tool** (`calculate_metrics`) allowing it to execute real Python code for calculations (e.g., QPS, storage estimation) during verification.

> [!WARNING]
> Enabling arbitrary Python code execution by the agent can introduce security risks. Ensure you are running this in a sandboxed or trusted environment.

### 3. Orchestration
The main application `app/streamlit_app.py` acts as the orchestrator:
*   **Graph Building**: It compiles the WIRL definition into an executable Pregel graph using `build_pregel_graph`.
*   **State Loop**: It manages the user session, handles Human-in-the-Loop (HITL) interrupts for user input, and resumes execution.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
