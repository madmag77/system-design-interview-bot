import logging
import streamlit as st
import sys
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add reference wirl paths if not installed
sys.path.insert(0, str(Path(__file__).parent.parent / "reference_wirl/packages/wirl-lang"))
sys.path.insert(0, str(Path(__file__).parent.parent / "reference_wirl/packages/wirl-pregel-runner"))

from wirl_pregel_runner.pregel_graph_builder import build_pregel_graph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langgraph.errors import GraphInterrupt
from workflow_definitions.system_design.functions import (
    generate_hypotheses,
    ask_user_verification,
    verify_hypotheses,
    ask_user_retry,
    generate_solution,
    critic_review,
    ask_user_next_steps,
    determine_next_state,
    save_results
)

load_dotenv()

st.set_page_config(page_title="System Design Interview Bot", layout="wide")
st.title("System Design Interview Bot")

# Sidebar
with st.sidebar:
    st.header("Instructions")
    st.write("1. Enter a system design question.")
    st.write("2. Answer verification questions.")
    st.write("3. Review the solution.")

# Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "app" not in st.session_state:
    # Build app once
    fn_map = {
        "generate_hypotheses": generate_hypotheses,
        "ask_user_verification": ask_user_verification,
        "verify_hypotheses": verify_hypotheses,
        "ask_user_retry": ask_user_retry,
        "generate_solution": generate_solution,
        "critic_review": critic_review,
        "ask_user_next_steps": ask_user_next_steps,
        "determine_next_state": determine_next_state,
        "save_results": save_results
    }
    workflow_path = "workflow_definitions/system_design/workflow.wirl"

    st.session_state.app = build_pregel_graph(workflow_path, fn_map, checkpointer=MemorySaver())

if "messages" not in st.session_state:
    st.session_state.messages = [] # For chat history if needed

if "workflow_status" not in st.session_state:
    st.session_state.workflow_status = "idle" # idle, running, interrupted, finished

if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None

# Main UI
if st.session_state.workflow_status == "idle":
    with st.form("initial_form"):
        question = st.text_area("Enter System Design Question", "Design a URL Shortener like Bit.ly")
        submitted = st.form_submit_button("Start Designing")
        if submitted:
            st.session_state.workflow_status = "running"
            st.session_state.initial_question = question
            st.rerun()

elif st.session_state.workflow_status == "running":
    with st.spinner("Running workflow..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        try:
            # Initial run
            logger.info("Calling app.invoke...")
            res = st.session_state.app.invoke(
                {"initial_question": st.session_state.initial_question}, 
                config
            )
            logger.info(f"app.invoke returned: {res}")
            
            if res and "__interrupt__" in res:
                logger.info("Interrupt detected in return value.")
                st.session_state.workflow_status = "interrupted"
                st.rerun()
            else:
                st.session_state.workflow_status = "finished"
                st.rerun()
        except GraphInterrupt as e:
            logger.info(f"Caught GraphInterrupt in streamlit_app: {e}")
            st.session_state.workflow_status = "interrupted"
            st.rerun()
        except Exception as e:
            logger.info(f"Caught Exception in streamlit_app: {type(e)} {e}")
            import traceback
            logger.info(traceback.format_exc())
            st.error(f"Error: {e}")
            st.session_state.workflow_status = "idle"

elif st.session_state.workflow_status == "interrupted":
    # Fetch interrupt details
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state = st.session_state.app.get_state(config)
    if not state.tasks:
        st.error("Interrupted but no tasks found.")
        st.stop()
    
    interrupt_value = state.tasks[0].interrupts[0].value
    # Value is {"request": json_string}
    request_data = json.loads(interrupt_value.get("request", "{}"))
    
    st.subheader("Input Required")
    
    # Determine type of interrupt
    if "questions" in request_data:
        # Verification Questions
        st.subheader("Verification Questions")
        
        # Display Hypotheses if available
        if "hypotheses" in request_data:
            st.info("I have generated the following hypotheses based on the initial question:")
            for h in request_data["hypotheses"]:
                st.markdown(f"- {h}")
        
        st.write("Please answer the following verification questions:")
        questions = request_data["questions"]
        answers = []
        with st.form("verification_form"):
            for i, q in enumerate(questions):
                ans = st.text_input(f"Q{i+1}: {q}")
                answers.append(ans)
            submit = st.form_submit_button("Submit Answers")
            if submit:
                # Resume
                st.session_state.resume_value = answers # List[str]
                st.session_state.workflow_status = "resuming"
                st.rerun()

    elif "solution" in request_data:
        # Next Steps
        st.success("Solution Generated!")
        st.markdown(request_data["solution"])
        
        with st.form("next_steps_form"):
            action = st.radio("What would you like to do?", ["Continue (Loop)", "Stop & Save"])
            new_input = st.text_area("If continuing, enter new input (optional):")
            submit = st.form_submit_button("Proceed")
            if submit:
                next_action = "loop" if "Continue" in action else "stop"
                st.session_state.resume_value = {"next_action": next_action, "new_input": new_input}
                st.session_state.workflow_status = "resuming"
                st.rerun()

    elif "is_valid" in request_data:
        # Retry (Hypotheses invalid)
        st.warning("The generated hypotheses were not valid based on your answers.")
        
        # Display reason if available
        if "reason" in request_data and request_data["reason"]:
            st.info(f"Reason: {request_data['reason']}")
            
        with st.form("retry_form"):
            new_input = st.text_area("Please provide more context or a refined question:", st.session_state.initial_question)
            submit = st.form_submit_button("Retry")
            if submit:
                st.session_state.resume_value = new_input
                st.session_state.workflow_status = "resuming"
                st.rerun()
    else:
        st.error(f"Unknown interrupt: {request_data}")

elif st.session_state.workflow_status == "resuming":
    with st.spinner("Resuming workflow..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        try:
            resume_val = st.session_state.resume_value
            logger.info("Calling app.invoke (resume)...")
            res = st.session_state.app.invoke(Command(resume=resume_val), config)
            logger.info(f"app.invoke (resume) returned: {res}")
            
            if res and "__interrupt__" in res:
                logger.info("Interrupt detected in resume return value.")
                st.session_state.workflow_status = "interrupted"
                st.rerun()
            else:
                st.session_state.workflow_status = "finished"
                st.rerun()
        except GraphInterrupt:
            logger.info("Caught GraphInterrupt in resume")
            st.session_state.workflow_status = "interrupted"
            st.rerun()
        except Exception as e:
            logger.info(f"Caught Exception in resume: {e}")
            st.error(f"Error resuming: {e}")
            st.session_state.workflow_status = "idle"

elif st.session_state.workflow_status == "finished":
    st.success("Workflow Completed!")
    
    # Get final output
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state = st.session_state.app.get_state(config)
    
    values = state.values
    # Check for final report in values
    report_key = "SaveResults.report"
    if report_key in values:
        st.markdown(values[report_key])
        st.download_button("Download Report", values[report_key], "report.md")
    elif "final_report" in values:
        st.markdown(values["final_report"])
        st.download_button("Download Report", values["final_report"], "report.md")
    
    if st.button("Start New Interview"):
        st.session_state.clear()
        st.rerun()
