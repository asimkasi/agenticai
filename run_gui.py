import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import sys
import os
import re
import datetime

# --- Path Setup ---
# Assumes all necessary files (ai_app_builder_workflow.py, workflow_engine.py, model_router.py, workflow_config.json)
# are in the same directory as run_gui.py OR in a parent directory included in sys.path
try:
    # Adjust path if run_gui.py is in a subdirectory like 'gui/'
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ai_app_builder_workflow import GrandOrchestrator
    # workflow_engine and model_router are imported *by* GrandOrchestrator now
except ImportError as e:
    st.error(f"Could not import GrandOrchestrator. Ensure workflow files are accessible. Error: {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Agentic AI Builder")
st.title("🤖 Agentic AI Application Builder")

# --- Logging and Session State Initialization ---

if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

def gui_log(message: str):
    """A custom logger that appends messages to the session_state list for display in the GUI."""
    # Add a timestamp to GUI logs for clarity
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.log_messages.append(f"[{timestamp}] {message.strip()}")

# 2. Initialize the Orchestrator ONCE and store it in the session state
if 'orchestrator' not in st.session_state:
    # Determine if we should use real LLMs based on env var or flag
    USE_REAL_LLMS = os.getenv("USE_REAL_LLMS", "false").lower() == "true"
    if not USE_REAL_LLMS:
         gui_log("NOTE: Running in mock LLM mode. No real LLM calls will be made.")
         
    try:
        # Instantiate the main orchestrator, passing the use_llm flag
        orchestrator = GrandOrchestrator(use_llm_for_agents=USE_REAL_LLMS)
        
        # CRITICAL: Patch the _log method of the orchestrator and all agents
        # This redirects all their print outputs to our GUI logger
        orchestrator._log = gui_log
        for agent in orchestrator.agents.values():
            agent._log = gui_log
            
        st.session_state.orchestrator = orchestrator
        st.session_state.app_started = False
        st.toast("New orchestrator initialized!")
    except FileNotFoundError:
         st.error(f"FATAL ERROR: Workflow configuration file '{GrandOrchestrator.WORKFLOW_CONFIG_PATH}' not found. Please ensure it exists.")
         st.stop() # Stop the Streamlit app

# --- Helper Functions for UI Rendering ---

def handle_user_response(response: str, context_id: str = None):
    """Callback function to process human input and run the next simulation cycle."""
    orchestrator = st.session_state.orchestrator
    
    # IMPORTANT: Clear the current human prompt from the inbox *before* processing the response
    # This prevents the UI from immediately redisplaying the same prompt.
    # Find the prompt by context_id if provided, otherwise just pop the first one.
    if context_id:
        for i, prompt in enumerate(orchestrator.human_inbox):
            if prompt['context_id'] == context_id:
                orchestrator.human_inbox.pop(i)
                break # Found and removed
    elif orchestrator.human_inbox:
        orchestrator.human_inbox.popleft() # Just remove the oldest if no context specified


    # Pass the response and its original context_id to the orchestrator
    # get_human_input queues the response for processing in the *next* cycle
    orchestrator.get_human_input(response, context_id=context_id)
    
    # Running a cycle *immediately* after receiving input ensures the input is processed
    # and the workflow engine reacts to it.
    orchestrator.run_simulation_cycle()
    st.rerun() # Rerun the Streamlit app to update the UI based on the new state and logs

def display_human_prompt(orchestrator):
    """Checks for and displays prompts directed at the human user from the inbox."""
    # Only display the *latest* prompt intended for response (QUESTION, CRITICAL_ISSUE, INSTRUCTION)
    # Check the pending_human_approval_context to find the one being waited on.
    pending_context = orchestrator.project_state.get('pending_human_approval_context')

    prompt_to_display = None
    if pending_context:
        # Find the message in the inbox matching the pending context
        # The inbox might contain other message types (INFO, WARNING, SUCCESS)
        for msg in list(orchestrator.human_inbox): # Iterate over a copy in case we remove
            if msg.get('context_id') == pending_context:
                prompt_to_display = msg
                # Optional: Move this prompt to the front for easier access? No, deque handles order.
                break

    if prompt_to_display:
        # Display the prompt using st.chat_message or similar
        with st.chat_message("human", avatar="🧑‍💻"):
            st.write(prompt_to_display['content'])
            
            # Use a unique key for the input form/buttons based on the context_id
            input_key = f"human_input_{prompt_to_display['context_id']}"

            # Display buttons if options are provided
            if prompt_to_display['options']:
                cols = st.columns(len(prompt_to_display['options']))
                for i, option in enumerate(prompt_to_display['options']):
                    # Pass the context_id to the callback
                    cols[i].button(option, on_click=handle_user_response, args=[option, prompt_to_display['context_id']], key=f"{input_key}_btn_{option.replace(' ', '_')}", use_container_width=True)
            else:
                # Display a text input if no options are given
                # Ensure the form is unique per prompt using the context_id
                with st.form(key=f"text_input_form_{input_key}"):
                    # Pass the context_id to the callback when submitting
                    response_text = st.text_input("Your response:", key=f"human_text_input_{input_key}")
                    submit_button = st.form_submit_button("Submit", use_container_width=True)
                    if submit_button and response_text: # Ensure text is entered
                         handle_user_response(response_text, prompt_to_display['context_id'])
                         # st.rerun() is handled within handle_user_response

    # Display other non-blocking messages (INFO, WARNING, SUCCESS) as regular chat messages
    # We should process these out of the inbox once displayed
    non_prompt_messages = [msg for msg in orchestrator.human_inbox if msg.get('type') not in ['QUESTION', 'CRITICAL_ISSUE', 'INSTRUCTION']]
    for msg in non_prompt_messages:
         avatar = "🤖" if msg.get('sender') != "human" else "🧑‍💻" # Assuming messages in human_inbox are usually from agents/orch
         with st.chat_message(msg.get('type', 'info').lower(), avatar=avatar):
              st.write(f"**{msg.get('type')}**: {msg['content']}")
              # Remove the message from the inbox after displaying
              orchestrator.human_inbox.remove(msg)


def build_and_display_graph(orchestrator):
    """Constructs and renders the agent workflow graph using streamlit-agraph."""
    nodes = []
    edges = []
    active_nodes = set() # Track nodes involved in the most recent actions

    # Determine the currently active agent from the last executed action or log
    last_action_source = None # Could be an agent or Orchestrator
    last_action_target = None # Could be an agent or human/state/etc.
    last_action_label = None

    # Get the last few actions executed by the orchestrator for highlighting
    # This requires storing executed actions temporarily in the Orchestrator state
    # For now, rely on log parsing, which is less robust but matches current state

    # Simple log parsing for active agent and last message/delegation
    if st.session_state.log_messages:
        last_log_entry = st.session_state.log_messages[-1]
        
        # Find the agent speaking (active node)
        agent_match = re.search(r"\[(.*?) \((.*?)\)\]", last_log_entry) # Extracts "Agent Name" from "[Agent Name (Role)]"
        if agent_match:
            active_nodes.add(agent_match.group(1).strip())
        elif "[Orchestrator]" in last_log_entry:
            active_nodes.add("Orchestrator")

        # Find the last message/delegation (edge)
        if "Delegated" in last_log_entry and "task to" in last_log_entry:
            try:
                delegator = re.search(r"\[(.*?)\]", last_log_entry).group(1)
                delegatee = last_log_entry.split("task to ")[1].split(" ")[0].strip()
                last_action_source = delegator
                last_action_target = delegatee
                last_action_label = "delegates task"
                active_nodes.add(delegator)
                active_nodes.add(delegatee)
            except (AttributeError, IndexError):
                pass # Ignore parsing errors
        elif "Routed message to" in last_log_entry:
             try:
                sender_match = re.search(r"from '(.*?)'", last_log_entry)
                if sender_match:
                     sender = sender_match.group(1).strip()
                     recipient = last_log_entry.split("Routed message to '")[1].split("'")[0]
                     last_action_source = sender
                     last_action_target = recipient
                     last_action_label = "routed msg"
                     active_nodes.add(sender)
                     active_nodes.add(recipient)
             except (AttributeError, IndexError):
                  pass # Ignore parsing errors
        elif "Processing message for Orchestrator from" in last_log_entry:
             try:
                sender = last_log_entry.split("from '")[1].split("'")[0]
                last_action_source = sender
                last_action_target = "Orchestrator"
                last_action_label = "sends result/status"
                active_nodes.add(sender)
                active_nodes.add("Orchestrator")
             except (AttributeError, IndexError):
                  pass # Ignore parsing errors


    # Add nodes for the Orchestrator and all agents
    orchestrator_color = "#FF4B4B" if "Orchestrator" in active_nodes else "#D3D3D3"
    orchestrator_size = 25 if "Orchestrator" in active_nodes else 20
    nodes.append(Node(id="Orchestrator", label="Orchestrator", size=orchestrator_size, shape="star", color=orchestrator_color))

    # Add a Human node
    human_color = "#FFA500" # Orange for human
    human_size = 20 if orchestrator.project_state.get('pending_human_approval_context') else 15
    nodes.append(Node(id="Human", label="User", size=human_size, shape="dot", color=human_color))


    for name in orchestrator.agents.keys():
        node_color = "#3498DB" if name in active_nodes else "#D3D3D3"
        node_size = 20 if name in active_nodes else 15
        nodes.append(Node(id=name, label=name, size=node_size, color=node_color))

    # Add edges based on common interactions
    # Orchestrator <-> Human
    edges.append(Edge(source="Orchestrator", target="Human", label="Sends Prompt", color="#FFA500", type="arrow", width=1))
    edges.append(Edge(source="Human", target="Orchestrator", label="Provides Input", color="#FFA500", type="arrow", width=1))

    # Orchestrator -> Agents (Delegation)
    for agent_name in orchestrator.agents.keys():
        edges.append(Edge(source="Orchestrator", target=agent_name, label="Delegates Task", color="#FF4B4B", type="arrow", width=1))

    # Agents -> Orchestrator (Results/Status)
    for agent_name in orchestrator.agents.keys():
        edges.append(Edge(source=agent_name, target="Orchestrator", label="Sends Result/Status", color="#2ECC71", type="arrow", width=1))

    # Highlight the last action edge
    if last_action_source and last_action_target:
         # Find the edge to highlight
         for edge in edges:
              if edge.source == last_action_source and edge.target == last_action_target and edge.label == last_action_label:
                   edge.color = "#FFFF00" # Yellow highlight
                   edge.width = 3
                   edge.font = {"size": 14, "color": "#FFFF00"}
                   break # Highlight only the first matching edge


    config = Config(
        width=700,
        height=500,
        directed=True,
        hierarchical=False,  # Enable physics for better layout
        physics={
            "enabled": True,
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 95,
                "springConstant": 0.04,
                "damping": 0.09,
                "avoidOverlap": 0.5,
            },
        },
                    edges={"color": {"inherit": "from"}, "smooth": True},
                    nodes={"font": {"size": 12}},
                    #cluster={"enabled": True, "titleTemplate": "Cluster of {clusterTitle}"} # Optional: Cluster nodes if desired
    )
    
    # Render the graph representing the current workflow state
    # streamlit-agraph does not expose a `key` argument, so we simply
    # call the component each time the layout updates.
    agraph(nodes=nodes, edges=edges, config=config)

# --- Main Application Layout ---

# Get a fresh reference to the orchestrator from session state
orchestrator = st.session_state.orchestrator
# Initialize step counter if not exists (used for graph key)
if 'simulation_steps' not in st.session_state:
    st.session_state.simulation_steps = 0


# --- Sidebar ---
with st.sidebar:
    st.header("Control Panel")
    if not st.session_state.app_started:
        idea = st.text_area("What app idea do you have today?", "An app to catalog my vinyl record collection, with notes on each record.", height=150)
        if st.button("Start Building", type="primary", use_container_width=True):
            st.session_state.app_started = True
            # Starting the app development now triggers the workflow engine via the Orchestrator
            orchestrator.start_app_development(idea)
            # Run an initial cycle to process the 'start' event and delegate the first task
            orchestrator.run_simulation_cycle() 
            st.session_state.simulation_steps += 1
            st.rerun()
    else:
        st.success(f"Project: '{orchestrator.project_state.get('app_idea', 'N/A')}'")
        
        # Manual Step Button (visible unless human input is explicitly required)
        if not orchestrator.project_state.get('pending_human_approval_context'):
            if st.button("Run Next Simulation Cycle", use_container_width=True):
                orchestrator.run_simulation_cycle()
                st.session_state.simulation_steps += 1
                st.rerun()
        else:
            st.info("Waiting for human input before running next cycle.")


        st.button("Reset Simulation", on_click=lambda: st.session_state.clear(), use_container_width=True) # Use lambda for simple action

    st.divider()
    st.header("Project Dashboard")
    st.metric("Current Phase", orchestrator.project_state['current_phase'])
    
    status_color = "green"
    if "Failed" in orchestrator.project_state['status'] or "Escalated" in orchestrator.project_state['status'] or "Awaiting" in orchestrator.project_state['status']:
        status_color = "red"
    elif "Approved" in orchestrator.project_state['status'] or "Coding" in orchestrator.project_state['status'] or "Refining" in orchestrator.project_state['status']:
        status_color = "orange"
        
    st.markdown(f"**Status:** :{status_color}[{orchestrator.project_state['status']}]")
    st.metric("Simulation Step", st.session_state.simulation_steps)


    with st.expander("Project State (Advanced)"):
        # Display relevant project state details here for debugging/insight
        display_state = orchestrator.project_state.copy()
        # Avoid showing the entire task contexts unless necessary
        if 'current_task_contexts' in display_state:
            display_state['current_task_contexts_summary'] = {
                 ctx_id[:8]: info['task_name'] for ctx_id, info in display_state['current_task_contexts'].items()
            }
            del display_state['current_task_contexts']
        st.json(display_state)

    # Optional: Display agent inboxes/KBs (can be verbose)
    # with st.expander("Agent States"):
    #     agent_states = {}
    #     for name, agent in orchestrator.agents.items():
    #         agent_states[name] = {
    #             "Inbox Size": len(agent.inbox),
    #             "Knowledge Base Keys": list(agent.knowledge_base.keys())
    #         }
    #     st.json(agent_states)


# --- Main Content Area ---
main_container = st.container()

with main_container:
    # The main interaction area is at the top
    interaction_placeholder = st.empty()

    # Use the placeholder to display prompts or status messages
    with interaction_placeholder.container():
         st.subheader("Action Center")
         if st.session_state.app_started:
              # display_human_prompt handles checking the inbox and displaying the relevant prompt/messages
              display_human_prompt(orchestrator)
              
              # Display status if no human input is pending response
              if not orchestrator.project_state.get('pending_human_approval_context'):
                   if orchestrator.project_state['status'] in ['App Live!', 'Project Cancelled', 'Deployment Failed (Escalated)', 'Project Ended']:
                        st.success(f"Workflow concluded: {orchestrator.project_state['status']}")
                   else:
                       st.info("Agents are working... Awaiting next agent result or human interaction point.")
              # The manual "Run Next Cycle" button is now in the sidebar, only when no human input is pending


         else:
              st.info("Use the sidebar to start a new app development project.")


    st.divider()

    # The graph and log are below
    graph_col, log_col = st.columns([0.6, 0.4])

    with graph_col:
        st.subheader("Agent Workflow Visualizer")
        # build_and_display_graph implicitly uses the orchestrator from session_state
        build_and_display_graph(orchestrator)

    with log_col:
        st.subheader("Live Event Log")
        # Use a chat-like format for logs
        log_container = st.container(height=500)
        # Display logs in chronological order for easier reading, but reverse display order
        for msg in reversed(st.session_state.log_messages):
             # Simple rendering for now, could add formatting based on log prefixes
             log_container.text(msg)