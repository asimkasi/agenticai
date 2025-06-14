from dotenv import load_dotenv
load_dotenv()
import uuid
import datetime
import time # For simulating delays
import collections # For deque for task queuing
from openai import OpenAI # Import the OpenAI client
import os # Import os for environment variables
import re # Import re for parsing (already used in agents, keep it)

# Import the model router and new workflow engine
from model_router import get_model_details_for_agent

import sys # Keep sys import if not already there, for sys.exit
from workflow_engine import WorkflowEngine # Import the new engine

# --- 1. Core AI Agent Base Class ---
# (Keep as is, no changes needed here)
class AIAgent:
    def __init__(self, name: str, role: str, description: str, orchestrator_ref=None, use_llm=True): # Added use_llm flag for testing
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.description = description
        self.knowledge_base = {}
        self.inbox = collections.deque()
        self.orchestrator = orchestrator_ref

        self._current_client = None
        self._current_model_details = {}
        self._use_llm = use_llm # Control whether to call real LLM or mock
        
        # Simple mock LLM response for testing without real LLM
        # This will be overridden by the orchestrator's mock if set
        self._mock_llm_response = "Default mock response."

    def _get_llm_client_details(self):
        """Fetches current model details from the router and initializes client if needed."""
        # Bypass client initialization if not using LLM
        if not self._use_llm:
            # Provide dummy details consistent with model_router output structure
            return None, "mock-model-local" 
            
        new_details = get_model_details_for_agent(self.name)
        
        # Re-initialize client only if model details have changed
        if new_details != self._current_model_details or self._current_client is None:
            self._log(f"Fetching new LLM details for {self.name}: Provider '{new_details['provider']}', Model '{new_details['model_name']}', Base '{new_details['api_base']}'")
            try:
                self._current_model_details = new_details
                self._current_client = OpenAI(
                    base_url=new_details["api_base"],
                    api_key=new_details["api_key"]
                )
            except Exception as e:
                 self._log(f"ERROR initializing OpenAI client for {self.name}: {e}")
                 self._current_client = None # Ensure client is None on error
        return self._current_client, self._current_model_details.get("model_name", "unknown-model") # Return details even if client failed


    def receive_message(self, message: dict):
        self.inbox.append(message)
        self._log(f"Received message from '{message.get('sender', 'Unknown')}' (Type: '{message.get('type')}'). Queued.")

    def send_message(self, recipient_name: str, message_type: str, content: dict, context_id: str = None):
        message = {
            'message_id': str(uuid.uuid4()),
            'timestamp': datetime.datetime.now().isoformat(),
            'sender': self.name,
            'recipient': recipient_name,
            'type': message_type,
            'content': content,
            'context_id': context_id # Important to preserve context through agents
        }
        if self.orchestrator:
            self.orchestrator.route_internal_message(message)
            # self._log(f"Sent message to '{recipient_name}' (Type: '{message_type}').") # Log moved to orchestrator processing
        else:
            print(f"WARNING: No orchestrator set for {self.name}. Message not routed.")

    def _log(self, message: str):
        print(f"[{self.name} ({self.role})] {message}")

    def process_pending_tasks(self):
        if not self.inbox:
            return

        message = self.inbox.popleft()
        
        if message['type'] == 'task':
            task_content = message['content']
            task_name = task_content.get('task_name', 'Unnamed Task')
            context_id = message.get('context_id', 'NoContext')
            self._log(f"Processing task: {task_name} (Context: {context_id[:8]})")
            
            # Simulate work being done
            time.sleep(0.1) # Reduced sleep for faster simulation

            try:
                result_content = self.process_task(task_content)
                # Ensure task_name is in the result content
                result_content['task_name'] = task_name 

                # Attach original context_id to the result message
                self.send_message(
                    recipient_name='Orchestrator',
                    message_type='result',
                    content=result_content,
                    context_id=context_id
                )
            except NotImplementedError:
                self._log(f"ERROR: process_task not implemented for {self.name}.")
                self.send_message(
                    recipient_name='Orchestrator',
                    message_type='status_update', # Use status_update for internal errors
                    content={'task_name': task_name, 'status': 'failed', 'message': f"Task failed: {self.name} has no process_task implementation."},
                    context_id=context_id
                )
            except Exception as e:
                self._log(f"ERROR processing task '{task_name}' (Context: {context_id[:8]}): {e}")
                self.send_message(
                    recipient_name='Orchestrator',
                    message_type='status_update', # Use status_update for exceptions
                    content={'task_name': task_name, 'status': 'failed', 'message': f"Task failed due to exception: {e}"},
                    context_id=context_id
                )
        elif message['type'] == 'feedback':
             # Generic feedback handling
             self._log(f"Received feedback: {message['content'].get('description', 'No description')} (Context: {message.get('context_id', 'NoContext')[:8]})")
             # Specific agents might override receive_message or add logic here
        # Add other message types handling here for general agents

    def process_task(self, task_content: dict):
        """Placeholder for an agent's specific task processing logic."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    def update_knowledge_base(self, key: str, value: any):
        self.knowledge_base[key] = value

    def get_knowledge(self, key: str):
        return self.knowledge_base.get(key)
    
    def generate_response_with_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500):
        """
        Generates a response using the agent's assigned LLM, dynamically fetched from the router.
        Includes a basic mock for testing/development without a running LLM server.
        """
        # Check for orchestrator's mock override first (for specific test responses)
        if self.orchestrator and hasattr(self.orchestrator, '_mock_llm_response_override') and self.orchestrator._mock_llm_response_override:
             # Find a matching key in the override dictionary
             for key, mock_response in self.orchestrator._mock_llm_response_override.items():
                  if key in prompt: # Simple substring match
                       self._log(f"Using Orchestrator-level mock response for prompt starting '{prompt[:50]}...'")
                       # If the mock response is callable, call it to get the actual response
                       if callable(mock_response):
                            return mock_response(prompt)
                       return mock_response
             # If no specific override matches, fall through to agent's mock or real LLM call
             
        if not self._use_llm:
            self._log(f"Using internal mock LLM response for prompt starting '{prompt[:50]}...'")
            # Use agent-specific mock if available, otherwise base class default
            return getattr(self, '_mock_llm_response', "Default mock response: " + prompt[:50] + "...")

        client, model_name = self._get_llm_client_details()
        if client is None:
             self._log(f"ERROR: LLM client not initialized for {self.name}. Cannot generate response.")
             return "Error: Could not generate response - LLM client unavailable."

        try:
            self._log(f"Calling LLM '{model_name}'...")
            chat_completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_content = chat_completion.choices[0].message.content
            self._log(f"LLM response received ({len(response_content)} chars).")
            return response_content
        except Exception as e:
            self._log(f"ERROR: Could not generate response with LLM ({model_name}): {e}")
            return "Error: Could not generate response from LLM."

# --- 2. Specialized AI Agents (Subclasses of AIAgent) ---
# (Keep as is, minimal changes needed within process_task if relying on mock LLM)

class DreamWeaver(AIAgent):
    def __init__(self, orchestrator_ref=None, use_llm=True): # Pass use_llm flag
        super().__init__("Dream Weaver", "Ideator",
                         "Brainstorms, refines app ideas, and defines core features based on user input.", orchestrator_ref, use_llm)
        # Define specific mock response if not using real LLM and no orchestrator override
        if not use_llm:
             self._mock_llm_response = """
Purpose: An innovative app to enhance 'user idea'
Target Users: Tech-savvy individuals and general public
Features:
- Basic Functionality
- User Interface
Monetization Strategy: Freemium model with premium features
""" # Minimal default mock response

    def process_task(self, task_content: dict):
        user_idea = task_content.get('user_idea', 'A general purpose application.')
        refinement_input = task_content.get('refinement_input', '')

        # The generate_response_with_llm method now handles LLM calls or mocks
        prompt = f"""
        Based on the user's idea: "{user_idea}" and refinement input: "{refinement_input}", generate a detailed app concept brief.
        The brief should include:
        - A clear purpose for the app.
        - The primary target users.
        - A list of core features (at least 3, be specific).
        - A suggested monetization strategy.

        Example output format:
        Purpose: An innovative app to enhance 'user idea'
        Target Users: Tech-savvy individuals and general public
        Features:
        - User authentication
        - Data visualization
        - Collaboration tools
        Monetization Strategy: Freemium model with premium features
        """
        llm_response = self.generate_response_with_llm(prompt, temperature=0.7, max_tokens=300)
        
        # Parse the LLM response into a dictionary
        concept = self._parse_llm_concept_response(llm_response, user_idea) # Pass original user_idea for fallback

        self.update_knowledge_base('current_concept', concept)
        self._log("Generated app concept.")
        # Return concept_brief key as expected by Orchestrator/Workflow
        return {"concept_brief": concept, "status": "completed"}

    def _parse_llm_concept_response(self, response_text: str, user_idea_fallback: str) -> dict:
        concept = {
            "purpose": user_idea_fallback.strip('.'), # Use user idea as fallback
            "target_users": "Various users.",
            "features": [],
            "monetization_strategy": "Undefined."
        }
        lines = response_text.split('\n')
        features_section = False
        for line in lines:
            line = line.strip()
            if line.startswith("Purpose:"):
                purpose_text = line.replace("Purpose:", "").strip()
                if purpose_text: # Only update if LLM provided non-empty purpose
                     concept["purpose"] = purpose_text
                features_section = False # Ensure we are out of feature section
            elif line.startswith("Target Users:"):
                concept["target_users"] = line.replace("Target Users:", "").strip()
                features_section = False
            elif line.startswith("Features:"):
                features_section = True
            elif line.startswith("- ") and features_section:
                feature = line.replace("- ", "").strip()
                if feature: # Only add non-empty features
                    concept["features"].append(feature)
            elif line.startswith("Monetization Strategy:"):
                concept["monetization_strategy"] = line.replace("Monetization Strategy:", "").strip()
                features_section = False # Ensure we are out of feature section
            else:
                # If we are in the features section and the line doesn't start with '-',
                # it could be empty lines or other formatting.
                pass

        # Fallback if LLM doesn't list features correctly or features section was empty
        if not concept["features"]:
            concept["features"] = ["Basic Functionality", "User Interface"]
            self._log("WARNING: No features parsed from LLM response. Using fallback features.")

        # Basic cleanup for purpose if it's just the idea + generic text
        if concept["purpose"].lower().startswith("an innovative app to enhance"):
             concept["purpose"] = user_idea_fallback.strip('.') + " App Concept"


        return concept


class MasterBuilder(AIAgent):
    def __init__(self, orchestrator_ref=None, use_llm=True): # Pass use_llm flag
        super().__init__("Master Builder", "Architect",
                         "Translates app concepts into a feasible technical blueprint and selects technologies.", orchestrator_ref, use_llm)
        if not use_llm:
             self._mock_llm_response = """
Blueprint:
Architecture Type: Microservices (scalable)
Modules: Auth, Data, UI
API Specs Summary: RESTful API
Security Considerations: Basic Auth

Tech Stack:
Backend: Python, Flask, SQLite
Frontend: HTML, JS, CSS
Cloud Provider: AWS
CI/CD Tool: Jenkins
"""
    def process_task(self, task_content: dict):
        concept_brief = task_content.get('concept_brief')
        if not concept_brief:
            self._log("ERROR: No concept_brief provided for architecture task.")
            return {"status": "failed", "message": "Missing concept brief"}

        # The generate_response_with_llm method now handles LLM calls or mocks
        prompt = f"""
        Based on the following app concept brief, generate a technical blueprint and propose a suitable tech stack.
        Concept Purpose: {concept_brief.get('purpose', 'N/A')}
        Target Users: {concept_brief.get('target_users', 'N/A')}
        Features: {', '.join(concept_brief.get('features', ['N/A']))}

        Technical Blueprint should include:
        - Architecture Type (e.g., Microservices, Monolithic, Serverless)
        - Key Modules/Services (e.g., AuthenticationService, DataService, UIService)
        - API Specs Summary (e.g., RESTful APIs with OpenAPI documentation)
        - Security Considerations (e.g., OAuth2, Data encryption)

        Tech Stack Proposal should include:
        - Backend (language, framework, database)
        - Frontend (language, framework, styling)
        - Cloud Provider
        - CI/CD Tool

        Example output format:
        Blueprint:
        Architecture Type: Microservices (scalable)
        Modules: AuthenticationService, DataService, UIService, NotificationService
        API Specs Summary: RESTful APIs with OpenAPI documentation
        Security Considerations: OAuth2 for auth, Data encryption at rest/in transit

        Tech Stack:
        Backend: Python, FastAPI, PostgreSQL
        Frontend: TypeScript, React, TailwindCSS
        Cloud Provider: AWS (EC2, RDS, S3, Lambda)
        CI/CD Tool: GitHub Actions
        """
        llm_response = self.generate_response_with_llm(prompt, temperature=0.3, max_tokens=400)

        blueprint, tech_stack = self._parse_llm_architecture_response(llm_response)

        self.update_knowledge_base('current_blueprint', blueprint)
        self.update_knowledge_base('current_tech_stack', tech_stack)
        self._log("Technical blueprint and tech stack proposed.")
        # Return keys expected by Orchestrator/Workflow
        return {"technical_blueprint": blueprint, "tech_stack_proposal": tech_stack, "status": "completed"}

    def _parse_llm_architecture_response(self, response_text: str) -> tuple[dict, dict]:
        blueprint = {
            "architecture_type": "Undefined",
            "modules": [],
            "api_specs_summary": "Undefined",
            "security_considerations": []
        }
        tech_stack = {
            "backend": {"language": "Undefined", "framework": "Undefined", "database": "Undefined"},
            "frontend": {"language": "Undefined", "framework": "Undefined", "styling": "Undefined"},
            "cloud_provider": "Undefined",
            "ci_cd_tool": "Undefined"
        }

        current_section = None
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith("blueprint:"):
                current_section = "blueprint"
            elif line.lower().startswith("tech stack:"):
                current_section = "tech_stack"
            elif current_section == "blueprint":
                if line.lower().startswith("architecture type:"):
                    blueprint["architecture_type"] = line.replace("Architecture Type:", "", 1).strip()
                elif line.lower().startswith("modules:"):
                    # Split by comma, handle empty parts, take first 5-10 modules max for parsing robustness
                    modules_str = line.replace("Modules:", "", 1).strip()
                    blueprint["modules"] = [m.strip() for m in modules_str.split(',') if m.strip()][:10]
                elif line.lower().startswith("api specs summary:"):
                    blueprint["api_specs_summary"] = line.replace("API Specs Summary:", "", 1).strip()
                elif line.lower().startswith("security considerations:"):
                    security_str = line.replace("Security Considerations:", "", 1).strip()
                    blueprint["security_considerations"] = [s.strip() for s in security_str.split(',') if s.strip()][:10]
            elif current_section == "tech_stack":
                if line.lower().startswith("backend:"):
                    parts = [p.strip() for p in line.replace("Backend:", "", 1).split(',') if p.strip()]
                    if len(parts) >= 3:
                        tech_stack["backend"] = {"language": parts[0], "framework": parts[1], "database": parts[2]}
                    elif len(parts) == 2: # Allow just lang, framework
                         tech_stack["backend"] = {"language": parts[0], "framework": parts[1], "database": "Undefined"}
                    elif len(parts) == 1: # Allow just lang
                         tech_stack["backend"] = {"language": parts[0], "framework": "Undefined", "database": "Undefined"}
                elif line.lower().startswith("frontend:"):
                    parts = [p.strip() for p in line.replace("Frontend:", "", 1).split(',') if p.strip()]
                    if len(parts) >= 3:
                        tech_stack["frontend"] = {"language": parts[0], "framework": parts[1], "styling": parts[2]}
                    elif len(parts) == 2: # Allow just lang, framework
                         tech_stack["frontend"] = {"language": parts[0], "framework": parts[1], "styling": "Undefined"}
                    elif len(parts) == 1: # Allow just lang
                         tech_stack["frontend"] = {"language": parts[0], "framework": "Undefined", "styling": "Undefined"}
                elif line.lower().startswith("cloud provider:"):
                    tech_stack["cloud_provider"] = line.replace("Cloud Provider:", "", 1).strip()
                elif line.lower().startswith("ci/cd tool:"):
                    tech_stack["ci_cd_tool"] = line.replace("CI/CD Tool:", "", 1).strip()

        # Basic validation/fallback for critical items
        if not blueprint["modules"]:
             blueprint["modules"] = ["CoreService"]
             self._log("WARNING: No modules parsed from LLM response. Using fallback module.")
        if tech_stack["backend"]["language"] == "Undefined" and tech_stack["frontend"]["language"] == "Undefined":
             tech_stack["backend"] = {"language": "Python", "framework": "Flask", "database": "SQLite"}
             tech_stack["frontend"] = {"language": "HTML", "framework": "None", "styling": "CSS"}
             self._log("WARNING: No tech stack parsed from LLM response. Using fallback tech stack.")


        return blueprint, tech_stack


class AestheticArtist(AIAgent):
    def __init__(self, orchestrator_ref=None, use_llm=True): # Pass use_llm flag
        super().__init__("Aesthetic Artist", "UI/UX Designer",
                         "Designs the app's look, feel, and user experience, creating wireframes and prototypes.", orchestrator_ref, use_llm)
        if not use_llm:
             self._mock_llm_response = """
UI/UX Prototype URL: https://mockup.example.com/mock-prototype-v1-abc
Design Guidelines:
Color Palette: Mock Primary, Secondary
Typography: MockSans
Layout Style: Mock Flow
Icon Style: Mock Solid
"""

    def process_task(self, task_content: dict):
        task_name = task_content.get('task_name')
        # Ensure task_name is available for result processing in Orchestrator
        result_content = {"task_name": task_name, "status": "completed"}

        if task_name == "change_ui":
            refinement_input = task_content.get('refinement_input', 'no changes specified')
            self._log(f"Incorporating UI/UX feedback: '{refinement_input}'")

            prompt = f"""
            The user wants to refine the UI/UX design. Current prototype URL is: {self.get_knowledge('current_prototype_url')}.
            User feedback: "{refinement_input}".

            Generate a new mock URL for the updated prototype and describe the key changes made based on the feedback.
            Example:
            New Prototype URL: https://mockup.example.com/app-prototype-v2-abcde
            Changes Made: Adjusted button sizes to be larger and changed their color to blue. Increased spacing between elements for a more spacious feel.
            """
            llm_response = self.generate_response_with_llm(prompt, temperature=0.5, max_tokens=200)
            
            new_prototype_url = "https://mockup.example.com/app-prototype-v2-" + str(uuid.uuid4())[:4]
            changes_message = "Updated the design per feedback."

            lines = llm_response.split('\n')
            for line in lines:
                if line.startswith("New Prototype URL:"):
                    new_prototype_url = line.replace("New Prototype URL:", "", 1).strip()
                elif line.startswith("Changes Made:"):
                    changes_message = line.replace("Changes Made:", "", 1).strip()

            self.update_knowledge_base('current_prototype_url', new_prototype_url)
            self._log(f"Updated the design per feedback. New prototype: {new_prototype_url}")
            
            result_content["ui_ux_prototype_url"] = new_prototype_url
            result_content["message"] = f"Thanks! {changes_message}"


        else: # Default to the original 'design_ui_ux' task
            concept = task_content.get('concept_brief')
            user_prefs = task_content.get('user_preferences', {})

            self._log(f"Creating UI/UX for app based on concept and preferences: {user_prefs.get('theme', 'default')}")

            prompt = f"""
            Design the UI/UX for an app with the following concept:
            Purpose: {concept.get('purpose', 'N/A')}
            Features: {', '.join(concept.get('features', ['N/A']))}
            User Preferences: Theme - {user_prefs.get('theme', 'clean and modern')}, Color Scheme - {user_prefs.get('color_scheme', 'default')}.

            Provide:
            - A mock URL for the prototype.
            - Design guidelines including color palette, typography, layout style, and icon style.

            Example:
            UI/UX Prototype URL: https://mockup.example.com/app-prototype-v1-abcdef
            Design Guidelines:
            Color Palette: Professional Blue & Grey
            Typography: Inter, sans-serif
            Layout Style: Clean & Grid-based
            Icon Style: Minimalist Line Icons
            """
            llm_response = self.generate_response_with_llm(prompt, temperature=0.6, max_tokens=300)

            prototype_url = "https://mockup.example.com/app-prototype-v1-" + str(uuid.uuid4())[:4]
            design_guidelines = {
                "color_palette": "Professional Blue & Grey",
                "typography": "Inter, sans-serif",
                "layout_style": "Clean & Grid-based",
                "icon_style": "Minimalist Line Icons"
            }

            current_section = None
            lines = llm_response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("UI/UX Prototype URL:"):
                    prototype_url = line.replace("UI/UX Prototype URL:", "", 1).strip()
                elif line.startswith("Design Guidelines:"):
                    current_section = "guidelines"
                elif current_section == "guidelines":
                    if line.startswith("Color Palette:"):
                        design_guidelines['color_palette'] = line.replace("Color Palette:", "", 1).strip()
                    elif line.startswith("Typography:"):
                        design_guidelines['typography'] = line.replace("Typography:", "", 1).strip()
                    elif line.startswith("Layout Style:"):
                        design_guidelines['layout_style'] = line.replace("Layout Style:", "", 1).strip()
                    elif line.startswith("Icon Style:"):
                        design_guidelines['icon_style'] = line.replace("Icon Style:", "", 1).strip()

            self.update_knowledge_base('current_prototype_url', prototype_url)
            self.update_knowledge_base('current_design_guidelines', design_guidelines)
            self._log(f"Generated UI/UX prototype and design guidelines: {prototype_url}")
            
            result_content["ui_ux_prototype_url"] = prototype_url
            result_content["design_guidelines"] = design_guidelines
            

        return result_content # Return the prepared result content


class CodeSage(AIAgent):
    def __init__(self, orchestrator_ref=None, language_preference: str = "Python", use_llm=True): # Pass use_llm flag
        super().__init__("Code Sage", "Developer",
                         "Generates functional code based on technical designs and UI/UX mockups.", orchestrator_ref, use_llm)
        self.language_preference = language_preference
        if not use_llm:
             self._mock_llm_response = """
Generated Code Summary: Mock Python code generated based on requirements.
Unit Tests Summary: Mock unit tests generated (~70% coverage).
"""

    def process_task(self, task_content: dict):
        task_name = task_content.get('task_name')
        module_name = task_content.get('module_name', 'unknown_module')
        requirements = task_content.get('requirements', '')
        design_details = task_content.get('design_details', {})
        bug_report = task_content.get('bug_report', None)
        
        result_content = {"task_name": task_name, "module_name": module_name, "status": "ready_for_qa"}

        if bug_report:
            self._log(f"Fixing bug in {module_name}: {bug_report.get('description')}")
            prompt = f"""
            Fix the following bug in the {self.language_preference} code for the '{module_name}' module.
            Bug Description: {bug_report.get('description')}
            Severity: {bug_report.get('severity')}
            Original Module: {bug_report.get('module')}

            Provide a summary of the code changes made and any updated unit tests.
            Example:
            Generated Code Summary: Fixed Python code for user_authentication based on bug: Login fails when username has special characters. Implemented input sanitization.
            Unit Tests Summary: Updated unit tests for user_authentication to cover bug fix, adding test cases for special characters.
            """
            llm_response = self.generate_response_with_llm(prompt, temperature=0.2, max_tokens=300)
            
            code_summary = f"Fixed {self.language_preference} code for {module_name} based on bug: {bug_report.get('description')}."
            unit_tests_summary = f"Updated unit tests for {module_name} to cover bug fix."

            lines = llm_response.split('\n')
            for line in lines:
                if line.startswith("Generated Code Summary:"):
                    code_summary = line.replace("Generated Code Summary:", "", 1).strip()
                elif line.startswith("Unit Tests Summary:"):
                    unit_tests_summary = line.replace("Unit Tests Summary:", "", 1).strip()

            result_content["generated_code_summary"] = code_summary
            result_content["unit_tests_summary"] = unit_tests_summary

        else: # Normal code generation
            self._log(f"Generating code for '{module_name}' using {self.language_preference}.")
            prompt = f"""
            Generate {self.language_preference} code for the '{module_name}' module based on the following requirements and design details:
            Requirements: {requirements}
            Design Details: {design_details}

            Provide a summary of the generated code and unit tests.
            Example:
            Generated Code Summary: Generated Python code for user_authentication adhering to design: Clean & Grid-based, including user registration, login, and session management.
            Unit Tests Summary: Generated unit tests for user_authentication (coverage ~80%), including tests for valid/invalid credentials.
            """
            llm_response = self.generate_response_with_llm(prompt, temperature=0.7, max_tokens=500)

            code_summary = f"Generated {self.language_preference} code for {module_name}." # Fallback summary
            unit_tests_summary = f"Generated unit tests for {module_name}." # Fallback summary

            lines = llm_response.split('\n')
            for line in lines:
                if line.startswith("Generated Code Summary:"):
                    code_summary = line.replace("Generated Code Summary:", "", 1).strip()
                elif line.startswith("Unit Tests Summary:"):
                    unit_tests_summary = line.replace("Unit Tests Summary:", "", 1).strip()

            result_content["generated_code_summary"] = code_summary
            result_content["unit_tests_summary"] = unit_tests_summary

        time.sleep(0.5) # Simulate coding time
        self.update_knowledge_base(f'code_{module_name}', result_content["generated_code_summary"])
        self._log(f"Finished code generation for {module_name}. Status: ready_for_qa.")
        return result_content # Return the prepared result content


class QualityGuardian(AIAgent):
    def __init__(self, orchestrator_ref=None, use_llm=True): # Pass use_llm flag
        super().__init__("Quality Guardian", "Quality Assurance",
                         "Tests generated code, identifies bugs, analyzes performance, and ensures security.", orchestrator_ref, use_llm)
        if not use_llm:
             self._mock_llm_response = """
Status: passed
Bugs Found: []
Performance Notes: Looks okay.
Security Notes: Basic checks pass.
"""
             # Add a mock response that *will* fail QA based on specific input/attempt
             self._mock_failure_response = """
Status: failed_with_bugs
Bugs Found:
- Description: Simulated bug for testing. Severity: high. Module: {{state.current_task_contexts.{{event.context_id}}.content.module_name}}.
Performance Notes: None.
Security Notes: None.
"""

    def generate_response_with_llm(self, prompt: str, temperature: float = 0.8, max_tokens: int = 400):
        """Override to add specific mock failure logic before standard LLM call."""
        if not self._use_llm:
            # Basic check: if the prompt mentions attempt 0 and it's for 'critical_module', simulate failure
            # This requires the Orchestrator mock override mechanism or a more complex internal mock state
            # For simplicity now, rely on Orchestrator's mock override if needed for specific test cases.
            # Otherwise, use the base class/agent's default mock response.
            return super().generate_response_with_llm(prompt, temperature, max_tokens) # Calls _use_llm=False path


        # If not mocking, proceed with real LLM call
        return super().generate_response_with_llm(prompt, temperature, max_tokens)


    def process_task(self, task_content: dict):
        task_name = task_content.get('task_name')
        code_ref = task_content.get('code_ref', 'N/A')
        test_scope = task_content.get('test_scope', 'unit')
        module_name = task_content.get('module_name', 'unknown_module')
        retry_attempt = task_content.get('retry_attempt', 0)
        
        result_content = {"task_name": task_name, "module_name": module_name, "status": "completed"}


        self._log(f"Running {test_scope} tests on module '{module_name}' ({code_ref}). Attempt: {retry_attempt}")

        prompt = f"""
        Perform a quality assurance test on the '{module_name}' module with a '{test_scope}' scope.
        The code reference is: {code_ref}.
        This is retry attempt: {retry_attempt}.

        Based on these inputs, determine if there are any bugs, performance issues, or security concerns.
        If bugs are found, describe them with severity and module.
        If no bugs, state that.

        Example (No Bugs):
        Status: passed
        Bugs Found: []
        Performance Notes: Initial performance looks good.
        Security Notes: Basic security checks passed.

        Example (With Bugs):
        Status: failed_with_bugs
        Bugs Found:
        - Description: Login fails when username has special characters. Severity: high. Module: user_authentication.
        Performance Notes: Initial performance looks good.
        Security Notes: Basic security checks passed.
        """
        llm_response = self.generate_response_with_llm(prompt, temperature=0.8, max_tokens=400)

        bugs = []
        status = 'passed' # Assume passed unless parsed otherwise
        performance_notes = "N/A"
        security_notes = "N/A"

        lines = llm_response.split('\n')
        current_section = None
        for line in lines:
            line = line.strip()
            if line.lower().startswith("status:"):
                status = line.replace("Status:", "", 1).strip()
            elif line.lower().startswith("bugs found:"):
                current_section = "bugs"
            elif current_section == "bugs" and line.startswith("- "):
                 # Robust parsing for bug description
                 match = re.match(r"^- Description: (.*?)\. Severity: (.*?)\. Module: (.*?)\.$", line)
                 if match:
                     bugs.append({
                         "description": match.group(1).strip(),
                         "severity": match.group(2).strip(),
                         "module": match.group(3).strip()
                     })
                 else:
                      self._log(f"WARNING: Could not parse bug line: {line}")
            elif line.lower().startswith("performance notes:"):
                performance_notes = line.replace("Performance Notes:", "", 1).strip()
                current_section = None
            elif line.lower().startswith("security notes:"):
                security_notes = line.replace("Security Notes:", "", 1).strip()
                current_section = None

        # Ensure status reflects bugs found, prioritizing parsed status if valid
        parsed_status_valid = status in ['passed', 'failed_with_bugs']
        if bugs and (not parsed_status_valid or status != 'failed_with_bugs'):
             status = 'failed_with_bugs'
             self._log(f"Status corrected to 'failed_with_bugs' due to bugs found: {bugs}")
        elif not bugs and parsed_status_valid and status == 'failed_with_bugs':
             status = 'passed'
             self._log("Status corrected to 'passed' as no bugs were found despite LLM claiming failure.")
        elif not parsed_status_valid:
             status = 'failed' # Default to generic failed if parsed status is weird
             self._log(f"WARNING: Unrecognized status '{status}' from LLM. Defaulting to 'failed'.")


        report = {
            "status": status,
            "bugs_found": bugs,
            "performance_notes": performance_notes,
            "security_notes": security_notes,
            "test_scope": test_scope,
            "module_name": module_name
        }
        self.update_knowledge_base(f'test_report_{module_name}_{test_scope}', report)
        self._log(f"Testing completed for {module_name} ({test_scope}). Status: {report['status']}")
        
        result_content["test_report"] = report
        
        # The workflow engine will decide what happens based on report['status']
        return result_content


class DeploymentMaster(AIAgent):
    def __init__(self, orchestrator_ref=None, use_llm=True): # Pass use_llm flag
        super().__init__("Deployment Master", "DevOps Engineer",
                         "Manages infrastructure, deploys the application, and sets up monitoring.", orchestrator_ref, use_llm)
        if not use_llm:
             self._mock_response_success = """
Deployment Status: success
App URL: https://mock-app-live.com/{{event.context_id}}
Monitoring Dashboard URL: https://mock-monitor.com/{{event.context_id}}
"""
             self._mock_response_failure = """
Deployment Status: failure
App URL: N/A
Monitoring Dashboard URL: N/A
Reason: Simulated deployment issue for testing.
"""

    def generate_response_with_llm(self, prompt: str, temperature: float = 0.9, max_tokens: int = 200):
        """Override to inject specific mock responses based on retry attempt if not using real LLM."""
        if not self._use_llm:
             # Parse retry attempt from prompt or task_content if available in self.orchestrator
             retry_attempt = 0
             # Accessing task_content requires a different approach, perhaps pass it to generate_response_with_llm?
             # For now, rely on prompt parsing or a simple attempt counter in the Orchestrator mock if needed.
             # Let's assume for mock purposes, attempt 0 or 1 fails, subsequent attempts succeed.
             # This relies on the orchestrator setting a specific mock override that uses the context_id or retry count.
             # If no specific orchestrator override, fall back to a default mock behavior (e.g., always success or always fail based on agent's default mock).
             
             # If a specific orchestrator mock is NOT set, implement a simple retry simulation here:
             # This is less flexible than the orchestrator override, but works if no override is used.
             # We need the context_id to track attempts persistently *within the agent's mock logic*.
             # This quickly becomes complex without state passed *to* the mock generation call.
             # Sticking to the Orchestrator override pattern is cleaner for complex test flows.
             
             # Default mock behavior if no orchestrator override: Use success mock
             # The orchestrator will handle the retry/failure simulation by providing specific mocks.
             self._log(f"Using Deployment Master's internal mock response for prompt starting '{prompt[:50]}...'")
             return self._mock_response_success # Default mock is success

        # If not mocking, proceed with real LLM call
        return super().generate_response_with_llm(prompt, temperature, max_tokens)


    def process_task(self, task_content: dict):
        task_name = task_content.get('task_name')
        app_package_ref = task_content.get('app_package_ref', 'final_app_build.zip')
        deployment_target = task_content.get('deployment_target', 'cloud')
        environment = task_content.get('environment', 'prod')
        retry_attempt = task_content.get('retry_attempt', 0)
        
        result_content = {"task_name": task_name, "status": "completed"}


        self._log(f"Deploying '{app_package_ref}' to {deployment_target} in {environment} environment. Attempt: {retry_attempt}")
        time.sleep(1) # Simulate deployment time (reduced)

        prompt = f"""
        Simulate a deployment of an application package '{app_package_ref}' to '{deployment_target}' in the '{environment}' environment.
        This is deployment attempt: {retry_attempt}.

        Determine the deployment status (success/failure) and if successful, provide a mock app URL and monitoring dashboard URL.
        Simulate a failure for the first attempt in a 'cloud' environment randomly, but succeed on retries.
        If deployment fails, state 'failure' and 'N/A' for URLs.

        Example (Success):
        Deployment Status: success
        App URL: https://your-app-live-on-cloud.com/12345678
        Monitoring Dashboard URL: https://monitor.example.com/dashboard-abcdef

        Example (Failure):
        Deployment Status: failure
        App URL: N/A
        Monitoring Dashboard URL: N/A
        Reason: Simulated network issue during deployment.
        """
        # generate_response_with_llm handles mocking internally based on self._use_llm
        llm_response = self.generate_response_with_llm(prompt, temperature=0.9, max_tokens=200)

        deployment_status = 'failure' # Default to failure in case parsing fails
        app_url = "N/A"
        monitoring_url = "N/A"
        
        lines = llm_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith("deployment status:"):
                deployment_status = line.replace("Deployment Status:", "", 1).strip()
            elif line.lower().startswith("app url:"):
                app_url = line.replace("App URL:", "", 1).strip()
            elif line.lower().startswith("monitoring dashboard url:"):
                monitoring_url = line.replace("Monitoring Dashboard URL:", "", 1).strip()

        # Validate parsed status
        if deployment_status not in ['success', 'failure']:
             self._log(f"WARNING: Unrecognized deployment status '{deployment_status}' from LLM. Defaulting to 'failure'.")
             deployment_status = 'failure'

        self.update_knowledge_base('last_deployment_url', app_url)

        if deployment_status == 'failure':
            self._log("Deployment failed.")
        else:
            self._log(f"Deployment successful! App URL: {app_url}")

        result_content["deployment_status"] = deployment_status
        result_content["app_url"] = app_url
        result_content["monitoring_dashboard_url"] = monitoring_url
        
        # The workflow engine will decide what happens based on result_content['deployment_status']
        return result_content


# --- 3. The Grand Orchestrator (Refactored) ---

class GrandOrchestrator:
    # Define the path to the workflow configuration file
    WORKFLOW_CONFIG_PATH = 'workflow_config.json'

    def __init__(self, use_llm_for_agents=True): # Added flag to control agent LLM usage
        self.name = "Orchestrator"
        
        # Initialize the Workflow Engine
        self.workflow_engine = WorkflowEngine(self.WORKFLOW_CONFIG_PATH)

        # Instantiate agents, passing the use_llm flag
        self.agents = {}
        self.agents["Dream Weaver"] = DreamWeaver(self, use_llm=use_llm_for_agents)
        self.agents["Master Builder"] = MasterBuilder(self, use_llm=use_llm_for_agents)
        self.agents["Aesthetic Artist"] = AestheticArtist(self, use_llm=use_llm_for_agents)
        self.agents["Code Sage"] = CodeSage(self, use_llm=use_llm_for_agents)
        self.agents["Quality Guardian"] = QualityGuardian(self, use_llm=use_llm_for_agents)
        self.agents["Deployment Master"] = DeploymentMaster(self, use_llm=use_llm_for_agents)

        self.project_state = {
            'status': 'Idle',
            'current_phase': 'Initiation',
            'pending_human_approval_context': None,
            'app_idea': None,
            'concept_brief': None,
            'technical_blueprint': None,
            'tech_stack_proposal': None, # Added missing state
            'ui_ux_prototype_url': None,
            'design_guidelines': None,
            'code_modules_status': {}, # {module_name: 'pending'|'coding'|'ready_for_qa'|'qa_failed'|'completed'|'escalated'}
            'test_results': {},        # {module_name: 'passed'|'failed'|'bypassed'}
            'module_test_retries': collections.defaultdict(int),
            'deployment_retries': 0,
            'final_app_url': None,
            'selected_deployment_target': None, # Added missing state
            'current_task_contexts': {}, # Tracks active tasks by their context_id and original content
            'escalated_issues': {}
        }
        self.internal_message_queue = collections.deque()
        self.human_inbox = collections.deque()
        self.human_outbox = collections.deque()

        # Attribute for tests to override LLM responses
        self._mock_llm_response_override = None


        self._log("Grand Orchestrator initialized. Ready to build!")

    def _log(self, message: str):
        print(f"\n[{self.name}] {message}")
        # Optional: Print state summary less often or controlled by verbosity
        # print(f"  [STATUS] Current Phase: {self.project_state['current_phase']} | Overall Status: {self.project_state['status']}")


    def route_internal_message(self, message: dict):
        self.internal_message_queue.append(message)
        # self._log(f"Queued internal message from '{message.get('sender', 'Unknown')}' (Type: '{message.get('type')}').")

    def _process_internal_messages(self):
        """Processes messages in the internal queue by feeding them to the workflow engine."""
        processed_count = 0
        # Process a batch of internal messages before potentially waiting for human input
        while self.internal_message_queue and processed_count < 10: # Process up to 10 internal messages per cycle
            message = self.internal_message_queue.popleft()
            processed_count += 1
            
            recipient = message['recipient']
            if recipient == self.name: # Message for Orchestrator (results, status_updates)
                self._log(f"Processing message for Orchestrator from '{message.get('sender')}' (Type: '{message['type']}', Context: {message.get('context_id', 'N/A')[:8]}).")
                self._process_orchestrator_message(message)
            elif recipient in self.agents:
                # Route messages destined for other agents directly to their inbox
                self.agents[recipient].receive_message(message)
                self._log(f"Routed message to '{recipient}' (Type: '{message['type']}').")
            else:
                self._log(f"ERROR: Internal message with unknown recipient '{recipient}'. Message: {message}")
        
        if processed_count > 0:
             self._log(f"Processed {processed_count} internal messages.")

    def _process_orchestrator_message(self, message: dict):
         """Handles messages specifically addressed to the Orchestrator."""
         message_type = message['type']
         sender = message['sender']
         content = message['content']
         context_id = message.get('context_id')

         if message_type == 'result' or message_type == 'status_update':
             # Feed agent results/status updates into the workflow engine
             event_data = {
                 'sender': sender,
                 'type': message_type,
                 'content': content,
                 'context_id': context_id
             }
             actions = self.workflow_engine.process_event('agent_result', event_data, self.project_state)
             self._execute_actions(actions, event_data) # Pass event_data for templating context
         # Add handling for other message types if needed


    def send_to_human(self, message_type: str, content: str, options: list = None, context_id: str = None):
        """Sends a user-friendly message to the simulated human interface."""
        human_message = {
            'timestamp': datetime.datetime.now().isoformat(),
            'type': message_type,
            'content': content,
            'options': options if options else [],
            'context_id': context_id # Crucial for flexible human input handling
        }
        self.human_inbox.append(human_message)
        # We store the context_id we are waiting for a *response* to.
        # If a new message arrives for the human *before* they respond to the last one,
        # we overwrite this. This models the UI only being able to show/wait for one explicit prompt response at a time.
        if message_type in ['QUESTION', 'CRITICAL_ISSUE', 'INSTRUCTION']:
             self.project_state['pending_human_approval_context'] = context_id 

        # The GUI (run_gui.py) is responsible for displaying the latest message in human_inbox
        # and handling the interaction, then calling get_human_input.
        self._log(f"Queued message for human (Type: {message_type}, Context: {context_id[:8] if context_id else 'N/A'}).")


    def get_human_input(self, response: str, context_id: str = None) -> None:
        """
        Simulates receiving input from the human user.
        Feeds the input as a 'human_input' event to the workflow engine.
        """
        # If a specific context_id isn't passed with the input, use the one being waited on.
        # This allows responding to the current prompt, or providing unsolicited input with a specific context.
        final_context_id = context_id if context_id is not None else self.project_state['pending_human_approval_context']

        self._log(f"Received human input for context {final_context_id[:8] if final_context_id else 'N/A'}: '{response}'.")

        # Clear the pending flag once input is received for the awaited context
        if self.project_state['pending_human_approval_context'] == final_context_id:
             self.project_state['pending_human_approval_context'] = None

        # Create a human_input event
        event_data = {
            'response': response,
            'context_id': final_context_id
        }

        # Process the human input event through the workflow engine
        actions = self.workflow_engine.process_event('human_input', event_data, self.project_state)
        self._execute_actions(actions, event_data) # Pass event_data for templating context


    def display_project_summary(self):
        """Presents a simplified project summary to the human user."""
        self._log("Here's a quick summary of your app project so far:")
        print(f"  App Idea: {self.project_state.get('app_idea', 'Not yet defined')}")
        print(f"  Current Phase: {self.project_state['current_phase']}")
        print(f"  Overall Status: {self.project_state['status']}")
        if self.project_state.get('concept_brief'):
            print(f"  Concept: {self.project_state['concept_brief'].get('purpose', 'N/A')}")
            print(f"    Features: {', '.join(self.project_state['concept_brief'].get('features', []))}")
        if self.project_state.get('technical_blueprint'):
             print(f"  Architecture: {self.project_state['technical_blueprint'].get('architecture_type', 'N/A')}")
             print(f"    Tech Stack: {self.project_state['tech_stack_proposal'].get('backend', {}).get('framework', 'N/A')} / {self.project_state['tech_stack_proposal'].get('frontend', {}).get('framework', 'N/A')}")
        if self.project_state.get('ui_ux_prototype_url'):
            print(f"  Design Prototype: {self.project_state['ui_ux_prototype_url']}")
        if self.project_state.get('final_app_url'):
            print(f"  Live App: {self.project_state['final_app_url']}")
        
        module_statuses = [f"{m}: {s}" for m, s in self.project_state['code_modules_status'].items()]
        if module_statuses:
            print(f"  Module Statuses: {', '.join(module_statuses)}")
        
        if self.project_state['escalated_issues']:
            print("  !! ESCALATED ISSUES NEEDING YOUR ATTENTION:")
            for issue_id, issue_details in self.project_state['escalated_issues'].items():
                print(f"    - {issue_details['reason']} (Source: {issue_details.get('source_agent', 'N/A')}, Context: {issue_id[:8]})")
        print("--------------------------------------")


    def start_app_development(self, user_idea: str):
        """Initiates the entire app development workflow."""
        self._log(f"Hello! Let's build your app '{user_idea}'. Starting workflow...")
        self.project_state['app_idea'] = user_idea
        
        # Trigger the initial workflow event
        actions = self.workflow_engine.process_event('start', {'user_idea': user_idea}, self.project_state)
        self._execute_actions(actions, {'user_idea': user_idea})


    def _execute_actions(self, actions: list, event_data: dict):
        """Executes a list of actions prescribed by the workflow engine."""
        if not actions:
             self._log("No actions prescribed by the workflow engine.")
             return

        self._log(f"Executing {len(actions)} actions...")
        for action in actions:
            action_type = action.get('type')
            if action_type == 'update_state':
                # Use the engine's templating to get the value, then update state
                path = action.get('path')
                value_template = action.get('value')
                if path is not None and value_template is not None:
                     value = self.workflow_engine._substitute_template(value_template, event_data, self.project_state)
                     self._update_project_state(path, value)
                else:
                     self._log(f"WARNING: Malformed update_state action: {action}")

            elif action_type == 'send_human_message':
                 message_type = action.get('message_type')
                 content_template = action.get('content')
                 options_template = action.get('options', [])
                 context_id_template = action.get('context_id') # Can reference event context

                 if message_type and content_template is not None:
                      content = self.workflow_engine._substitute_template(content_template, event_data, self.project_state)
                      options = self.workflow_engine._substitute_template(options_template, event_data, self.project_state)
                      context_id = self.workflow_engine._substitute_template(context_id_template, event_data, self.project_state) if context_id_template else None

                      # If context_id template didn't resolve or wasn't provided, use the event's context_id
                      # This helps carry context through human interactions derived from agent tasks
                      if not context_id and event_data.get('context_id'):
                          context_id = event_data['context_id']
                      # If still no context, generate a new one for a standalone message
                      if not context_id:
                           context_id = str(uuid.uuid4())

                      self.send_to_human(message_type, content, options, context_id)
                 else:
                      self._log(f"WARNING: Malformed send_human_message action: {action}")


            elif action_type == 'delegate_task':
                agent_name = action.get('agent')
                task_name = action.get('task')
                content_template = action.get('content', {})
                context_id_template = action.get('context_id') # Can reference event context

                if agent_name and task_name:
                     content = self.workflow_engine._substitute_template(content_template, event_data, self.project_state)

                     # Determine the context_id for the delegated task
                     delegated_context_id = None
                     if context_id_template:
                          delegated_context_id = self.workflow_engine._substitute_template(context_id_template, event_data, self.project_state)
                     
                     # If the action specifies to re-use the event's context, use that
                     if action.get('use_event_context_id', False) and event_data.get('context_id'):
                          delegated_context_id = event_data['context_id']

                     # If still no context_id resolved, generate a new one
                     if not delegated_context_id:
                           delegated_context_id = str(uuid.uuid4())
                           self._log(f"Generated new context_id {delegated_context_id[:8]} for delegated task {task_name}")

                     # Add retry attempt count to task content for QA/Deployment
                     if agent_name in ["Quality Guardian", "Deployment Master"]:
                          # Get current retry count for this context/task
                          current_retries = self.project_state['module_test_retries'][delegated_context_id] if agent_name == "Quality Guardian" else self.project_state['deployment_retries']
                          content['retry_attempt'] = current_retries
                          self._log(f"Added retry_attempt {current_retries} to task content for {agent_name}.")


                     self._delegate_task(agent_name, task_name, content, delegated_context_id)
                else:
                     self._log(f"WARNING: Malformed delegate_task action: {action}")

            # check_condition actions are handled *within* process_event before returning actions
            # so they don't need an execution handler here.

            else:
                self._log(f"WARNING: Unknown action type received from workflow engine: {action_type}. Action: {action}")


    def _update_project_state(self, path: str, value: any):
        """Updates a value in the project_state dictionary using a dot-separated path."""
        keys = path.split('.')
        d = self.project_state
        for key in keys[:-1]:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                self._log(f"ERROR: Invalid path '{path}' for state update. Could not find key '{key}'.")
                return
        
        last_key = keys[-1]
        if isinstance(d, dict):
             self._log(f"Updating state: project_state['{path}'] = {value}")
             d[last_key] = value
        else:
             self._log(f"ERROR: Invalid path '{path}' for state update. Parent element is not a dictionary.")


    def _delegate_task(self, agent_name: str, task_name: str, content: dict, context_id: str = None):
        """Helper to delegate a task and track its context."""
        if agent_name not in self.agents:
            self.send_to_human(
                "ERROR",
                f"Orchestrator failed to delegate task '{task_name}': Agent '{agent_name}' not found.",
                context_id=context_id
            )
            self._log(f"ERROR: Attempted to delegate task '{task_name}' to unknown agent '{agent_name}'.")
            return

        if not context_id:
            context_id = str(uuid.uuid4())
            self._log(f"Generated new context_id {context_id[:8]} for task {task_name} delegation.")

        # Store context information for later use, especially for human input and retries
        # Store the *original* content requested for delegation here for potential retries
        self.project_state['current_task_contexts'][context_id] = {
            'task_name': task_name,
            'agent': agent_name,
            'original_content': content # Store content for potential retry re-delegation
        }

        # Ensure task_name is part of the content for the agent to use
        # Create a copy to avoid modifying the original content dict stored for retries
        task_payload = content.copy()
        task_payload['task_name'] = task_name # Explicitly add task_name to content

        self.agents[agent_name].receive_message({
            'sender': self.name,
            'type': 'task',
            'content': task_payload,
            'context_id': context_id
        })
        self._log(f"Delegated '{task_name}' task to {agent_name} (Context: {context_id[:8]}).")
        # Return context_id in case the caller needs it
        return context_id


    def run_simulation_cycle(self):
        """
        Simulates a single tick of the system:
        1. Agents process one task from their inboxes.
        2. Orchestrator processes internal messages (agent results) -> Triggers workflow engine.
        3. Orchestrator processes human input (if any pending in human_outbox) -> Triggers workflow engine.
        """
        self._log(f"--- Running Simulation Cycle ---")
        
        # Process any human input that was received since the last cycle
        # This will trigger the workflow engine via _handle_human_input
        if self.human_outbox:
             # self._handle_human_input processes one item from the outbox
             # Let's process all pending human inputs in one go within the cycle
             processed_human_inputs = 0
             while self.human_outbox and processed_human_inputs < 5: # Limit processing human inputs per cycle
                  response, context_id = self.human_outbox.popleft()
                  self._log(f"Processing human input from outbox: '{response}' (Context: {context_id[:8] if context_id else 'N/A'})")
                  # Directly call the get_human_input logic which processes the event
                  self.get_human_input(response, context_id=context_id) 
                  processed_human_inputs += 1
             if processed_human_inputs > 0:
                  self._log(f"Processed {processed_human_inputs} human inputs from outbox.")


        # Agents process ONE task from their inboxes (to simulate queuing)
        # Agent processing might send messages *to the orchestrator* which end up in internal_message_queue
        self._log("Agents processing tasks...")
        for agent_name, agent_obj in self.agents.items():
             # Note: An agent processing a task might add message(s) to internal_message_queue
            agent_obj.process_pending_tasks()
        self._log("Agent task processing cycle complete.")
        
        # Orchestrator processes internal messages (results, status updates)
        # This feeds results into the workflow engine, potentially triggering more actions/delegations
        self._log("Orchestrator processing internal messages...")
        self._process_internal_messages() # This calls _process_orchestrator_message which calls the workflow engine
        self._log("Orchestrator internal message processing complete.")

        # After processing internal messages, new human prompts might have been generated
        # Check human inbox to see if a new prompt is pending response
        if self.project_state['pending_human_approval_context'] and not self.human_inbox:
             # This can happen if _execute_actions added a message to human_inbox, but it was already processed by the GUI's display loop
             # We need to ensure the GUI displays the latest prompt. The GUI loop (run_gui.py) should handle this.
             pass # Orchestrator's job is done for this cycle, it waits for the next input/cycle trigger

        self._log("--- Cycle End ---")
        # time.sleep(0.1) # Small pause moved to GUI loop for better control

# --- Main Simulation Loop ---
# Keep the main loop as is, it drives the Orchestrator cycles and human interaction.
# It now relies on the Orchestrator's internal logic which uses the WorkflowEngine.

if __name__ == "__main__":
    # Determine if we should use real LLMs or mocks based on env var or flag
    USE_REAL_LLMS = os.getenv("USE_REAL_LLMS", "false").lower() == "true"
    if not USE_REAL_LLMS:
         print("\nNOTE: Running in mock LLM mode. No real LLM calls will be made.")
    orchestrator = GrandOrchestrator(use_llm_for_agents=USE_REAL_LLMS)
    
    simulation_steps = 0

    print("\n\n####################################################################")
    print("##                                                                ##")
    print("##      Welcome to the Genesis AI App Builder!                    ##")
    print("##      (Designed for your Great-Great-Grandfather's Ease)        ##")
    print("##                                                                ##")
    print("####################################################################\n")

    # Check if workflow config exists
    if not os.path.exists(GrandOrchestrator.WORKFLOW_CONFIG_PATH):
         print(f"FATAL ERROR: Workflow configuration file '{GrandOrchestrator.WORKFLOW_CONFIG_PATH}' not found.")
         sys.exit(1)


    initial_idea = input("What app idea do you have today? Tell me in simple words: ")
    orchestrator.start_app_development(initial_idea)

    # The loop continues as long as the project is active or human input is pending.
    # Added a max_steps to prevent infinite loops in simulation if logic goes awry
    MAX_SIMULATION_STEPS = 100 # Increased steps limit

    print("\n--- STARTING SIMULATION ---")
    while orchestrator.project_state['status'] not in ['App Live!', 'Project Cancelled'] and simulation_steps < MAX_SIMULATION_STEPS:
        simulation_steps += 1
        orchestrator._log(f"--- SIMULATION STEP {simulation_steps} ---") # Use orchestrator log for cycle start

        # Run one cycle of processing for agents and internal messages
        # This might generate human prompts in orchestrator.human_inbox
        orchestrator.run_simulation_cycle()

        # After the cycle, check if human input is needed
        # The run_gui.py loop handles displaying the prompt and getting input.
        # In the command line version, we simulate this directly.
        if orchestrator.project_state['pending_human_approval_context'] and orchestrator.human_inbox:
            # In CLI mode, we only handle one human prompt at a time from the inbox
            latest_human_prompt = orchestrator.human_inbox.popleft() 
            print(f"\n[GREAT-GREAT-GRANDFATHER'S TURN (Context: {latest_human_prompt['context_id'][:8]})]")
            
            # Display prompt and options clearly
            print(f"PROMPT: {latest_human_prompt['content']}")
            if latest_human_prompt['options']:
                 print(f"OPTIONS: {', '.join(latest_human_prompt['options'])}")
                 user_response = input("Your choice: ").strip()
            else:
                 user_response = input("Your response: ").strip()
                 
            # Feed the human response back into the orchestrator, linking it to the prompt's context
            # This response will be processed at the start of the *next* simulation cycle
            orchestrator.get_human_input(user_response, context_id=latest_human_prompt['context_id'])
        elif orchestrator.project_state['status'] not in ['App Live!', 'Project Cancelled', 'Deployment Failed (Escalated)', 'Project Ended']:
            # If no human input pending, and workflow is not finished, continue to next cycle automatically in CLI
             print("\n[SYSTEM] No human input needed. Running next simulation cycle automatically...")
             # The loop condition will cause the next cycle to run automatically

        time.sleep(0.1) # Small pause between cycles for readability in CLI


    print("\n\n####################################################################")
    print("##                  SIMULATION SUMMARY                            ##")
    print("####################################################################")
    orchestrator.display_project_summary()

    final_status = orchestrator.project_state['status']
    if final_status == 'App Live!':
        print("\n**SUCCESS! Your app is live!**")
    elif final_status == 'Project Cancelled':
        print("\n**Project was cancelled by human intervention.**")
    elif 'Failed (Escalated)' in final_status or final_status == 'Project Ended':
        print(f"\n**Simulation ended with status: {final_status}.**")
    else:
        print(f"\nSimulation ended at step {simulation_steps} with status: {final_status}. Max steps reached or an unhandled state occurred.")
    
    # Example of trying a new feature idea after success (CLI only)
    if final_status == 'App Live!':
         print("\n--- POST-LAUNCH INTERACTION DEMONSTRATION ---")
         post_launch_input = input("Your app is live! You can type 'new feature [your idea]' to request a new feature, or 'exit' to quit: ").strip().lower()
         if post_launch_input.startswith("new feature"):
              new_feature_idea = post_launch_input.replace("new feature", "", 1).strip()
              if new_feature_idea:
                   orchestrator.get_human_input(f"new feature: {new_feature_idea}", context_id="live_app_feedback") # Use a specific context for live feedback
                   print("\nNew feature request received. Run more cycles or the GUI to see progress!")
                   # You would typically run more simulation cycles here to process the new request
                   # orchestrator.run_simulation_cycle()
              else:
                   print("Please provide an idea after 'new feature'.")
         elif post_launch_input == 'exit':
             print("Exiting.")
         else:
              print("Input ignored. Exiting.")
