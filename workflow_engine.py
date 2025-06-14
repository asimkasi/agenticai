import json
import os
import re

import collections # Needed for defaultdict if used internally, keeping it minimal for now
# No direct imports of agents or Orchestrator

class WorkflowEngine:
    """
    Manages the workflow transitions based on a configuration file.
    Decides the next actions based on the current project state and received events (agent results, human input).
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.workflow_config = self._load_config()

    def _load_config(self) -> dict:
        """Loads the workflow configuration from a JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                # Basic validation: Check for 'events' key
                if 'events' not in config or not isinstance(config['events'], dict):
                     raise ValueError("Workflow config must contain an 'events' dictionary.")
                print(f"[WorkflowEngine] Loaded workflow config from {self.config_path}")
                return config
        except FileNotFoundError:
            print(f"[WorkflowEngine] ERROR: Workflow config file not found at {self.config_path}")
            # Return a minimal fallback config or raise an error depending on desired behavior
            return {'events': {}} # Empty events means no transitions happen
        except json.JSONDecodeError as e:
            print(f"[WorkflowEngine] ERROR: Invalid JSON in {self.config_path}: {e}")
            return {'events': {}}
        except ValueError as e:
             print(f"[WorkflowEngine] ERROR: Invalid workflow config structure: {e}")
             return {'events': {}}
        except Exception as e:
            print(f"[WorkflowEngine] ERROR: Unexpected error loading config {self.config_path}: {e}")
            return {'events': {}}

    def process_event(self, event_type: str, event_data: dict, project_state: dict) -> list:
        """
        Processes an incoming event (e.g., agent result, human input) and determines
        a list of actions to be performed by the Orchestrator based on the workflow config.

        Args:
            event_type (str): The type of event ('agent_result', 'human_input', 'start', etc.).
            event_data (dict): Data associated with the event (e.g., agent result content, human response).
            project_state (dict): The current state of the project from the Orchestrator.

        Returns:
            list: A list of action dictionaries prescribing what the Orchestrator should do next.
                  Actions can include: {'type': 'update_state', 'path': '...', 'value': '...'}
                                       {'type': 'send_human_message', 'message_type': '...', 'content': '...'}
                                       {'type': 'delegate_task', 'agent': '...', 'task': '...', 'content': '...'}
                                       {'type': 'check_condition', 'condition_type': '...', 'params': {...}} # Special action
        """
        actions_to_execute = []
        event_handlers = self.workflow_config.get('events', {}).get(event_type, [])

        # Simple event matching: Find the first handler whose conditions match the event data/state
        # For more complex workflows, this might need to evaluate multiple handlers or priorities
        for handler in event_handlers:
            conditions_met = self._check_conditions(handler.get('conditions', {}), event_data, project_state)

            if conditions_met:
                print(f"[WorkflowEngine] Matched handler for event type '{event_type}'")
                # Actions defined in the config handler
                config_actions = handler.get('actions', [])

                # Process actions, evaluating conditions if necessary
                should_continue = True
                for action in config_actions:
                    if action['type'] == 'check_condition':
                        condition_met = self._evaluate_condition(action, project_state)
                        if not condition_met:
                            print(f"[WorkflowEngine] Condition '{action['condition_type']}' not met. Stopping actions for this handler.")
                            should_continue = False
                            break # Stop processing actions for this handler
                        else:
                            print(f"[WorkflowEngine] Condition '{action['condition_type']}' met. Continuing actions.")
                            continue # Condition met, continue to the next action

                    # If not a check_condition or if check_condition was met, add the action
                    # Add context and state/event data for potential templating or execution context
                    executable_action = action.copy()
                    executable_action['_event_data'] = event_data
                    executable_action['_project_state'] = project_state # Include current state for context/templating
                    actions_to_execute.append(executable_action)

                # We found a matching handler and processed its actions, stop looking
                break
        else:
             # No matching handler found
             print(f"[WorkflowEngine] No matching handler found for event type '{event_type}'.")


        return actions_to_execute

    def _check_conditions(self, conditions: dict, event_data: dict, project_state: dict) -> bool:
        """
        Evaluates if the conditions defined in a handler match the event data and project state.
        Simple key-value matching for now.
        """
        if not conditions:
            return True # No conditions means always match

        for condition_type, condition_value in conditions.items():
            if condition_type == 'event_data':
                # Check if specific keys in event_data match expected values
                if not isinstance(condition_value, dict):
                    print(f"[WorkflowEngine] WARNING: 'event_data' condition value must be a dict.")
                    return False
                for key, expected_value in condition_value.items():
                    # Use nested access like 'content.task_name'
                    parts = key.split('.')
                    current_data = event_data
                    try:
                        for part in parts:
                            current_data = current_data[part]
                        if current_data != expected_value:
                            # print(f"[WorkflowEngine] Condition failed: event_data['{key}'] ('{current_data}') != expected ('{expected_value}')")
                            return False # Mismatch
                    except (KeyError, TypeError):
                        # print(f"[WorkflowEngine] Condition failed: Key '{key}' not found in event_data.")
                        return False # Key not found

            elif condition_type == 'project_state':
                 # Check if specific keys in project_state match expected values
                if not isinstance(condition_value, dict):
                    print(f"[WorkflowEngine] WARNING: 'project_state' condition value must be a dict.")
                    return False
                for key, expected_value in condition_value.items():
                    parts = key.split('.')
                    current_state_data = project_state
                    try:
                        for part in parts:
                             current_state_data = current_state_data[part]
                        if current_state_data != expected_value:
                             # print(f"[WorkflowEngine] Condition failed: project_state['{key}'] ('{current_state_data}') != expected ('{expected_value}')")
                             return False # Mismatch
                    except (KeyError, TypeError):
                         # print(f"[WorkflowEngine] Condition failed: Key '{key}' not found in project_state.")
                         return False # Key not found

            # Add other condition types here (e.g., 'has_escalated_issue': true)
            else:
                print(f"[WorkflowEngine] WARNING: Unknown condition type '{condition_type}'")
                return False # Unknown condition type is a failure

        return True # All conditions met

    def _evaluate_condition(self, action_config: dict, project_state: dict) -> bool:
        """Evaluates a specific condition defined in an action."""
        condition_type = action_config.get('condition_type')
        params = action_config.get('params', {})

        if condition_type == 'all_modules_completed':
            # Requires access to project_state['code_modules_status']
            module_statuses = project_state.get('code_modules_status', {})
            if not module_statuses:
                return False # No modules means not all completed

            all_completed = True
            for status in module_statuses.values():
                # Consider 'escalated' as outside the 'completed' path for simple check
                if status not in ['completed']: # If escalated, it's not 'completed' by AI flow
                    all_completed = False
                    break
            print(f"[WorkflowEngine] Condition 'all_modules_completed' evaluated to {all_completed}")
            return all_completed

        # Add other specific condition evaluation logic here
        # Example: if condition_type == 'has_escalated_issue': ...

        print(f"[WorkflowEngine] WARNING: Unknown condition type for action '{condition_type}'")
        return False # Unknown condition evaluation fails the check

    def _substitute_template(self, template: any, event_data: dict, project_state: dict) -> any:
        """
        Basic template substitution (e.g., replace {{event.key}} or {{state.key}}).
        Handles strings and recursively processes dicts/lists.
        """
        if isinstance(template, str):
            # Simple string replacement for keys like {{event.key}} or {{state.key}}
            output = template
            # Substitute event data (using 'result' alias for agent results for consistency)
            for match in re.findall(r"\{\{(event|result)\.([\w\.]+)\}\}", output):
                source, key_path = match
                source_data = event_data if source == 'event' else event_data # event and result alias the same for simplicity now
                parts = key_path.split('.')
                value = source_data
                try:
                    for part in parts:
                        value = value.get(part) if isinstance(value, dict) else getattr(value, part, None) # Handle dict or object access
                        if value is None: break
                    output = output.replace(f"{{{{{source}.{key_path}}}}}", str(value) if value is not None else 'N/A')
                except Exception:
                    output = output.replace(f"{{{{{source}.{key_path}}}}}", 'ERROR_SUBSTITUTING')

            # Substitute project state data
            for match in re.findall(r"\{\{(state|project_state)\.([\w\.]+)\}\}", output):
                 source, key_path = match
                 source_data = project_state
                 parts = key_path.split('.')
                 value = source_data
                 try:
                      for part in parts:
                           value = value.get(part) if isinstance(value, dict) else getattr(value, part, None)
                           if value is None: break
                      output = output.replace(f"{{{{{source}.{key_path}}}}}", str(value) if value is not None else 'N/A')
                 except Exception:
                      output = output.replace(f"{{{{{source}.{key_path}}}}}", 'ERROR_SUBSTITUTING')

            return output
        elif isinstance(template, dict):
            return {k: self._substitute_template(v, event_data, project_state) for k, v in template.items()}
        elif isinstance(template, list):
            return [self._substitute_template(item, event_data, project_state) for item in template]
        else:
            return template # Return other types directly