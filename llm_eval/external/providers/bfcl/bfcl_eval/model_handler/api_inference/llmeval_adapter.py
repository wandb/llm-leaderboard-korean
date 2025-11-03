import os
import time
import yaml
from typing import Any, Dict, Optional

from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import (
    system_prompt_pre_processing_chat_model,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)

# We will import llm_eval models lazily to avoid heavy imports at module load time


class LLMEvalBFCLHandler(BaseHandler):
    """
    An adapter that lets BFCL use llm_eval/models backends (openai_backend, litellm_backend, huggingface_backend).

    Scope: Prompting flow only (is_fc_model=False). FC flow is not implemented in this initial adapter.

    The handler reads the llm-eval model config YAML path from env "LLM_EVAL_MODEL_CONFIG"
    and instantiates the requested backend once, reusing it across queries.
    """

    def __init__(self, model_name: str, temperature: float, registry_name: str, is_fc_model: bool = False, **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS  # Prompting-compatible

        # Load llm-eval model config
        cfg_path = os.environ.get("LLM_EVAL_MODEL_CONFIG")
        if not cfg_path or not os.path.exists(cfg_path):
            raise ValueError("LLM_EVAL_MODEL_CONFIG env var must point to a valid llm-eval model config YAML file")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        model_block = cfg.get("model") or {}
        llm_backend_name: str = model_block.get("name")
        llm_backend_params: Dict[str, Any] = (model_block.get("params") or {}).copy()

        if not llm_backend_name:
            raise ValueError("model.name is required in llm-eval model config")

        # Normalize token parameter naming differences
        # - Some GPT models use max_completion_tokens; llm_eval openai_backend expects max_tokens
        if "max_tokens" not in llm_backend_params and "max_completion_tokens" in llm_backend_params:
            llm_backend_params["max_completion_tokens"] = llm_backend_params.pop("max_completion_tokens")

        # Temperature override from BFCL if not explicitly set in config
        if "temperature" not in llm_backend_params and self.temperature is not None:
            llm_backend_params["temperature"] = self.temperature

        # Batch size = 1 for BFCL step-wise querying
        if "batch_size" not in llm_backend_params:
            llm_backend_params["batch_size"] = 1

        # Instantiate backend
        from llm_eval.models import load_model  # Lazy import

        # For openai backend, ensure chat API is used
        if llm_backend_name == "openai":
            llm_backend_params.setdefault("use_chat_api", True)

        self._backend = load_model(llm_backend_name, **llm_backend_params)

        # Optional EvaluationLogger for token/latency tracking
        self._evaluation_logger: Optional[Any] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Prompting flow
    # ──────────────────────────────────────────────────────────────────────────
    def set_evaluation_logger(self, evaluation_logger: Any) -> None:
        """Set EvaluationLogger for automatic token/latency tracking."""
        self._evaluation_logger = evaluation_logger

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        # Preprocess first system prompt with function docs for prompting models
        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )
        return {"message": []}

    def _query_prompting(self, inference_data: dict):
        """
        Call llm_eval backend with chat messages. Returns raw response object and latency.
        """
        messages = inference_data["message"]

        # The llm_eval openai_backend can accept messages as the "input" when use_chat_api=True
        inputs = [{"input": messages}]

        start = time.time()

        # Wrap inference in Weave op for automatic token/latency tracking
        import weave

        @weave.op()
        def bfcl_inference(backend, inputs_data):
            return backend.generate_batch(
                inputs=inputs_data,
                return_logits=False,
                show_progress=False,
            )

        # If EvaluationLogger is available, wrap inference in log_prediction context
        # to capture token/latency automatically
        if self._evaluation_logger is not None:
            # Extract test ID for tracking
            test_id = inference_data.get("id", "unknown")
            inputs_payload = {
                "input": repr(messages),
                "test_id": test_id,
            }

            with self._evaluation_logger.log_prediction(
                inputs=inputs_payload,
                output=""  # Will be populated automatically
            ):
                results = bfcl_inference(self._backend, inputs)
        else:
            results = bfcl_inference(self._backend, inputs)

        latency = time.time() - start

        # Store input for optional debug logging
        inference_data["inference_input_log"] = {"message": repr(messages)}

        # Return a simple tuple similar to other handlers (result container, latency)
        return results[0], latency

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        # The llm_eval backends return a dict with key "prediction"
        content = api_response.get("prediction", "") if isinstance(api_response, dict) else str(api_response)

        # Token usage is not available from llm_eval backends by default; set to 0 to keep schema
        return {
            "model_responses": content,
            "model_responses_message_for_chat_history": {"role": "assistant", "content": content},
            "input_token": 0,
            "output_token": 0,
        }

    def add_first_turn_message_prompting(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data.setdefault("message", []).extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data.setdefault("message", []).extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        # Append assistant message from parsed response
        inference_data.setdefault("message", []).append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        # Add execution results as a user message for the next turn
        formatted = "\n".join(execution_results)
        inference_data.setdefault("message", []).append({"role": "user", "content": formatted})
        return inference_data

    # Decoders (prompting)
    def decode_ast(self, result, language, has_tool_call_tag):
        return default_decode_ast_prompting(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag: bool):
        return default_decode_execute_prompting(result)

    # ──────────────────────────────────────────────────────────────────────────
    # FC flow is intentionally not implemented in the first iteration
    # ──────────────────────────────────────────────────────────────────────────

