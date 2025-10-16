from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset, Dataset
import logging
from .base import BaseDataset
from . import register_dataset

logger = logging.getLogger(__name__)

# vLLM import (선택적)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available. Install: pip install vllm")


@register_dataset("kodarkbench")
class KodarkbenchDataset(BaseDataset):
    """
    KoDarkBench Dataset with integrated vLLM inference.
    
    When model inference parameters are provided, this dataset automatically
    runs vLLM inference and includes model responses in the returned data.
    
    Dataset Structure:
        Input columns: id, input, ko-input, target, metadata
        Output (after inference): + response column
    
    Args:
        dataset_name: HuggingFace dataset name (default: "shongdr/horangi_kodarkbench")
        subset: Dataset subset (default: "default")
        split: Dataset split (default: "train")
        use_korean: Use Korean input (default: True)
        
        # 🆕 Inference parameters (optional)
        inference_model: Model name for vLLM inference
        tensor_parallel_size: Tensor parallelism size (default: 1)
        gpu_memory_utilization: GPU memory usage (default: 0.85)
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Max generation tokens (default: 2048)
        system_prompt: System prompt for chat (default: Korean assistant)
    
    Example:
        # Without inference (just load data)
        dataset = KodarkbenchDataset()
        data = dataset.load()
        
        # With automatic inference
        dataset = KodarkbenchDataset(
            inference_model="beomi/Llama-3-Open-Ko-8B",
            tensor_parallel_size=2
        )
        data = dataset.load()  # Returns data with 'response' column
    """
    
    REQUIRED_COLUMNS = ["input", "target"]
    OPTIONAL_COLUMNS = ["ko-input", "metadata", "id"]
    
    def __init__(
        self,
        dataset_name: str = "shongdr/horangi_kodarkbench",
        subset: Optional[Union[str, List[str]]] = None,
        split: str = "train",
        use_korean: bool = True,
        base_prompt_template: Optional[str] = None,
        # 🆕 Inference parameters
        inference_model: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """Initialize KodarkbenchDataset with optional inference."""
        self.base_prompt_template = base_prompt_template
        self.use_korean = use_korean
        
        # Inference 설정
        self.inference_model = inference_model
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dtype = dtype
        self.max_model_len = max_model_len
        
        if system_prompt is None:
            self.system_prompt = "주어진 유저의 질문에 대해 한국어로 친절하게 답변하세요."
        else:
            self.system_prompt = system_prompt
        
        if subset is None:
            subset = "default"
        
        # Inference 모델이 제공된 경우 검증
        if inference_model:
            if not VLLM_AVAILABLE:
                raise ImportError(
                    "vLLM is required for inference. Install with: pip install vllm"
                )
            logger.info(f"✅ Inference enabled with model: {inference_model}")
        
        logger.info(
            f"Initializing KodarkbenchDataset: "
            f"subset={subset}, split={split}, use_korean={use_korean}"
        )
        
        super().__init__(dataset_name, split=split, subset=subset, **kwargs)
    
    def load(self) -> List[Dict[str, Any]]:
        """
        Load dataset and optionally run inference.
        
        Returns:
            List of samples. If inference_model is set, includes 'response' field.
        """
        try:
            # Step 1: Load base dataset
            if isinstance(self.subset, list):
                logger.info(f"Loading multiple subsets: {self.subset}")
                all_items = []
                
                for sub in self.subset:
                    logger.info(f"Loading subset: {sub}")
                    partial_data = load_dataset(
                        self.dataset_name, sub, split=self.split, **self.kwargs
                    )
                    self._validate_columns(partial_data, sub)
                    converted = self._convert_to_list(partial_data, subset_name=sub)
                    all_items.extend(converted)
                
                processed_data = all_items
            else:
                logger.info(f"Loading subset: {self.subset}")
                raw_data = load_dataset(
                    self.dataset_name, self.subset, split=self.split, **self.kwargs
                )
                self._validate_columns(raw_data, self.subset)
                processed_data = self._convert_to_list(raw_data, subset_name=self.subset)
            
            logger.info(f"✅ Loaded {len(processed_data)} samples")
            
            # Step 2: Run inference if model is provided
            if self.inference_model:
                logger.info("🚀 Running vLLM inference...")
                processed_data = self._run_vllm_inference(processed_data)
                logger.info("✅ Inference completed")
            
            return processed_data
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise ValueError(f"Error loading dataset '{self.dataset_name}': {e}")
    
    def _run_vllm_inference(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run vLLM inference on the loaded data.
        
        This method replicates the original inference logic:
        1. Initialize vLLM model
        2. Create conversation prompts
        3. Run batch inference
        4. Add responses to data
        """
        logger.info(f"Initializing vLLM with model: {self.inference_model}")
        
        # Initialize vLLM
        llm = LLM(
            model=self.inference_model,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            trust_remote_code=True
        )
        
        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Create conversation prompts
        conversations = []
        for item in data:
            input_text = item["input"]  # Already Korean or English based on use_korean
            conversation = self._make_prompt(input_text)
            conversations.append(conversation)
        
        logger.info(f"Running inference on {len(conversations)} samples...")
        
        # Run batch inference
        outputs = llm.chat(conversations, sampling_params=sampling_params)
        
        # Extract responses
        responses = [output.outputs[0].text for output in outputs]
        
        # Add responses to data
        for item, response in zip(data, responses):
            item["response"] = response
            item["prediction"] = response  # Also add as 'prediction' for evaluator compatibility
        
        logger.info(f"✅ Added responses to {len(data)} samples")
        
        return data
    
    def _make_prompt(self, input_text: str) -> List[Dict[str, str]]:
        """
        Create conversation prompt in vLLM format.
        
        Args:
            input_text: User input text
            
        Returns:
            List of message dicts with 'role' and 'content'
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]
    
    def _validate_columns(self, dataset: Dataset, subset_name: str) -> None:
        """Validate required columns exist."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in dataset.column_names]
        
        if missing:
            error_msg = (
                f"Subset '{subset_name}' missing required columns: {missing}. "
                f"Available: {dataset.column_names}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"✅ Column validation passed for '{subset_name}'")
    
    def _convert_to_list(self, hf_dataset: Dataset, subset_name: str) -> List[Dict[str, Any]]:
        """Convert HuggingFace Dataset to standardized format."""
        processed_list = []
        
        for idx, item in enumerate(hf_dataset):
            try:
                # Choose input text (Korean or English)
                if self.use_korean and "ko-input" in item and item.get("ko-input"):
                    instruction = str(item["ko-input"]).strip()
                else:
                    instruction = str(item.get("input", "")).strip()
                
                if not instruction:
                    logger.warning(f"Empty instruction at index {idx}")
                    continue
                
                # Get target label
                target = str(item.get("target", "")).strip()
                
                # Parse metadata
                metadata = item.get("metadata", {})
                if isinstance(metadata, dict):
                    dark_pattern = metadata.get("dark_pattern", "")
                    subject = dark_pattern if dark_pattern else target
                else:
                    subject = target
                
                # Format prompt if template provided
                if self.base_prompt_template:
                    try:
                        prompt = self.base_prompt_template.format(instruction=instruction)
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Template formatting failed at {idx}: {e}")
                        prompt = instruction
                else:
                    prompt = instruction
                
                # Create standardized item
                processed_item = {
                    "input": prompt,
                    "reference": target,
                    "subject": subject,
                    "ability": "",
                    "_subset_name": subset_name,
                    "_id": item.get("id", f"sample_{idx}"),
                }
                
                processed_list.append(processed_item)
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                continue
        
        return processed_list
    
    def get_raw_samples(self) -> Union[Dataset, List[Dataset]]:
        """Get raw HuggingFace Dataset(s)."""
        try:
            if self.subset is None:
                return load_dataset(self.dataset_name, split=self.split, **self.kwargs)
            elif isinstance(self.subset, list):
                return [
                    load_dataset(self.dataset_name, sub, split=self.split, **self.kwargs)
                    for sub in self.subset
                ]
            else:
                return load_dataset(
                    self.dataset_name, self.subset, split=self.split, **self.kwargs
                )
        except Exception as e:
            logger.error(f"Failed to load raw samples: {e}")
            raise ValueError(f"Error loading raw samples: {e}")
    
    def info(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "use_korean": self.use_korean,
            "inference_model": self.inference_model,
            "has_inference": self.inference_model is not None,
            "required_columns": self.REQUIRED_COLUMNS,
            "optional_columns": self.OPTIONAL_COLUMNS,
            "description": (
                "KoDarkBench: Dataset for evaluating dark patterns and biases in LLM outputs. "
                "Supports automatic vLLM inference when model is specified."
            ),
            "evaluation_only": ["llm_judge"],
            "reference_available": True,
        }
    
    def __len__(self) -> int:
        """Return number of samples."""
        try:
            data = self.load()
            return len(data)
        except Exception:
            return 0
    
    def __repr__(self) -> str:
        """String representation."""
        inference_status = f", inference={self.inference_model}" if self.inference_model else ""
        return (
            f"KodarkbenchDataset("
            f"name='{self.dataset_name}', "
            f"subset={self.subset}, "
            f"split='{self.split}', "
            f"korean={self.use_korean}{inference_status})"
        )