from typing import Any, Dict, List, Optional
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset
from llm_eval.utils.logging import get_logger

logger = get_logger(name="hle_dataset", level="INFO")

@register_dataset("hle")
class HLEDataset(BaseDataset):
    """
    HLE (HAERAE Language Evaluation) Dataset Loader.
    This loader handles the 'HLE' subset from 'HAERAE-HUB/KoSimpleEval'.

    By default, it formats data for multimodal (vision-language) models.
    If 'exclude_images' is set to True, it filters out all samples that contain an image,
    making it compatible with text-only LLMs by providing only text-based problems.
    """

    def __init__(
        self,
        dataset_name: str = "HAERAE-HUB/KoSimpleEval",
        subset: str = "HLE",
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        exclude_images: bool = True,
        **kwargs,
    ):
        if base_prompt_template is None:
            base_prompt_template = (
                "다음은 전문가 수준의 문제입니다. 문제 해결 과정을 단계별로 서술하고, "
                "마지막에 \"최종 정답:\" 형식으로 결론을 제시하십시오.\n\n"
                "문제: {question}"
            )
            
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            subset=subset,
            base_prompt_template=base_prompt_template,
            **kwargs,
        )
        self.exclude_images = exclude_images
        if self.exclude_images:
            logger.info("'exclude_images' is set to True. All samples originally containing images will be skipped.")

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the HLE subset, optionally filtering out image-based samples,
        and returns it in the HRET standard list format.
        """
        logger.info(f"Loading dataset '{self.dataset_name}' with subset '{self.subset}' for split '{self.split}'.")
        
        try:
            dataset = load_dataset(self.dataset_name, name=self.subset, split=self.split, **self.kwargs)
        except Exception as e:
            logger.error(f"Failed to load dataset '{self.dataset_name}' with name '{self.subset}'. Error: {e}")
            return []

        return self._convert_to_list(dataset)

    def _convert_to_list(self, hf_dataset) -> List[Dict[str, Any]]:
        """
        Converts the HuggingFace Dataset, filtering out samples with images if requested.
        """
        processed_list = []
        for item in hf_dataset:
            try: 
                image_base64 = item.get("image")

                # If exclude_images is True and the sample has an image, skip it entirely.
                if self.exclude_images and image_base64:
                    continue

                question_text = item.get("question", "").strip()
                gold_answer = item.get("gold", "").strip()
                
                formatted_question = self.base_prompt_template.format(question=question_text)

                if image_base64:
                    # Format for multimodal models
                    final_input = [
                        {"type": "text", "text": formatted_question},
                        {"type": "image_url", "image_url": {"url": image_base64}}
                    ]
                else:
                    # Format for text-only models
                    final_input = formatted_question

                possible_answers = [ans.strip() for ans in gold_answer.split(',')]
                reference = possible_answers[0] if possible_answers else ""

                processed_list.append(
                    {
                        "input": final_input,
                        "reference": reference,
                        "metadata": {
                            "all_references": possible_answers,
                            "has_image": bool(image_base64),
                        },
                    }
                )
            except Exception as e:
                logger.error(f"Failed to process item: {e}")
                continue
        
        logger.info(f"Loaded {len(processed_list)} samples. (Image-based samples were "
                    f"{'excluded' if self.exclude_images else 'included'}).")
        return processed_list

    def info(self) -> Dict[str, Any]:
        """Returns metadata about the dataset."""
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "description": "HLE: A multimodal QA dataset. Use 'exclude_images=True' to load only text-based problems.",
            "evaluation_only": None,
        }