from typing import List, Dict, Any
import openai


@register_model("openai")
class OpenAIModel(BaseModel):
    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: str = None,
        model_name: str = "gpt-4o",  # gpt-4o-mini, gpt-4o-turo 등
        **kwargs,
    ):
        super().__init__()
        if not api_key:
            raise ValueError("API key is required")

        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name

        openai.api_base = api_base
        openai.api_key = api_key

    def generate_batch(
        self, inputs: List[Dict[str, Any]], return_logits: bool = False
    ) -> List[Dict[str, Any]]:

        outputs = []

        for item in inputs:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": item["input"]}],
                    temperature=0.3,
                )

                item["prediction"] = response.choices[0].message.content

                outputs.append(item)

            except Exception as e:
                print(f"Error in OpenAI API calll: {str(e)}")

        return outputs
