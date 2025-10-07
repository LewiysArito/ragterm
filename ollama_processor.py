import abc
from ollama import Client
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_CLIENT

class AbstractLLMClient(abc.ABC):
    @abc.abstractmethod

    def generate(self, 
        prompt: str,
        model: str,
        temperature: float = 0.7,
    )-> str:
        raise NotImplementedError
    
class OllamaProcessor(AbstractLLMClient):
    def __init__(self, client = OLLAMA_CLIENT):
        self.client = client

    def generate(self, prompt: str, model: str = OLLAMA_MODEL, temperature: float = OLLAMA_TEMPERATURE )-> str:
        try:
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": temperature
                },
                stream=False
            )
            return response["response"]
        except Exception as e:
            raise Exception(f"Failed to generate completion: {e}") from e