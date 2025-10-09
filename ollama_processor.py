import abc
from typing import List
from ollama import Client
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_CLIENT, OLLAMA_TEMPLATE_TEXT 

class AbstractLLMClient(abc.ABC):
    @abc.abstractmethod
    def generate(self, 
        prompt: str, 
        model: str = OLLAMA_MODEL, 
        temperature: float = OLLAMA_TEMPERATURE
    )-> str:
        """Generate response from llm"""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_prompt_from_template(self,
        query: str,
        sources: List[str],
        template: str = OLLAMA_TEMPLATE_TEXT
    ):
        """Generate prompt from template"""
        raise NotImplementedError
    
class OllamaProcessor(AbstractLLMClient):
    def __init__(self, client = OLLAMA_CLIENT):
        self.client = client

    def generate(self, prompt: str, model: str = OLLAMA_MODEL, temperature: float = OLLAMA_TEMPERATURE)-> str:
        """Generate response from llm"""
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
    
    def generate_prompt_from_template(self,
        query: str,
        sources: List[str],
        template: str = OLLAMA_TEMPLATE_TEXT
    )->str:
        """Generate prompt from template"""
        prompt = template.replace("{query}", query)
        prompt = prompt.replace("{sources}", sources)

        return self.generate(prompt)
