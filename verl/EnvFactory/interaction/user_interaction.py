# Todo: Fix bug and improve the efficiency

from dataclasses import dataclass
from dotenv import load_dotenv
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4
from openai import AsyncOpenAI

from .base import BaseInteraction
from MCPFactory.configs.system_prompt import USER_SYSTEM_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class User:
    def __init__(self, config: dict = {}):
        self.base_url = os.environ.get('USER_BASE_URL')
        self.api_key = os.environ.get('USER_API_KEY')
        self.model = os.environ.get("USER_MODEL")
        self.config = config
        self.max_retry = config.get('max_retry', 3)
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def interact(self, messages: list, initial_config: dict) -> str:
        """Interact with the user model and get a response."""
        user_prompt = USER_SYSTEM_PROMPT.format(initial_config = initial_config)
        if messages[0]['role'] == 'system':
            messages[0]['content'] = user_prompt
        else:
            messages = [{"role": "system", "content": user_prompt}] + messages

        for _ in range(self.max_retry):    
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.get("temperature", 1),
                top_p=self.config.get("top_p", 0.95),
                presence_penalty=self.config.get("presence_penalty", 1),
                stream=False
            )

            content = response.choices[0].message.content
            try:
                match = re.search(r'<response>(.*?)</response>', content, re.DOTALL)
                if match:
                    return match.group(1).strip()
                else:
                    continue
            except Exception as e:
                continue

        return "###FAIL###"


class UserInteraction(BaseInteraction):
    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.user = User(config)

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:
        """Initialize a new interaction session."""
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": 0.0,
        }
        
        logger.info(f"Started interaction {instance_id}")
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """
        Interaction with the user.
        
        Returns:
            should_terminate: True if the user output stop tokens or reach max turns.
            response_text: The user response.
            reward_score: 1.0 for success, 0.0 for failure, 0.5 otherwise.
            metadata: Additional information.
        """
        instance = self._instance_dict[instance_id]

        user_response = await self.user.interact(messages=messages, initial_config=kwargs['initial_config'])

        instance['response'] = user_response

        should_terminate = "###SUCCESS###" in user_response or "###FAILURE###" in user_response

        reward = await self.calculate_score(instance_id)

        return should_terminate, user_response, reward, {}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        if instance_id not in self._instance_dict:
            return 0.0
        
        instance = self._instance_dict[instance_id]
        user_response = instance['response']
        if "###SUCCESS###" in user_response:
            return 1.0
        elif "###FAILURE###" in user_response:
            return 0.0
        else:
            return 0.5

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]