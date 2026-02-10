# Prompt templates for the formal verification pipeline
from .agent1_prompts import (
    AGENT1_SYSTEM_PROMPT,
    AGENT1_DECOMPOSITION_PROMPT
)
from .agent2_prompts import (
    get_agent2_system_prompt,
    get_agent2_user_prompt_template,
    get_agent2_prompt  # For backward compatibility
)
from .agent3_prompts import (
    get_agent3_system_prompt,
    get_agent3_user_prompt_template,
    get_agent3_prompt  # For backward compatibility
)

__all__ = [
    'AGENT1_SYSTEM_PROMPT',
    'AGENT1_DECOMPOSITION_PROMPT',
    'get_agent2_system_prompt',
    'get_agent2_user_prompt_template',
    'get_agent2_prompt',
    'get_agent3_system_prompt',
    'get_agent3_user_prompt_template',
    'get_agent3_prompt',
]

