from pydantic import SecretStr, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

from ai.prompts.summary_llm_prompt import (
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_TEMPLATE,
)
from ai.prompts.research_prompt import (
    RESEARCH_SYSTEM_PROMPT,
    RESEARCH_USER_TEMPLATE,
    RESEARCH_FINALIZE_INSTRUCTION,
)
from ai.prompts.draft_prompt import (
    DRAFT_SYSTEM_PROMPT,
    DRAFT_USER_TEMPLATE,
)
from ai.prompts.classify_prompt import (
    CLASSIFY_SYSTEM_PROMPT,
    CLASSIFY_USER_TEMPLATE,
)


class PromptSettings(BaseModel):
    summary_system_prompt: str = SUMMARY_SYSTEM_PROMPT
    summary_user_template: str = SUMMARY_USER_TEMPLATE
    research_system_prompt: str = RESEARCH_SYSTEM_PROMPT
    research_user_template: str = RESEARCH_USER_TEMPLATE
    research_finalize_instruction: str = RESEARCH_FINALIZE_INSTRUCTION
    draft_system_prompt: str = DRAFT_SYSTEM_PROMPT
    draft_user_template: str = DRAFT_USER_TEMPLATE
    classify_system_prompt: str = CLASSIFY_SYSTEM_PROMPT
    classify_user_template: str = CLASSIFY_USER_TEMPLATE


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    anthropic_api_key:SecretStr
    openai_api_key:SecretStr
    voyageai_api_key:SecretStr


    database_url:str

    debug:bool = False
    request_timeout:int = 30

    rerank_model:str = "rerank-2.5"

    repo_owner:str
    repo_name:str

    github_pat_key:str
    
    classify_model:str = "claude-haiku-4-5"
    research_model:str = "claude-sonnet-4-6"
    research_max_iterations: int = 6
    github_mcp_url:str = "https://api.githubcopilot.com/mcp/"

    prompts: PromptSettings = Field(default_factory=PromptSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
