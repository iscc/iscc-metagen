from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from instructor.mode import Mode


class MetaGenSettings(BaseSettings):

    litellm_model_name: str = Field(
        "ollama/qwen2.5:7b-instruct-q8_0", description="Litellm model name"
    )
    instructor_mode: Mode = Field(Mode.TOOLS, description="Instructor tool calling mode")
    ollama_num_ctx: int = Field(8192, description="Default context size for loading Ollama models")
    ollama_num_gpu: int = Field(100, description="The number of layers to send to the GPU(s).")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


opts = MetaGenSettings()
