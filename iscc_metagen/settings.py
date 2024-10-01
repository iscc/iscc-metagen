from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from instructor.mode import Mode


class MetaGenSettings(BaseSettings):

    # litellm_model_name: str = Field("ollama/qwen2.5:7b-instruct-q8_0", description="Litellm model name")
    litellm_model_name: str = Field("gpt-4o", description="Litellm model name")

    instructor_mode: Mode = Field(Mode.TOOLS, description="Instructor tool calling mode")

    ollama_num_ctx: int = Field(8192, description="Default context size for loading Ollama models")
    ollama_num_gpu: int = Field(100, description="The number of layers to send to the GPU(s).")
    ollama_temperature: float = Field(
        0.4,
        description="The temperature of the model. Higher temperature will make the model answer more creatively.",
    )

    ollama_num_predict: int = Field(
        -1,
        description=(
            "num_predict (int): Maximum number of tokens to predict when generating text. Default: 128, -1 ="
            " infinite generation, -2 = fill context. Example usage: num_predict 42"
        ),
    )

    ollama_system: str = Field(
        "You are a Metadata expert responsible for collecting comprehensive and precise metadata!",
        description="system prompt for model (overrides what is defined in the Modelfile",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


opts = MetaGenSettings()
