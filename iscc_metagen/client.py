from litellm import OllamaConfig, completion
import instructor
from iscc_metagen.settings import opts

OllamaConfig(
    num_ctx=opts.ollama_num_ctx,
    num_gpu=opts.ollama_num_gpu,
)

client = instructor.from_litellm(completion, mode=opts.instructor_mode)
