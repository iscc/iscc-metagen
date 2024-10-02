from litellm import OllamaConfig, completion
import instructor
from iscc_metagen.settings import mg_opts

OllamaConfig(
    num_ctx=mg_opts.ollama_num_ctx,
    num_gpu=mg_opts.ollama_num_gpu,
)

client = instructor.from_litellm(completion, mode=mg_opts.instructor_mode)
