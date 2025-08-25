from vllm import LLM

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct") # need authentication
output = llm.generate("Hello, my name is")
print(output)