from vllm import LLM

llm = LLM(
    model="/users/sicheng/Llama-2-7B") # need authentication
output = llm.generate("Hello, my name is")
print("the response is", output)