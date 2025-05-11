from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

model_file = "models/llama-2-7b.Q3_K_L.gguf"

def load_llm(model_file):
    llm = CTransformers(
        model = model_file,
        model_type = "llama",
        max_new_tokens = 1024,
        temperature = 0.01,
        stop = ["<|im_end|>"]
    )
    return llm


def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt


def create_qa_chain(prompt, llm):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain


if __name__ == "__main__":
    template = """<|im_start|>system
    You are a helpful AI assistant. Respond to users accurately then stop and generate no more answers.
    <|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant"""

    prompt = create_prompt(template)
    llm = load_llm(model_file)
    llm_chain = create_qa_chain(prompt, llm)

    question = "What is one plus one?"
    response = llm_chain.invoke({"question":question})
    print(response)
