from langchain_community.llms import CTransformers
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

model_file = "models/llama-2-7b.Q3_K_L.gguf"
vector_db_path = ""


def load_llm(model_file):
    llm = CTransformers(
        model = model_file,
        model_type = "llama",
        max_new_tokens = 1024,
        temperature = 0.01,
    )
    return llm


def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt


def create_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k":3}),# search most 3 similar documents 
        return_source_documents = True, # trả về các document liên quan
        chain_type_kwargs={"prompt":prompt}
    )
    return llm_chain

def read_vector_db(vector_db_path):
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db


if __name__ == "__main__":
    db = read_vector_db("vector_db/info_history_db_faiss")
    llm = load_llm(model_file)

    template = """<|im_start|>system\nUse the following information to answer the question. If you don't know the answer, say you don't know, don't try to make up the answer.\n
        {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""

    prompt = create_prompt(template)

    llm_chain  =create_chain(prompt, llm, db)

    question = "Who named the city of Pittsburgh?"
    response = llm_chain.invoke({"query": question})
    print(response)