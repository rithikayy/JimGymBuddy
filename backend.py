import os
import getpass
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent  

def create_rag_agent():
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    model = init_chat_model("gpt-4o-mini")


    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = InMemoryVectorStore(embeddings)

    #STEP A: indexing
    # step 1: loading the files from the folder

    # notes for you: DocumentLoader is an abstract class, DirecLoader and TextLoader is implementation
    loader = DirectoryLoader(
        'gym_data/',
        glob="**/*.txt",
        loader_cls=TextLoader
    ) # to do it per folder, use TextLoader !

    docs = loader.load()

    # step 2: splitting docs

    # notes for you: TextSplitter is an abstract class
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True,)
    all_splits = text_splitter.split_documents(docs)

    # step 3: embeding and storing all these sub docs
    doc_ids = vector_store.add_documents(documents=all_splits)
    print(doc_ids[:3])

    #STEP B: RAG ! relevant splits based on input is retrived (via Retrieval) and a model generates an answer 
    # two types: agentic RAG, two-step chain
    # agentic is when models can request a tool call (fetching data, searching web, etc) when answering a question
    # two-step only required one inference call (agentic needs 2 for generating and producing final) but may no tbe as flexible

    @dynamic_prompt
    def prompt_w_context(request: ModelRequest):
        # Adding context 
        last_query = request.state["messages"][-1].text
        retrieved_docs = vector_store.similarity_search(last_query) # get relevant docs
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        system_msg = ("You are a helpful assistant named Jim, a gym buddy. Answer questions based ONLY on the context "
        "provided. If the context says something specific, represent it accurately. Do not add information from your "
        "general knowledge. If the answer isn't in the context, say so.\n\n" 
                      f"Context:\n{docs_content}")

        return system_msg

    agent = create_agent(model, tools=[], middleware=[prompt_w_context])
    return agent

if __name__ == "__main__":
    agent = create_rag_agent()
    query = "What is a good workoout for begineers? i just started going to the gym"
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
