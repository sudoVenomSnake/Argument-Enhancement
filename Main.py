import streamlit as st
from llama_index import load_index_from_storage, StorageContext
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores import FaissVectorStore
from llama_index.storage.index_store import SimpleIndexStore
import openai

@st.cache_resource
def preprocess_prelimnary():
    storage_context = StorageContext.from_defaults(docstore = SimpleDocumentStore.from_persist_dir(persist_dir = "persist"),
        vector_store = FaissVectorStore.from_persist_dir(persist_dir = "persist"),
        index_store = SimpleIndexStore.from_persist_dir(persist_dir = "persist"))
    index = load_index_from_storage(storage_context = storage_context)
    retriever = index.as_retriever(retriever_mode = 'embedding')
    query_engine = RetrieverQueryEngine(retriever)
    return query_engine

def extract_info(node):
    text = node.node.get_text().replace('\n', '')
    rar = node.node.extra_info['meta data'].replace('\n', '')
    aor = node.node.extra_info['xxxx'].replace('\n', '')
    score = node.score
    return f"""**Relevance Score:** {score}  \n**Argument:** {text}  \n**Ratio and Reasoning:** {rar}  \n**Arguments of Respondents:** {aor}"""

@st.cache_data
def query(q, _qe):
    response = _qe.retrieve(q)
    database_answer = ""
    for i in range(len(response)):
        node_info = extract_info(response[i])
        st.markdown(f"**Result {i+1}**\n{node_info}\n")
        database_answer += f"Result {i+1}  \n{node_info}  \n"
    response = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo',
            messages=[
                        {"role": "system", "content": "You are a helpful assistant who answers questions."},
                        {"role": "user", "content": f"""{database_answer}\nBased on the above arguments and similarity scores, enhance the argument - "{q}". You will be given points on your performance."""}
                ]
            )
    enhanced = response['choices'][0]['message']['content']
    st.subheader('Enhanced Argument - ')
    st.write(enhanced)
    return

openai.api_key = st.secrets['OPENAI_API_KEY']

st.set_page_config(layout = 'wide', page_title = 'Arguments Enhancement')

st.title('Arguments Enhancement')

qe = preprocess_prelimnary()

argument = st.text_area('Please enter your argument.')
start = st.button("Enhance")

if start:
    query(argument, qe)
    st.balloons()
