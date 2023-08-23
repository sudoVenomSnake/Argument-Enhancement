import streamlit as st
from llama_index import load_index_from_storage, StorageContext
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores import FaissVectorStore
from llama_index.storage.index_store import SimpleIndexStore
import pandas as pd
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
    # return f"""**Relevance Score:** {score}  \n**Argument:** {text}  \n**Ratio and Reasoning:** {rar}  \n**Arguments of Respondents:** {aor}"""
    return text, rar, aor, score

# def final_answer(database_answer, q):
#     enhance = st.button(label = 'Enhance your argument.')
#     if enhance:
#         response = openai.ChatCompletion.create(
#                 model = 'gpt-3.5-turbo',
#                 messages=[
#                             {"role": "system", "content": "You are a helpful assistant who answers questions."},
#                             {"role": "user", "content": f"""{database_answer}\nBased on the above arguments and similarity scores, enhance the argument - "{q}". You will be given points on your performance."""}
#                     ]
#                 )
#         enhanced = response['choices'][0]['message']['content']
#         st.subheader('Enhanced Argument - ')
#         st.write(enhanced)
#         return

@st.cache_data
def query(q, _qe):
    response = _qe.retrieve(q)
    all_text = []
    all_score = []
    all_rar = []
    all_aor = []
    numbers = []
    for i in range(len(response)):
        text, rar, aor, score = extract_info(response[i])
        st.markdown(f"**Result {i + 1}**\n**Relevance Score:** {score}  \n**Argument:** {text}  \n**Ratio and Reasoning:** {rar}  \n**Arguments of Respondents:** {aor}\n")
        # nodes_info.append(f"Result {i+1}" + node_info)
        numbers.append(f"Argument {i + 1}")
        all_text.append(text)
        all_score.append(score)
        all_rar.append(rar)
        all_aor.append(aor)
    return pd.DataFrame({'Argument Number' : numbers,
                         'Argument' : all_text,
                         'Score' : all_score,
                         'Ratio and Reasoning' : all_rar,
                         'Arguments of Respondents' : all_aor})

openai.api_key = st.secrets['OPENAI_API_KEY']

st.set_page_config(layout = 'wide', page_title = 'Arguments Enhancement')

st.title('Arguments Enhancement')

qe = preprocess_prelimnary()

argument = st.text_area('Please enter your argument.')
start = st.checkbox("Enhance")

if start:
    nodes_info = query(argument, qe)
    
    choice = st.selectbox(label = 'Choose the responses to append.', options = nodes_info['Argument Number'].to_list())
    database_answer = ""
    # for i, x in enumerate(choice):
    #     database_answer += f"Result {i+1}  \n{x}  \n"
    if choice:
        enhance = st.button(label = 'Enhance your argument.')
        if enhance:
            # st.write(f"""Consider my following argument: {argument}.  \nFollowing the exisiting argument and the reasoning given by the judge improve my argument.  \nThink step by step, identify the variables, make a plan and execute.\nThe argument is this: {nodes_info[nodes_info['Argument Number'] == choice]['Argument'].values[0]}.  \nThe reasoning given by the judge behind the argument is : {nodes_info[nodes_info['Argument Number'] == choice]['Ratio and Reasoning'].values[0]}  \nNow using the given argument and the reasoning improve my argument.  \nStay logical and to the point.""")
            response = openai.ChatCompletion.create(
                    model = 'gpt-3.5-turbo',
                    messages=[
                                {"role": "system", "content": "You are a smart, a cunning and an experienced lawyer. You are very shrewd and stay logical. You can find faults and improve existing things logically."},
                                {"role": "user", "content": f"""Consider my following argument: {argument}.\nFollowing the exisiting argument and the reasoning given by the judge improve my argument.\nThink step by step, identify the variables, make a plan and execute.\nThe argument is this: {nodes_info[nodes_info['Argument Number'] == choice]['Argument'].values[0]}.\nThe reasoning given by the judge behind the argument is : {nodes_info[nodes_info['Argument Number'] == choice]['Ratio and Reasoning'].values[0]}\nNow using the given argument and the reasoning improve my argument.\nStay logical and to the point. Based on the recommendation do self reflection, find faults and revise."""}
                        ]
                    )
            enhanced = response['choices'][0]['message']['content']


            st.subheader('Enhanced Argument - ')
            st.write(enhanced)
