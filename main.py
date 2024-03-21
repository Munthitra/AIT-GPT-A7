import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFacePipeline

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

#calling vector from local
db_file_name = './vector-store/nlp_stanford'
emb_model_name = 'hkunlp/instructor-base'
model_id = './models/fastchat-t5-3b-v1.0/'

prompt_template = """
    Welcome to AIT! I'm AIT-GPT, your friendly campus chatbot, here to help you navigate our university.
    Whether you have questions about AIT's everything please feel free to ask.
    I'm here to provide you with the information you need!
    Context: {context}
    Question: {question}
    Answer:
    """.strip()

PROMPT = PromptTemplate.from_template(
    template = prompt_template
)

embedding_model = HuggingFaceInstructEmbeddings(
    model_name = emb_model_name,
    model_kwargs = {"device" : device}
)

vectordb = FAISS.load_local(
    folder_path = db_file_name,
    embeddings = embedding_model,
    index_name = 'nlp' #default index
)

retriever = vectordb.as_retriever()

tokenizer = AutoTokenizer.from_pretrained(
    model_id)

tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    device_map = 'cpu',
)

pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens = 256,
    model_kwargs = {
        "temperature" : 0,
        "repetition_penalty": 1.5
    }
)

llm = HuggingFacePipeline(pipeline = pipe)

question_generator = LLMChain(
    llm = llm,
    prompt = CONDENSE_QUESTION_PROMPT,
    verbose = True
)

doc_chain = load_qa_chain(
    llm = llm,
    chain_type = 'stuff',
    prompt = PROMPT,
    verbose = True
)

memory = ConversationBufferWindowMemory(
    k=3, 
    memory_key = "chat_history",
    return_messages = True,
    output_key = 'answer'
)

chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True,
    memory=memory,
    verbose=True,
    get_chat_history=lambda h : h
)
# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("AIT-GPT Demo"),
    dcc.Textarea(id='input-text', placeholder="Enter text...", rows=4, cols=50),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='output-text')
])

# Define callback to translate text when button is clicked
@app.callback(
    Output('output-text', 'children'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-text', 'value')]
)
def update_output(n_clicks, input_text):
    if n_clicks > 0:
        output = chain({"question": input_text})
        answer = output['answer'].replace('<pad> ', '').replace('\n', '')
        ref_list = []

        for doc in output['source_documents']:
            metadata = doc.metadata
            filename = metadata['source'].split('/')[-1]
            page_no = metadata['page'] + 1
            total_pages = metadata['total_pages']
            ref_list.append({"ref_text": f"{filename} - page {page_no}/{total_pages}",
                             "ref_link": f"{filename}#page={page_no}"})
        return html.Div([
            html.P(f"Answer: {answer}"),
            html.P("Source Documents:"),
            html.Ul([html.Li(html.A(item['ref_text'], href=item['ref_link'])) for item in ref_list])
        ])

if __name__ == '__main__':
    app.run_server(debug=True)

