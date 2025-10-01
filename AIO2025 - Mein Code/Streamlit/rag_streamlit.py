import streamlit as st #? run app streamlit run file_name.py
import tempfile
import os
import torch

from transformers.utils.quantization_config import BitsAndBytesConfig # for compressing model e.g. 16bits -> 4bits
from transformers import (
                          AutoTokenizer, # Tokenize Model
                          AutoModelForCausalLM,  # LLM Loader - used for loading and using pre-trained models designed for causal language modeling tasks
                          pipeline) # pipline to setup llm-task oritented model
                                    # pipline("text-classification", model='model', device=0)

from langchain_huggingface import HuggingFaceEmbeddings # huggingface sentence_transformer embedding models
from langchain_huggingface.llms import HuggingFacePipeline # like transformer pipeline

from langchain.memory import ConversationBufferMemory # Deprecated
from langchain_community.chat_message_histories import ChatMessageHistory # Deprecated
from langchain_community.document_loaders import PyPDFLoader, TextLoader # PDF Processing
from langchain.chains import ConversationalRetrievalChain # Deprecated
from langchain_experimental.text_splitter import SemanticChunker # module for chunking text

from langchain_chroma import Chroma # AI-native vector databases (ai-native mean built for handle large-scale AI workloads efficiently)
from langchain_text_splitters import RecursiveCharacterTextSplitter # recursively divide text, then merge them together if merge_size < chunk_size
from langchain_core.runnables import RunnablePassthrough # Use for testing (make 'example' easy to execute and experiment with)
from langchain_core.output_parsers import StrOutputParser # format LLM's output text into (list, dict or any custom structure we can work with)
from langchain import hub
from langchain_core.prompts import PromptTemplate
import json

# Save RAG chain builded from PDF
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

# Check if models downloaded or not
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# save downloaded embeding model
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

# Save downloaded LLM
if 'llm' not in st.session_state:
    st.session_state.llm = None

@st.cache_resource # cache model embeddings, avoid model reloading each runtime
def load_embeddings():
    return HuggingFaceEmbeddings(model_name='bkai-foundation-models/vietnamese-bi-encoder')


# set up config
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

#? Read huggingface token in token.txt file. Please paste your huggingface token in token.txt
@st.cache_resource
def get_hg_token():
    with open('token.txt', 'r') as f:
        hg_token = f.read()

@st.cache_resource
def load_llm():
    # MODEL_NAME= "lmsys/vicuna-7b-v1.5"
    MODEL_NAME = "google/gemma-2b-it"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=nf4_config, # add config
        torch_dtype=torch.bfloat16, # save memory using float16
        # low_cpu_mem_usage=True,
        token=get_hg_token(),
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_pipeline = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024, # output token
        device_map="auto" # auto allocate GPU if available
    )

    return HuggingFacePipeline(pipeline=model_pipeline)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
    except Exception as e:
        st.error(f"Đọc file thất bại: {e}")
        return None, 0

    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1, # total sentence collected before perform text split
        breakpoint_threshold_type='percentile', # set splitting style: 'percentage' of similarity
        breakpoint_threshold_amount=95, # split text if similarity score > 95%
        min_chunk_size=500,
        add_start_index=True, # assign index for chunk
    )

    docs = semantic_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(documents=docs,
                                        embedding=st.session_state.embeddings)

    retriever = vector_db.as_retriever()
    parser = StrOutputParser()

    # prompt = PromptTemplate.from_template("""
        # Trả lời ngắn gọn, rõ ràng bằng tiếng việt và chỉ dựa trên thông tin có sẵn bên dưới.
        # Nếu không tìm thấy thông tin, hãy nói rõ là không có dữ liệu liên quan.

        # Nội dung tài liệu:
        # {context}

        # Câu hỏi:
        # {question}

        # Trả lời:
    # """)


    # prompt = PromptTemplate.from_template("""
    #     Dựa vào nội dung sau, hãy:
    #     1. Tóm tắt tối đa 3 ý chính, kèm theo số trang nếu có.
    #     2. Trả lời câu hỏi bằng tiếng Việt ngắn gọn và chính xác.
    #     3. Nếu không có thông tin liên quan, hãy để `"Trả lời"` là `"Không có dữ liệu liên quan"`.

    #     Nội dung tài liệu:
    #     {context}

    #     Câu hỏi:
    #     {question}

    #     Trả lời:
    # """)

    prompt = PromptTemplate.from_template("""
        Bạn là trợ lý AI.

        Dựa vào nội dung sau, hãy:
        1. Tóm tắt tối đa 3 ý chính, kèm theo số trang nếu có.
        2. Trả lời câu hỏi bằng tiếng Việt ngắn gọn và chính xác.
        3. Nếu không có thông tin liên quan, hãy để "Answer" là "Không có dữ liệu liên quan".



		Đảm bảo trả kết quả **ở dạng JSON** với cấu trúc sau:
		{{"main_ideas": [
			{{"point": "Ý chính 1", "source": "Trang ..."}},
			{{"point": "Ý chính 2", "source": "Trang ..."}},
			{{"point": "Ý chính 3", "source": "Trang ..."}}
		],
		"answer": "Câu trả lời ngắn gọn"
		}}

		Vui lòng chỉ in JSON, không giải thích thêm.

		Context:
		{context}

		Question:
		{question}

		Answer:

	""") #? dùng {{ }} để langchain không nhận string bên trong {} là Biến


    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | st.session_state.llm
    | parser
    )

    os.unlink(tmp_file_path)
    return rag_chain, len(docs)

st.set_page_config(page_title="PDF RAG Assistant", layout='wide')
st.title('PDF RAG Assistant')

st.markdown("""
  **Ứng dụng AI giúp bạn hỏi đáp trực tiếp với nội dung tài liệu PDF bằng tiếng Việt**
  **Cách sử dụng đơn giản:**
  1. **Upload PDF** Chọn file PDF từ máy tính và nhấn "Xử lý PDF"
  2. **Đặt câu hỏi** Nhập câu hỏi về nội dung tài liệu và nhận câu trả lời ngay lập tức
""")

#? Tải models
if not st.session_state.models_loaded:
    st.info("Đang tải models...")
    st.session_state.embeddings = load_embeddings()
    st.session_state.llm = load_llm()
    st.session_state.models_loaded = True
    st.success("Models đã sẵn sàng!")
    st.rerun()

#? Upload and Process PDF
uploaded_file = st.file_uploader("Upload file PDF", type="pdf")
if uploaded_file and st.button("Xử lý PDF"):
    with st.spinner("Đang xử lý..."):
        st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
        st.success(f"Hoàn thành! {num_chunks} chunks")


#? Answers UI
if st.session_state.rag_chain:
    question = st.text_input("Đặt câu hỏi:")
    if question:
        with st.spinner("Đang trả lời..."):
            raw_output = st.session_state.rag_chain.invoke(question)
            try:
                result = json.loads(raw_output)
                st.write("📌 **Nội dung chính:**")
                st.write("raw_output:", raw_output)
                for idea in result["main_ideas"]:
                    st.markdown(f"- {idea['point']} (📄 {idea['source']})")

                st.write("🧠 **Trả lời:**")
                st.markdown(result["answer"])

            except json.JSONDecodeError:
                st.error("⚠️ Output không đúng JSON")
                st.text(raw_output)

            # answer = output.split("Answer:")[1].strip() if "Answer:" in output else output.strip()
            # st.write("**Trả lời:**")
            # st.write(answer)