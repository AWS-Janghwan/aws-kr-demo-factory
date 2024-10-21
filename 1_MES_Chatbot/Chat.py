"""Python file to serve as the frontend"""

import json
import logging
import os
from time import time

from dotenv import load_dotenv
import pandas as pd
import sqlparse
import streamlit as st

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from anthropic_bedrock import AnthropicBedrock


from chatbot_lib import (
    generate_nlp_prompt,
    generate_sql_prompt,
    parse_generated_nlp,
    parse_generated_sql,
    query_sqlite,
    reset_conversation_agent,
)


Anthropic_Claude3_haiku="anthropic.claude-3-haiku-20240307-v1:0"
Anthropic_Claude3_5_Sonnet="anthropic.claude-3-5-sonnet-20240620-v1:0"

# Configuration
load_dotenv()
proj_dir = os.path.abspath('')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Functions
def clear_input():
    if "question" in st.session_state:
        del st.session_state["question"]
    if "messages" in st.session_state:
        del st.session_state["messages"]
    if "selected_question" in st.session_state:
        st.session_state.selected_question = ""

# Page config
st.set_page_config(page_title="MES Chatbot", page_icon=":factory:")
st.header(":factory: MES Chatbot for Manufacturing :factory:")
st.subheader(" MES(제조 관리 시스템)에 질문해 보세요.")
st.caption("이 챗봇 서비스는 제조 관리시스템(MES)을 보다 효율적으로 활용하여 제품의 생산 프로세스를 관리하도록 설계되었습니다. MES는 생산 프로세스를 추적하고, 재고를 관리하며, 제품의 품질을 보장하는 데 사용됩니다. MES는 제품이 제조되고, 기계가 제품 생산에 사용되며, 작업 주문이 생성 및 추적되고, 품질 관리가 수행되는 제조 환경에서 사용되도록 설계되었습니다.")

# Sidebar
reset = st.sidebar.button("Reset Chat",
                          on_click=clear_input)

# Model Temperature setting
# st.sidebar.slider(
#     label='Model Temperature',
#     min_value=0.,
#     max_value=1.,
#     value=0.1,
#     step=0.01,
#     key='temperature'
# )



model_id = st.sidebar.selectbox(
    'Select LLM Model:',
    # ["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-5-sonnet-20240620-v1:0"],
    # ["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-5-sonnet-20240620-v1:0"],
    ["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-5-sonnet-20240620-v1:0"],
    index=0,  # default to claude 3 haiku
    key='model_id'
)

# Load data
with open('prompt_instructions.json', 'r', encoding="utf-8") as file:
    data = json.load(file)
sql_instructions = data['sql_prompt']
nlp_instructions = data['nlp_prompt']

with open('sample_questions.json', 'r', encoding="utf-8") as file:
    example_questions = json.load(file)
question_list = [q for q in example_questions.values()]

db_path = os.path.join(proj_dir, 'mes.db')  # Path to the SQLite database file

# if (model_id == "anthropic.claude-3-haiku-20240307-v1:0") or (model_id == "anthropic.claude-3.5-sonnet-20240229-v1:0"):
if (model_id == "Anthropic_Claude3_haiku") or (model_id == "Anthropic_Claude3_5_Sonnet"):
    model_kwargs = {
        "max_tokens": 4096,
        "temperature": 0.1,
    }
else:
    model_kwargs = {
        "max_tokens": 4096,
        "temperature": 0.1,
    }

# initialize state
if "question" not in st.session_state:
    st.session_state.question = ""
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "무엇을 도와드릴까요?"}]

# # Use a selectbox in the sidebar for the example questions
# question = st.selectbox("Example Questions:", [""] + question_list, key="selected_question")

# initialize the question
# if (question != ""):
#     st.session_state.messages.append({"role": "user", "content": question})


st.write("질문 예시:")

# CSS를 사용하여 박스 스타일 정의
st.markdown("""
<style>
.custom-box {
    border: 2px solid #f0f2f6;
    border-radius: 5px;
    padding: 10px;
    background-color: #aaaaaaa;
}

.copyable {
    user-select: all;
}
</style>
""", unsafe_allow_html=True)

# 텍스트 정의
texts_to_copy = ["마지막으로 취소된 작업 주문은 무엇입니까?",
                "시설 내 각 기계의 상태는 어떻습니까?",
                "각 제품의 평균 생산 시간은 어떻습니까?",
                "현재 유휴 상태인 기계 수는 몇 대입니까?",
                "최근 완료된 작업 주문 생산량보다 재고가 더 많은 제품은 무엇입니까?"]

# 각 텍스트 라인에 대해 박스와 복사 버튼 생성
for i, text in enumerate(texts_to_copy):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f'<div class="custom-box"><span class="copyable">{text}</span></div>', unsafe_allow_html=True)

st.markdown("\n")
    
# question = st.chat_input()
# for question in question_list:
#     col1, col2 = st.columns([10, 1])
#     with col1:
#         st.write(question)
#     with col2:
#         if st.button("Copy", key=question):
#             st.session_state.messages.append({"role": "user", "content": question})

# ...
if prompt := st.chat_input():
    question = {st.session_state.messages.append({"role": "user", "content": prompt})}
    # with st.chat_message("user"):
    #     st.write(prompt)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if type(message["content"]) == pd.DataFrame:
            st.dataframe(message["content"])
        else:
            st.write(message["content"])

### Take input user prompt
# if prompt := st.chat_input():
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.write(prompt)


# Get the latest message in the messages
last_msg = st.session_state.messages[-1]

# Chat logic for follow up
# if last_msg["role"] == "user":
#     messages = []
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             # If the latest message is the question, reset conversation and pass through the initial prompt
#             if last_msg["content"] == question:
#                 st.session_state.conversation = reset_conversation_agent(model_id=model_id, model_kwargs=model_kwargs)
#                 prompt = generate_sql_prompt(question=question, instructions=sql_instructions, db_path=db_path)
#             else:
#                 prompt = last_msg["content"]
#             # Model invocation
#             call_start_time = time()
#             token_client = AnthropicBedrock()
#             logging.info(f"Number of input tokens for sql generation: {token_client.count_tokens(prompt)}")
#             response = st.session_state.conversation.predict(input=prompt)
        
# if last_msg["role"] == "user":
#     messages = []
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             # 사용자가 입력한 메시지를 바로 처리합니다
#             prompt = last_msg["content"]
#             # Model invocation
#             call_start_time = time()
#             token_client = AnthropicBedrock()
#             logging.info(f"Number of input tokens for sql generation: {token_client.count_tokens(prompt)}")
#             if "conversation" not in st.session_state:
#                 st.session_state.conversation = reset_conversation_agent(model_id=model_id, model_kwargs=model_kwargs)
#             response = st.session_state.conversation.predict(input=prompt)
if last_msg["role"] == "user":
    messages = []
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성 중 입니다..."):
            # 사용자가 입력한 메시지를 바로 처리합니다
            prompt = last_msg["content"]
            # 사용자가 입력한 질문을 Text-to-SQL로 변환합니다
            sql_prompt = generate_sql_prompt(question=prompt, instructions=sql_instructions, db_path=db_path)
            # Model invocation
            call_start_time = time()
            token_client = AnthropicBedrock()
            logging.info(f"Number of input tokens for sql generation: {token_client.count_tokens(sql_prompt)}")
            if "conversation" not in st.session_state:
                st.session_state.conversation = reset_conversation_agent(model_id=model_id, model_kwargs=model_kwargs)
            response = st.session_state.conversation.predict(input=sql_prompt)
            logging.info(f"Number of output tokens for sql generation: {token_client.count_tokens(response)}")
            sql_running_time = round(time() - call_start_time, 2)
            logger.info(f"Bedrock SQL generation calling time: {sql_running_time}s\n")
            # 변환된 SQL Query를 SQLite 데이터베이스에 질의합니다
            data = query_sqlite(query=response, db_path=db_path)
            # SQL Query의 결과를 추출합니다
            query, has_sql = parse_generated_sql(response=data)
            # If sql is generated, query the database
            if has_sql:
                # Print the sql query in the chat
                query_fmt = sqlparse.format(query, reindent=True, keyword_case='upper')
                st.text(query_fmt)
                messages.append(query_fmt)
            
    
            if "conversation" not in st.session_state:
                st.session_state.conversation = reset_conversation_agent(model_id=model_id, model_kwargs=model_kwargs)
            response = st.session_state.conversation.predict(input=prompt)

            logging.info(f"Number of output tokens for sql generation: {token_client.count_tokens(response)}")
            sql_running_time = round(time() - call_start_time, 2)
            logger.info(f"Bedrock SQL generation calling time: {sql_running_time}s\n")        
            
            logging.info(f"Number of output tokens for sql generation: {token_client.count_tokens(response)}")
            sql_running_time = round(time() - call_start_time, 2)
            logger.info(f"Bedrock SQL generation calling time: {sql_running_time}s\n")
            query, has_sql = parse_generated_sql(response)
            # If sql is generated, query the database
            if has_sql:
                # Print the sql query in the chat
                query_fmt = sqlparse.format(query, reindent=True, keyword_case='upper')
                st.text(query_fmt)
                messages.append(query_fmt)
                data = query_sqlite(query=query, db_path=db_path)
                # If query returns errors reprompt the model with the supplied error
                trial_cnt = 0
                while type(data) != pd.core.frame.DataFrame and time() - call_start_time < 120 and trial_cnt < 5:
                    pred_start_time = time()
                    new_prompt = f'이전에 생성한 SQL에 다음과 같은 오류가 있습니다:{data}. 이전 오류를 수정하는 SQL을 다시 생성해 주세요'
                    logging.info(f"Number of input tokens for sql generation: {token_client.count_tokens(new_prompt)}")
                    response = st.session_state.conversation.predict(input=new_prompt)
                    logging.info(f"Number of output tokens for sql generation: {token_client.count_tokens(response)}")
                    logger.info(f"Bedrock SQL generation calling time: {round(time() - pred_start_time, 2)}s\n")
                    query, has_sql = parse_generated_sql(response)
                    query_fmt = sqlparse.format(query, reindent=True, keyword_case='upper')
                    st.text(response)
                    messages.append(response)
                    data = query_sqlite(query=query, db_path=db_path)
                    trial_cnt += 1
                if time() - call_start_time > 120 or trial_cnt >= 5:  # timeout
                    response = 'Time out, please retry'
                    nlp_start_time = time()
                else:  # Generate the response (NLP)
                    st.dataframe(data.head(50), hide_index=True)
                    messages.append(data.head(50))
                    nlp_start_time = time()
                    nlp_prompt = generate_nlp_prompt(data=data, question=question, query=query, instructions=nlp_instructions)
                    logging.info(f"Number of input tokens for nlp generation: {token_client.count_tokens(nlp_prompt)}")
                    response = st.session_state.conversation.predict(input=nlp_prompt)
                    logging.info(f"Number of output tokens for nlp generation: {token_client.count_tokens(response)}")
                logger.info(f"Bedrock NLP generation calling time: {round(time() - nlp_start_time, 2)}s\n")
                response = parse_generated_nlp(response)
                nlp_running_time = round(time() - nlp_start_time, 2)
                response += '\n\nSQL 생성 시간: %3.2fs, 답변 생성 시간 : %3.2fs, 총 소요 시 %3.2fs' % (
                sql_running_time, nlp_running_time, time() - call_start_time)
            st.write(response.replace('$', '\\$'))
            messages.append(response)
        messages = [{"role": "assistant", "content": m} for m in messages]
        st.session_state.messages += messages