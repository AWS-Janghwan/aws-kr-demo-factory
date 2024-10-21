from datetime import datetime
import logging
import os
import re
import time
import json
import sqlite3

import boto3
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase


def get_llm(model_id="anthropic.claude-v2:1", model_kwargs="""{"maxTokenCount": 4000,"temperature": 0.1}"""):
    # """Creates the LLM object for the langchain conversational bedrock agent.
    # Parameters:
    #     model_kwargs (dict): Dictionary of model_kwargs to be passed to the Bedrock model.
    # Returns:
    #     langchain.llms.bedrock.Bedrock: Bedrock model
    # """
    session = boto3.Session(
        region_name=os.getenv("AWS_REGION", "us-west-2"), profile_name=os.getenv("AWS_PROFILE")
    )
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        endpoint_url='https://bedrock-runtime.'+os.getenv('AWS_REGION', 'us-west-2')+'.amazonaws.com',
        )
    if (model_id == "anthropic.claude-3-haiku-20240307-v1:0") or (model_id == "anthropic.claude-3-5-sonnet-20240620-v1:0"):
        llm = BedrockChat(
            client=bedrock_client,
            model_id=model_id,
            model_kwargs=model_kwargs)
    else:
        llm = Bedrock(
            client=bedrock_client,
            model_id=model_id,
            model_kwargs=model_kwargs)

    return llm

def reset_conversation_agent(verbose=True, model_id="anthropic.claude-v2:1", model_kwargs="""{"maxTokenCount": 4000,"temperature": 0.1}"""):
# def reset_conversation_agent(verbose=True, model_id="anthropic.claude-v2:1", model_kwargs="""{"maxTokenCount": 4000,"temperature": 0.1}"""):
    # """Resets the langchain conversational bedrock agent with a new 
    # conversation history, tempreature and model_id.
    # Parameters
    # ----------
    # verbose :
    #     Flag for printing model react prompts (default=True)
    # model_kwargs:
    #     Dictionary of model_kwargs to be passed to the Bedrock model.
    # Returns
    # ----------
    # langchain.chains.ConversationChain :
    #     A langchain ConversationChain initialized with the specified parameters
    # """
    memory = ConversationBufferMemory(ai_prefix='Assistant')
    conversation = ConversationChain(
        llm=get_llm(model_id=model_id, model_kwargs=model_kwargs), verbose=verbose, memory=memory
    )

    # langchain prompts do not always work with all the models. This prompt is tuned for Claude
    claude_prompt = PromptTemplate.from_template("""다음은 사람과 AI 간의 친근한 대화입니다. 당신은 반드시 한글로 답변을 생성합니다.
    AI는 대화의 맥락에서 많은 구체적인 세부 사항을 제공합니다. AI가 질문에 대한 답변을 모르는 경우, 솔직하게 모르겠다고 말합니다. 질문에 대해 확실하지 않거나 질문의 일부 매개변수가 불분명하여 답변을 제공할 수 없는 경우, 추측하기보다는 후속 질문을 하십시오. AI는 한글로만 답변을 합니다.

    현재 대화 내용:
    {history}


    Human: {input}


    Assistant:
    """)

    conversation.prompt = claude_prompt
    return conversation


def read_content(file_path):
    """
    Reads and returns the entire content of a file.
    Parameters
    ----------
    file_path (str): 
        The path to the file that needs to be read.
    Returns
    ----------
    str: 
        The entire content of the file as a string
    """
    with open(file_path, 'r', encoding="utf-8") as file:
        # Read the entire content of the file
        message = file.read()
    return message


def generate_sql_prompt(question,
                        instructions,
                        db_path,
                        current_date=datetime.strftime(datetime.now(), '%A, %Y-%m-%d')):
    # """텍스트-SQL용 프롬프트를 생성합니다. 현재 Claude에 맞춰 조정되어 있으며 XML 태그를 사용하여 컨텍스트의 주요 부분을 구분합니다. 모델에 테이블 설명, 테이블 이름, 현재 날짜, 테이블 스키마, 샘플 데이터를 제공합니다. 모델 특정 지침을 포함합니다.
    # Parameters
    # ----------
    # question :
    #     답변 해야 할 질문
    # instructions :
    #     텍스트-SQL용으로 설계된 모델 특정 지침
    # db_path :
    #     SQLite 데이터베이스 파일의 경로
    # current_date :
    #     A specified date. (default=datetime.now())
    # Returns
    # ----------
    # str :
    #     텍스트-SQL 생성용 프롬프트
    # """

    # begin the prompt with the current date
    sql_prompt = f"Current Date: {current_date}\n\n"
    # add db description and schema
    sql_prompt += f"""<description>\n 이 데이터베이스는 제품의 생산 프로세스를 관리하도록 설계된 소프트웨어 시스템인 제조실행시스템(MES)을 시뮬레이션합니다. MES는 생산 프로세스를 추적하고, 재고를 관리하며, 제품의 품질을 보장하는 데 사용됩니다. MES는 제품이 제조되고, 기계가 제품 생산에 사용되며, 작업 주문이 생성 및 추적되고, 품질 관리가 수행되는 제조 환경에서 사용되도록 설계되었습니다.\n</description>\n\n데이터베이스 스키마는 다음과 같습니다:"""
    schema = get_mes_schema(db_path=db_path)
    sql_prompt += f"\n\n<schema> {schema} \n</schema>\n\n"
    # add in user question and task instructions
    sql_prompt += instructions
    sql_prompt += "\n\nThe task is:"
    sql_prompt += f"\n<task>\n{question}\n</task>\n\n"
    logging.info(f"Length of prompt for SQL generation: {len(sql_prompt)} characters\n")

    return sql_prompt


def generate_nlp_prompt(data, question, query, instructions):
    # """Generates a prompt given the results of the text-to-SQL request
    # Parameters
    # ----------
    # data :
    #     결과 데이터는 MES Database를 기반으로 생성.
    # question :
    #     원래 질문
    # query :
    #     데이터 조회에 사용된 쿼리
    # instructions :
    #     제공된 모델에 특화되어 설계된 지침
    # Returns
    # ----------
    # str :
    #     SQL 결과를 요약하기 위한 설계된 프롬프트
    # """
    # nlp_prompt = """<task>\n%s\n</task>\n\n<sql>%s</sql>\n\n<data>\n%s\n</data>\n\n%s"""

    # prompt = nlp_prompt % (question, query, data, instructions)

    # logging.info(f"NLP 답변 생성을 위한 프롬프트의 길이: {len(prompt)} 문자\n")

    # return prompt
    
    nlp_prompt = """다음은 SQL 쿼리 결과를 요약하는 텍스트입니다.
    원래 질문: %s
    SQL 쿼리: %s
    결과 데이터: %s
    지침: %s"""

    prompt = nlp_prompt % (question, query, data, instructions)

    logging.info(f"NLP 답변 생성을 위한 프롬프트의 길이: {len(prompt)} 문자\n")

    return prompt
    

def get_mes_schema(db_path):
    # """Get the schema of the mes database
    # Parameters
    # ----------
    # db_path :
    #     Path to the SQLite database file
    # Returns
    # ----------
    # str :
    #     schema of the mes database as a string
    # """
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=8)
    tables = db.get_usable_table_names()
    schema = db.get_table_info_no_throw(tables)
    return schema

def query_sqlite(query, db_path):
    # """Creates a connection to a SQLite database and executes a query
    # Parameters
    # ----------
    # query :
    #     An string containing SQL code
    # db_path :
    #     Path to the SQLite database file
    # Returns
    # ----------
    # pandas.DataFrame :
    #     the results of the SQL query
    # """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        return e

def parse_generated_sql(response):
    # """Given a text-to-SQL generated output, extract the provided SQL. If query
    # cannot be extracted than return the full response.
    # Parameters
    # ----------
    # response :
    #     text-to-SQL generated string
    # Returns
    # ----------
    # str :
    #     either the SQL from the generated text or the original response
    # """
    logging.info(f"\nResponse that should include SQL code:\n{response}\n")
    try:
        start_sql = re.search(r'<sql>', response).end()
        end_sql = re.search(r'</sql>', response).start()
        query = response[start_sql:end_sql].strip()
        return query, True
    except:
        return response, False


def parse_generated_nlp(response):
    # """Extract the response from the model. Currently built for Claude XML 
    # tagging format
    # Parameters
    # ----------
    # response :
    #     LLM generated string
    # Returns
    # ----------
    # str :
    #     extracted string
    # """
    # logging.info(f"NLP response:\n{response}\n")
    # try:
    #     start = re.search(r'<response>', response).end()
    #     end = re.search(r'</response>', response).start()
    #     # response = response[start:end].strip()
    #     response = response[start:end] + "test"
    #     logging.info(f"Final Output:\n{response}\n")
    #     return response
    # except:
    #     return response


    logging.info(f"NLP response:\n{response}\n")
    try:
        # XML 태그가 없는 경우, 전체 응답을 반환
        if '<response>' not in response or '</response>' not in response:
            return response.strip()
        else:
            start = re.search(r'<response>', response).end()
            end = re.search(r'</response>', response).start()
            response = response[start:end].strip()
            logging.info(f"Final Output:\n{response}\n")
            return response
    except:
        return response