import json

from langchain_core.prompts import ChatPromptTemplate

from langchain.agents import tool
from langchain.agents import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent

from langchain_community.tools import DuckDuckGoSearchRun

from langchain.prompts import MessagesPlaceholder

from langchain.memory import ConversationBufferMemory

from langchain_openai import ChatOpenAI

from langchain.globals import set_debug, set_verbose
set_debug(True)

search = DuckDuckGoSearchRun()


# 現在時刻を読み取るツール
@tool
def get_date_time() -> json:
    """datetime関数をつかい、「現在時刻」「今日の日付」を返します"""
    day_now = datetime.datetime.today().strftime("%-Y年%-m月%-d日")
    time_now = datetime.datetime.now().strftime("%-H時%-M分")
    date_time_data = {
        "day_now": day_now,
        "time_now": time_now,
    }
    print(day_now, time_now)
    return json.dumps(date_time_data)

tools = [
    Tool(
        name="duckduckgo-search",
        func=search.invoke,
        description="""
            ###目的###
            必要な情報を得るためウェブ上の最新情報を検索します
            
            ###回答例###
            Q: 東京の今日の天気予報を教えて
            A: 東京都の本日の天気予報は晴れのち曇り最高気温32度最低気温25度 今日も暑くなるでしょう

            ###制限###
            回答は140文字以内でおこなってください
            """,
    ),
    get_date_time,
]

MEMORY_KEY = "chat_history"
memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)

# template = """与えられたリクエストに回答してください
# input:{input}
# 回答:"""

# prompt = ChatPromptTemplate.from_template(template)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
                            あなたは垂直方向と水平方向に移動するカメラを搭載した音声チャットロボットです。
                            名前は「ゆっくり霊夢」です。
                            """),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# model = ChatOpenAI()
model = ChatOpenAI(openai_api_key="EMPTY", openai_api_base="http://192.168.11.144:8000/v1", temperature=0) # model_name="gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613"



# chain = prompt | model

# result = chain.invoke({"input":"大谷翔平は何歳で結婚しましたか"})

# print(result)

agent = create_openai_tools_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "現在時刻を教えて"}) # 東京の今日の天気予報を教えて

print(result)

