from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field  # for grading
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import threading
from langchain_community.llms import Tongyi
import os

# key
qianfan_ak = "DAEEqjuvglLTgQMCXqRvqfUj"
qianfan_sk = "s0AJ849GNB6440lwLWDvGuNEJNrgrbQ3"
os.environ["TAVILY_API_KEY"] = "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"
os.environ["DASHSCOPE_API_KEY"] = "sk-a9af7d03e7154f5c9cc80d649e309618"

# models
llm = QianfanChatEndpoint(model="ERNIE-4.0-8K", streaming=True,
                          qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk, penalty_score=1)

llm2 = Tongyi(model_name="qwen-max")

embed = QianfanEmbeddingsEndpoint(
    model="bge_large_zh", endpoint="bge_large_zh", qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk)

grade_prompt = PromptTemplate(
    template="""    
        搜索结果相关性检查的二进制评分。\n   
        搜索结果: \n\n {result} 
        \n\n 搜索输入: {query}
        \n\n搜索结果与搜索相关，'yes' 或 'no'
        """,
    input_variables=["query", "result"]
)

retrieval_grader = grade_prompt | llm | StrOutputParser()
retrieval_grader2 = grade_prompt | llm2 | StrOutputParser()

prompt1 = PromptTemplate(
    template="""
    请考虑{prod}的特点、市场定位和用户反馈，
    详细说明该产品在市场上的独特卖点和竞争优势。
    请提供关于该产品如何吸引目标用户群体的见解。\n\n
    主要使用以下信息来得出答案: \n
    {context}""",
    input_variables=["prod", "context"]
)

prompt2 = PromptTemplate(
    template="""
    请分析使用{prod}的主要目标人群的特征、偏好和需求。
    结合季节和节日，说明这些人群在这个时间点对{prod}的需求和期望。\n\n
    主要使用以下信息来得出答案: \n
    {context}""",
    input_variables=["prod", "context"]
)

prompt3 = PromptTemplate(
    template="""
    请详细说明{prod}在{season}时间点的最佳使用方法和搭配。
    考虑到食材的季节性和营养价值，
    解释为什么在这个时间点使用{prod}最为合适。\n\n
    主要使用以下信息来得出答案: \n
    {context}""",
    input_variables=["prod", "season", "context"]
)

prompt4 = PromptTemplate(
    template="""
    请提供关于在{season}时间点使用{prod}进行促销活动或节日活动的建议。
    详细说明如何利用节日气氛和季节特点来提升{prod}的销量和品牌知名度。\n\n
    主要使用以下信息来得出答案: \n
    {context}""",
    input_variables=["prod", "season", "context"]
)


def split_text(text):
    words = text.split()
    chunks, current_chunk = [], []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= 400 and word:
            current_chunk.append(word)
        elif current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def rag_search(query):
    url = "https://www.baidu.com/s"

    search_query = {'wd': query}

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, params=search_query, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')

    results, content = [], []

    for item in soup.find_all('div', class_='result'):
        link = item.find('a', href=True)  # 'a' is a link notation
        if link:
            results.append(link['href'])

    docs = get_page(results)

    for doc in docs:
        page_text = re.sub("\n\n+", "\n", doc)

        if page_text and page_text != "问题反馈":
            content.append(page_text)

    return content


def get_page(urls):
    docs = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    for url in urls:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            page_text = "\n".join([p.get_text() for p in paragraphs])
            chunks = split_text(page_text)
            docs.extend(chunks)

    return docs


def process_search(query):

    q_embed = embed.embed_query(query)
    search_result = rag_search(query)

    if not search_result:
        return "No relevant content found."

    all_chunks = []
    for result in search_result:
        chunks = split_text(result)
        all_chunks.extend(chunks)

    if not all_chunks:
        return "No valid chunks found in the search results."

    search_embed = []
    i1, i2 = 0, 0

    while i2 < len(all_chunks) - 1:
        i2 += 1
        search_embed.extend(embed.embed_documents(all_chunks[i1:i2]))
        i1 += 1

    if not search_embed:
        return "Failed to embed search results."

    search_embed = np.array(search_embed)
    similarity_scores = np.dot(q_embed, search_embed.T)
    filtered_results = [(result, score) for result, score in zip(
        search_result, similarity_scores) if score > 0.5]

    max_ctxt = 3
    if len(filtered_results) < 3:
        max_ctxt = len(filtered_results)
    top_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[
        :max_ctxt]

    rag_results = " ".join([result[0] for result in top_results])

    print(rag_results)

    return rag_results


def re_search(query):
    promptV2 = PromptTemplate(
        template="""    
        你是一个搜索输入重写器，将搜索输入转换为优化后的版本以便进行网络搜索。\n     
        优化以下内容。只回复优化后的搜索查询：\n    
        {query}     
        """,
        input_variables=["query"]
    )

    rewriter = promptV2 | llm | StrOutputParser()

    new_query = rewriter.invoke({"query": query})

    return process_search(new_query)


def crag_search(llm, prompt, prod_des, season, keywords, add_season=None):
    query = prod_des + keywords
    if add_season:
        query = query + season

    search_result = process_search(query)

    grade_prompt = PromptTemplate(
        template="""    
        搜索结果相关性检查的二进制评分。\n   
        搜索结果: \n\n {result} 
        \n\n 搜索输入: {query}
        \n\n搜索结果与搜索相关，'yes' 或 'no'
        """,
        input_variables=["query", "result"]
    )

    retrieval_grader = grade_prompt | llm | StrOutputParser()

    result = retrieval_grader.invoke(
        {"query": query, "result": search_result})

    if result == "no":
        search_result = re_search(query)

    expertGPT = prompt | llm | StrOutputParser()

    return expertGPT.invoke(
        {"prod": prod_des, "season": season, "context": search_result})


def call_reciGPT():
    # prod_des = input("Enter product:")
    # season = input("Enter season:")

    prod_des = "伊利羊奶粉"
    season = "万圣节"

    context = []

    def call_crag(llm, prompt, prod_des, season, keywords, add_season=None):
        ctxt = crag_search(llm, prompt, prod_des, season, keywords, add_season)
        context.append(ctxt)

    t1 = threading.Thread(target=call_crag, args=(
        llm, prompt1, prod_des, season, "产品特点 市场定位 用户反馈 卖点 竞争优势"))
    t2 = threading.Thread(target=call_crag, args=(
        llm2, prompt2, prod_des, season, "目标人群 特征 偏好 需求 季节 节日"))
    t3 = threading.Thread(target=call_crag, args=(
        llm, prompt3, prod_des, season, "最佳使用方法 搭配 食材季节性 营养价值", True))
    t4 = threading.Thread(target=call_crag, args=(
        llm2, prompt4, prod_des, season, "促销活动 节日活动 提升销量 品牌知名度", True))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()

    prompt5 = PromptTemplate(
        template="""你是一位美食和烹饪大师，能够为特定的食品或食材创作出应季的菜品推荐。
        你熟悉食材的生长规律、生长习性和成分变化，能准确推荐每种食材在一年中最佳的食用月份和季节。
        你将用专业的知识和优美清新的文字，向人们解释为什么这个月是最佳食用时间。
        你的独特见解和丰富知识，将为读者带来耳目一新的体验，并为目标客户人群提供情感价值。\n\n

        请参考以下专家的建议：\n\n

        产品运营专家说：{context1}\n
        人群运营专家说：{context2}\n
        时令食材运营专家说：{context3}\n
        节点活动运营专家说：{context4}\n
        针对给定的{prod}（商品）和{season}时间点，写三道推荐菜品。
        请输出如下格式的食谱，并确保所有准备步骤（如预热烤箱等）都在指示的开始部分，以便遵循食谱的人能够从头到尾直观地操作：\n\n

        菜名\n
        对该菜品的简短描述，包含亮点，如何最佳利用该食材，以及如何适应这个季节。\n\n

        食材\n
        总时间（准备时间，烹饪时间）\n\n

        准备步骤\n
        步骤1\n
        步骤2\n
        等等\n
        烹饪步骤\n
        步骤1\n
        步骤2\n
        等等\n\n
        享用！""",
        input_variables=["prod", "context1", "context2",
                         "context3", "context4", "season"]
    )

    reciGPT = prompt5 | llm | StrOutputParser()

    try:
        ans = reciGPT.invoke(
            {"prod": prod_des, "context1": context[0], "context2": context[1],
             "context3": context[2], "context4": context[3], "season": season})
    except Exception as e:
        print("error: ", e)
        return

    return ans


print(call_reciGPT())
