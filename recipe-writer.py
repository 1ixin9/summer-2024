from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field  # for grading
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.output_parsers import StrOutputParser
import numpy as np
from product_analysis import split_text, rag_search

# key
qianfan_ak = "DAEEqjuvglLTgQMCXqRvqfUj"
qianfan_sk = "s0AJ849GNB6440lwLWDvGuNEJNrgrbQ3"

# models
llm = QianfanChatEndpoint(model="ERNIE-4.0-8K", streaming=True,
                          qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk, penalty_score=1)
embed = QianfanEmbeddingsEndpoint(
    model="bge_large_zh", endpoint="bge_large_zh", qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk)


def make_prompts():

    prompts = {}

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

    prompts[prompt1] = "产品特点 市场定位 用户反馈 卖点 竞争优势"
    prompts[prompt2] = "目标人群 特征 偏好 需求 季节 节日"
    prompts[prompt3] = "最佳使用方法 搭配 食材季节性 营养价值"
    prompts[prompt4] = "促销活动 节日活动 提升销量 品牌知名度"

    return prompts


class GradeDocuments(BaseModel):
    """搜索结果相关性检查的二进制评分。"""

    binary_score: str = Field(
        description="搜索结果与搜索相关，'yes' 或 'no'"
    )


def process_search(query):

    q_embed = embed.embed_query(query)

    search_result = rag_search(query)

    all_chunks = []  # makes a list to store the chunks
    for result in search_result:  # for every single result we get
        chunks = split_text(result)
        all_chunks.extend(chunks)

    search_embed = []  # makes a list for the embeddings bc the embedding model has a max num of tokens that is exceeded by big chunks
    i1, i2 = 0, 0

    while i2 < len(all_chunks) - 1:
        i2 += 1
        search_embed.extend(embed.embed_documents(all_chunks[i1:i2]))
        i1 += 1

    search_embed = np.array(search_embed)

    # this will get all the dot products for our searches X context
    similarity_scores = np.dot(q_embed, search_embed.T)
    filtered_results = [(result, score) for result, score in zip(
        search_result, similarity_scores) if score > 0.5]

    max_ctxt = 3
    if len(filtered_results) < 3:
        max_ctxt = len(filtered_results)
    top_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[
        :max_ctxt]

    rag_results = " ".join([result[0] for result in top_results])

    return rag_results


def re_search(query):

    promptV2 = PromptTemplate(

        template="""    
        你是一个搜索输入重写器，将搜索输入转换为优化后的版本以便进行网络搜索。\n     
        优化以下内容：\n    
        {query}     
        """,
        input_variables=["query"]

    )

    rewriter = promptV2 | llm | StrOutputParser()

    new_query = rewriter.invoke({"query": query})

    return process_search(new_query)


def crag_search(prod_des, season):

    prompts = make_prompts()
    context, queries, search_results = [], [], []

    for prompt in prompts.keys():
        query = prod_des + prompts[prompt]
        queries.append(query)
        text = process_search(query)
        search_results.append(text)

    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    grade_prompt = PromptTemplate(
        template="""       
        搜索结果: \n\n {result} 
        \n\n 搜索输入: {query}
        """,
        input_variables=["query", "result"]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    grades = [True, True, True, True]

    for i, query in enumerate(queries):
        result = retrieval_grader.invoke(
            {"query": query, "result": search_results[i]})
        bs = result.binary_score

        if bs == "no":
            grades[i] = False

    if not grades[0] or not grades[1] or not grades[2] or not grades[3]:

        if not grades[0]:
            query = prod_des + prompts[0][1]
            text = re_search(query)
            search_results[0] = text

        if not grades[1]:
            query = prod_des + prompts[1][1]
            text = re_search(query)
            search_results[1] = text

        if not grades[2]:
            query = prod_des + prompts[2][1]
            text = re_search(query)
            search_results[2] = text

        if not grades[3]:
            query = prod_des + prompts[3][1]
            text = re_search(query)
            search_results[3] = text

    for i, prompt in enumerate(prompts):
        expertGPT = prompt[0] | llm | StrOutputParser()

        ans = expertGPT.invoke(
            {"prod": prod_des, "season": season, "context": search_results[i]})

        context.append(ans)

    return context


def call_reciGPT():

    prod_des = input("Enter product:")
    season = input("Enter season:")

    context = crag_search(prod_des, season)

    prompt5 = PromptTemplate(
        template="""你是一个美食大师、厨艺大师，你可以对指定的食品或食材，
        创作出当季的菜品推荐，你熟悉食材的生长规律、生长习性、成分变化，你极度推荐每个食材在一年中最佳食用的月份、
        季节，你用你的专业性和优美清新的文字功底，向人们解释为什么这个月食用最佳；
        你与众不同、独到的专业见解和知识面，让读者耳目一新，能提供目标客户人群的情绪价值。\n\n
        请考虑以下专家提供的建议。\n
        产品运营专家说：{context1}。\n
        人群运营专家说：{context2}。\n
        时令食材运营专家说：{context3}。\n
        节点活动运营专家说：{context4}。\n\n
        针对给定的{prod}（商品）和{season}时间点，写三道推荐菜品:""",
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


call_reciGPT()
