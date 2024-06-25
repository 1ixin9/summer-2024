from langchain.prompts import PromptTemplate
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from IPython.display import display
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from PIL import Image
import pytesseract

# key
qianfan_ak = "DAEEqjuvglLTgQMCXqRvqfUj"
qianfan_sk = "s0AJ849GNB6440lwLWDvGuNEJNrgrbQ3"

# models
llm = QianfanChatEndpoint(model="ERNIE-4.0-8K", streaming=True,
                          qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk, penalty_score=1)
embed = QianfanEmbeddingsEndpoint(
    model="bge_large_zh", endpoint="bge_large_zh", qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk)


def split_text(text):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:

        if len(" ".join(current_chunk + [word])) <= 400 and word:
            current_chunk.append(word)
        elif current_chunk:

            chunks.append(" ".join(current_chunk))

            current_chunk = [word]

    if current_chunk:

        chunks.append(" ".join(current_chunk))

    return chunks


def turn2key(prod_name):

    prompt1 = PromptTemplate(

        template="""
        你将获得一个中文产品名称。你的任务是将该中文产品名称转换为适合包含在URL查询参数中的URL编码字符串。
        URL编码字符串必须与1688.com使用的基本URL格式兼容。以下是你需要按照的步骤：\n\n

        将每个中文产品名称转换为相应的URL编码格式。\n
        确保输出的URL编码字符串可以与以下基本URL连接起来：https://s.1688.com/selloffer/offer_search.htm?keywords=\n
        为每个中文产品提供完整的URL。\n\n

        以下是一些关键字翻译的示例：\n\n

        中文产品名称：罐装红牛\n
        URL编码：%B9%DE%D7%B0%BA%EC%C5%A3\n
        完整URL：https://s.1688.com/selloffer/offer_search.htm?keywords=%B9%DE%D7%B0%BA%EC%C5%A3\n\n

        中文产品名称：三得利乌龙茶\n
        URL编码：%C8%FD%B5%C3%C0%FB%CE%DA%C1%FA%B2%E8\n
        完整URL：https://s.1688.com/selloffer/offer_search.htm?keywords=%C8%FD%B5%C3%C0%FB%CE%DA%C1%FA%B2%E8\n\n

        中文产品名称：蜜桃乌龙\n
        URL编码：%C3%DB%CC%D2%CE%DA%C1%FA\n
        完整URL：https://s.1688.com/selloffer/offer_search.htm?keywords=%C3%DB%CC%D2%CE%DA%C1%FA\n\n

        中文产品名称：菊花茶\n
        URL编码：%BE%D5%BB%A8%B2%E8\n
        完整URL：https://s.1688.com/selloffer/offer_search.htm?keywords=%BE%D5%BB%A8%B2%E8\n\n

        中文产品名称：植物饮料\n
        URL编码：%D6%B2%CE%EF%D2%FB%C1%CF\n
        完整URL：https://s.1688.com/selloffer/offer_search.htm?keywords=%D6%B2%CE%EF%D2%FB%C1%CF\n\n

        请按照这些步骤进行，并确保每个中文产品的URL编码的准确性。产品名称是：{prod}。请仅提供完整的URL:\n
        
        """,

        input_variables=["prod"]
    )

    transGPT = prompt1 | llm | StrOutputParser()

    key = transGPT.invoke({"prod": prod_name})

    return key


def tb_search(query):

    url = "https://s.1688.com/selloffer/offer_search.htm?keywords="

    query_key = turn2key(query)
    query_embed = embed.embed_query(query)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, params=query_key, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')

    key_prod = None

    for item in soup.find_all('div', class_='mojar-element-title'):
        title = item.find('div', class_='title')
        title_embed = embed.embed_query(title)
        similarity_scores = np.dot(query_embed, title_embed)
        if similarity_scores > .85:
            key_prod = item.find('a', href=True)
            break

    pic = get_page(key_prod)

    return image_process(pic)


def get_page(url):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:

        soup = BeautifulSoup(response.content, 'html.parser')

        pic = soup.find('img', class_='preview-img')

    return pic


def rag_search(query):

    return tb_search(query)


def image_process(filename):
    img = np.array(Image.open(filename))
    text = pytesseract.image_to_string(img)
    return text


def call_searchGPT():

    prod_name = "植物饮料"
    print(rag_search(prod_name))


call_searchGPT()
