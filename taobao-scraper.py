from langchain.prompts import PromptTemplate
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import requests
import time
import random
import os
from bs4 import BeautifulSoup
from urllib.parse import urlencode
from PIL import Image

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

service = ChromeService(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

pics = {}


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


def baidu_search(query):

    base_url = "https://www.baidu.com/s"

    search_query = urlencode({'wd': f"1688 {query}"})

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(f"{base_url}?{search_query}", headers=headers)

    if response.status_code == 200:

        soup = BeautifulSoup(response.text, 'html.parser')

        key_pages = []

        for item in soup.find_all('div', class_='result'):

            link = item.find('a', href=True)

            if link and 'href' in link.attrs:

                href = link['href']
                url = get_acc_url(href)
                key_pages.append(url)

                if len(key_pages) > 3:
                    break

    else:
        print("couldn't perform search")

    for page in key_pages:
        pics[page] = get_page(page)
        if pics[page] == [] or None:
            del pics[page]

    return pics


def get_acc_url(url):
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = session.get(
            url, headers=headers, allow_redirects=True)

        if response.history:
            actual_url = response.url
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            meta_refresh = soup.find('meta', attrs={'http-equiv': 'refresh'})
            if meta_refresh:
                content = meta_refresh.get('content')
                url_start = content.find('url=') + 4
                actual_url = content[url_start:]
            else:
                actual_url = url

        return actual_url

    except Exception as e:
        print(f"Error: {e}")
        return None


def get_page(url):

    driver.get(url)
    time.sleep(2)

    html = driver.page_source

    soup = BeautifulSoup(html, 'html.parser')

    unique_links = {}

    for a_tag in soup.find_all('a', href=True):
        link = a_tag['href']
        print(link)
        if link not in unique_links and prod_link(link):
            unique_links[link] = True

    links = list(unique_links.keys())

    pngs = []

    for page in links:
        if page[:6] == "https:":
            pngs.append(process_pics(page))
        else:
            pngs.append(process_pics("https:"+page[2:]))

    return pngs


def process_pics(url):

    driver.get(url)
    time.sleep(2)

    pic_name = str(random.randint(1, 9152003)) + '.png'

    total_height = driver.execute_script(
        "return document.body.parentNode.scrollHeight")
    total_width = driver.execute_script("return document.body.scrollWidth")
    driver.set_window_size(total_width, total_height)

    driver.save_screenshot(pic_name)
    image = Image.open(pic_name)
    pic = os.path.join('pics', pic_name)
    image.save(pic)

    return pic_name


def prod_link(url):
    if 'dj.1688.com/ci_bb?a=' in url or 'detail.1688.com/offer' in url:
        return True
    return False


# key
qianfan_ak = ""
qianfan_sk = ""

# models
llm = QianfanChatEndpoint(model="ERNIE-4.0-8K", streaming=True,
                          qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk, penalty_score=1)
embed = QianfanEmbeddingsEndpoint(
    model="bge_large_zh", endpoint="bge_large_zh", qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk)


def rag_search(query):
    return baidu_search(query)


def verify_info(ctxt, q):

    # ask the LLM if the stuff we've collected is sufficient
    prompt1 = PromptTemplate(

        template="""根据以下上下文：{context}，
        评估这些信息是否足以回应以下内容：{query}，'yes' 或 'no'。
        请深入分析，并给出是或否的回答。""",

        input_variables=["context", "query"]
    )

    marketinGPT = prompt1 | llm | StrOutputParser()

    ans = marketinGPT.invoke({"context": ctxt, "query": q})

    if ans == 'yes':
        return True

    return False


def call_marketinGPT():

    file_path = "products.txt"

    with open(file_path, 'r', encoding='utf-8') as file:
        prod_descr = file.readlines()

    prod_descr = [desc.strip() for desc in prod_descr]

    # ask the LLM if the stuff we've collected is sufficient
    prompt1 = PromptTemplate(

        template="""根据以下上下文：{context}，
        评估这些信息是否足以回应以下内容：{query}。
        请深入分析，并给出是或否的回答。""",

        input_variables=["context", "query"]
    )

    prompt2 = PromptTemplate(

        template="""作为一名零售顾问助手，你的任务是帮助用户分析他们的产品描述，
        并提供该产品的卖点、最佳营销卖点、目标受众以及针对目标受众的营销策略。
        请根据以下格式进行回复，并且仅根据用户提供的信息进行分析和回答：\n\n
            1. 产品描述：用户提供的产品详细信息。\n
            2. 产品卖点：根据产品描述，提炼出吸引潜在消费者的关键特点。\n
            3. 最佳营销卖点：从产品卖点中选择最具市场潜力的特点，并解释为何这个卖点最有吸引力。\n
            4. 目标受众：根据产品卖点，确定最适合的消费群体。\n\n
            
        以下是一个示例对话：\n
        
        用户：我们有一款新型的可折叠电动自行车，重量轻，电池续航长，适合城市通勤。\n\n
        系统：\n
            1. 产品描述：新型可折叠电动自行车，重量轻，电池续航长，适合城市通勤。\n
            2. 产品卖点：轻便设计、长续航电池、便捷的城市通勤工具。\n
            3. 最佳营销卖点：长续航电池，因为城市通勤用户对续航时间有较高需求，能够减少充电频率。\n
            4. 目标受众：城市白领、大学生、注重环保和便捷出行的用户。\n\n
            
        请提供您的产品描述：\n
        
        {prod}\n\n

        1. 产品描述：用户提供的产品描述\n
        2. 产品卖点：提炼出的产品卖点\n
        3. 最佳营销卖点：选择的最佳营销卖点及其原因\n
        4. 目标受众：确定的目标消费群体\n
        
        请注意，产品卖点部分应当是一个完整的句子，不要使用任何形式的项目符号或列表，以免造成文本格式的混乱。\n

        主要使用以下信息来得出答案。即使以下信息提到其他具体产品，也要专注于该产品的优点，不提及其他产品的名称。：\n\n
        
        {context}""",

        input_variables=["prod", "context"]
    )

    for prod_des in prod_descr:

        marketinGPT = prompt1 | llm | StrOutputParser()

        ctxt = marketinGPT.invoke({"prod": prod_des})

        ctxt_embed = embed.embed_query(ctxt)

        search_results = rag_search(ctxt)

        all_chunks = []  # makes a list to store the chunks
        for result in search_results:  # for every single result we get

            chunks = split_text(result)

            all_chunks.extend(chunks)

        search_embed = []  # makes a list for the embeddings bc the embedding model has a max num of tokens that is exceeded by big chunks
        i1 = 0  # initializes two counters
        i2 = 0
        while i2 < len(all_chunks) - 1:  # while loop for until all chunks r processed
            i2 += 1

            search_embed.extend(embed.embed_documents(all_chunks[i1:i2]))
            i1 += 1

        search_embed = np.array(search_embed)

        similarity_scores = np.dot(ctxt_embed, search_embed.T)
        filtered_results = [(result, score) for result, score in zip(
            search_results, similarity_scores) if score > 0.5]

        max_ctxt = 3  # sets the max num of ctxt to 3 to not overload the model
        if len(filtered_results) < 3:  # if there r less than 3 pieces of usable context
            max_ctxt = len(filtered_results)  # then we use all of them
        top_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[
            :max_ctxt]  # this sorts the searches for highest matches

        rag_results = " ".join([result[0] for result in top_results])

        marketinGPT = prompt2 | llm | StrOutputParser()

        try:
            ans = marketinGPT.invoke(
                {"prod": prod_des, "context": rag_results})

        except Exception as e:
            continue

    return ans


baidu_search("三得利乌龙茶")
