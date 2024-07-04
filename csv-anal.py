from langchain_community.chat_models import QianfanChatEndpoint
from langchain.embeddings import QianfanEmbeddingsEndpoint
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import re
import pandas as pd

# key
qianfan_ak = ""
qianfan_sk = ""

# models
llm = QianfanChatEndpoint(model="ERNIE-4.0-8K", streaming=True,
                          qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk, penalty_score=1)
embed = QianfanEmbeddingsEndpoint(
    model="bge_large_zh", endpoint="bge_large_zh", qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk)

query = "SUNTORY 三得利乌龙茶"


def csv2list(csv=None):

    if not csv:
        csv = '/data/ai/product_analysis_wlx/product_analysis/product_search_content.cvs'

    df = pd.read_csv(csv)
    column_data = df['url_content'].tolist()

    return column_data


def process_scraped_content(content, chunk_size=96, output_file="processed_content.txt"):
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'\s+', ' ', content)
    content = content.strip()
    chunks = [content[i:i+chunk_size]
              for i in range(0, len(content), chunk_size)]

    with open(output_file, 'w') as f:
        for chunk in chunks:
            f.write(chunk + '\n\n')

    return chunks


def embed_chunks(content, query):
    content_chunks = process_scraped_content(content)
    q_embed = embed.embed_query(query)

    chunk_embeddings = embed.embed_documents(content_chunks)

    chunk_embeddings = np.array(chunk_embeddings)

    similarity_scores = np.dot(chunk_embeddings, q_embed) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_embed))

    filtered_results = [(chunk, score)
                        for chunk, score in zip(content_chunks, similarity_scores)]

    top_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

    tlen = int(len(top_results))

    top_results_first_elements = [
        result for result, score in top_results[:tlen] if score > .8]

    return top_results_first_elements


def do_embedding(query, csv=None):

    embeddings = csv2list(csv)
    listlist = []

    for i in range(len(embeddings)):
        info = embed_chunks(embeddings[i], query)
        if info and len(info) > 4:
            listlist.append(info)

    return listlist


def llm_process(top_results, query):

    flattened_list = [item for sublist in top_results for item in sublist]

    flattened_list = sorted(flattened_list, key=lambda x: x[1], reverse=True)
    flattened_list = flattened_list[:40]

    prompt_selling_points = PromptTemplate(
        template="""\
		{product}产品相关信息如下：
		{search_info}

		以上是根据{product}产品名搜索爬取到的网页内容，可能包含大量无关的信息，请只考虑和{product}相关的信息，请根据以上信息，总结出产品的核心卖点和二级卖点。

		# Role : 电商商品卖点总结专家
		## Background :
		您作为电商行业的运营人员，需要对商品进行精准的卖点提炼，以提升商品的吸引力和市场竞争力。商品卖点分为多个类别，但并非所有卖点都适用于每一件商品。

		商品卖点大类包含以下内容，每个卖点下的条数和描述可根据实际情况进行扩展：
		1. 品质卖点：
		a. 使用的材料：如天然、有机、环保材料。
		b. 制造工艺：如手工制作、精密工艺。
		2. 设计卖点：
		a. 外观设计：如时尚、简约、复古等风格。
		b. 功能设计：如人体工程学设计、易于使用。
		3. 功能性卖点：
		a. 技术特点：如智能化、自动化功能。
		b. 性能指标：如速度、容量、耐用性。
		4. 价格卖点：
		a. 价格优势：如性价比高、折扣、促销活动。
		b. 价格定位：如高端、中端、经济型。
		5. 品牌卖点：
		a. 品牌历史：如历史悠久、品牌故事。
		b. 品牌影响力：如市场占有率、用户评价。
		6. 服务卖点：
		a. 客户服务：如24小时客服、专业咨询。
		b. 售后保障：如长保修期、无忧退换。
		7. 便捷性卖点：
		a. 购买流程：如一键购买、快速结账。
		b. 物流服务：如快速配送、全球配送。
		8. 个性化卖点：
		a. 定制服务：如个性化定制、刻字服务。
		b. 用户体验：如个性化推荐、用户友好界面。
		9. 社会责任卖点：
		a. 社会贡献：如慈善捐赠、社区支持。
		b. 环保理念：如使用可持续材料、减少碳足迹。
		10. 技术卖点：
		a. 创新技术：如最新科研成果、专利技术。
		b. 技术领先：如行业内技术领先、专业认证。
		11. 健康卖点：
		a. 健康益处：如有助于健康、无添加剂。
		b. 安全标准：如符合食品安全标准、无有害物质。
		12. 情感卖点：
		a. 情感联系：如礼物推荐、节日特供。
		b. 情感体验：如使用愉悦感、满足感。

		## Constraints :
		1. 对于单个商品，需从提供的卖点大类中选择所有适用的卖点。
		2. 在已选择的卖点中，总结出核心卖点，要求不超过10个字。
		3. 在已选择的卖点中，根据不同商品，总结出1到4条不等二级卖点。
		4. 二级卖点每条不超过4个字，或每条不超过8个字，保持形式一致。

		## Goals :
		1. 为单个商品选择适用的卖点。
		2. 提炼核心卖点，简明扼要地展示商品最大亮点。
		3. 撰写二级卖点，详细描述商品的其他优势。

		## Skills :
		1. 商品卖点分析能力。
		2. 文案撰写与优化技巧。
		3. 快速准确的信息提炼能力。

		## Workflows :
		1. 根据商品特性，从提供的卖点分类中选择适用的卖点。
		2. 从已选卖点中提炼出核心卖点，简洁明了地表达商品的最大卖点。
		3. 编写二级卖点，突出商品的其他重要优势，保持简洁。
		
  		请按照上述格式提供响应：
		""",
        input_variables=["product", "search_info"]
    )

    prompt_product_desc = PromptTemplate(
        template="""\
	{product}产品相关信息如下：
	{search_info}

	以上是根据{product}产品名搜索爬取到的网页内容，可能包含大量无关的信息，请只考虑和{product}相关的信息，请根据以上信息，\
	提取出{product}的商品描述，描述应该简洁准确，不要包含其他多余的内容。

	# Role : 电商商品描述撰写专家
	## Background :
	您作为电商行业的运营人员，需要对商品进行精准的描述，以提升商品的吸引力和市场竞争力。商品描述应简洁、准确、引人入胜，并能充分展示商品的特点和优势。

	商品描述大类包含以下内容，每个类别可根据实际情况进行扩展：
	1. 基本信息：
	a. 商品名称：如品牌名、产品型号。
	b. 商品类别：如电子产品、家居用品。
	2. 规格参数：
	a. 尺寸和重量：如产品的具体尺寸、重量。
	b. 材质和成分：如使用的材料、成分。
	3. 主要功能：
	a. 功能特点：如多功能、智能化。
	b. 使用场景：如家庭、办公室、户外。
	4. 外观设计：
	a. 设计风格：如现代、简约。
	b. 颜色选择：如多种颜色可选。
	5. 使用体验：
	a. 操作简便：如易于安装、操作简单。
	b. 用户评价：如用户好评、推荐指数。
	6. 售后服务：
	a. 保修服务：如提供保修、维修服务。
	b. 退换政策：如无理由退换、快速退款。
	7. 价格信息：
	a. 定价策略：如高性价比、折扣信息。
	b. 促销活动：如限时优惠、赠品。
	8. 品牌信息：
	a. 品牌历史：如品牌创立时间、发展历程。
	b. 品牌声誉：如市场口碑、用户认可。

	## Constraints :
	1. 对于单个商品，需从提供的描述大类中选择所有适用的类别。
	2. 在已选择的类别中，确保描述简洁准确，不超过100字。
	3. 描述内容需真实可信，不夸大其词。

	## Goals :
	1. 为单个商品选择适用的描述类别。
	2. 撰写简洁准确的商品描述，展示商品的核心特点和优势。
	3. 确保描述内容真实可靠，提升商品的市场吸引力。

	## Skills :
	1. 商品信息分析能力。
	2. 文案撰写与优化技巧。
	3. 快速准确的信息提炼能力。

	## Workflows :
	1. 根据商品特性，从提供的描述分类中选择适用的类别。
	2. 撰写简洁准确的商品描述，简明扼要地展示商品的核心特点。
	3. 确保描述内容真实可靠，提升商品的市场吸引力。
	""",
        input_variables=["product", "search_info"]
    )

    pdGPT = prompt_selling_points | llm | StrOutputParser()

    ans = pdGPT.invoke({"product": query, "search_info": flattened_list})

    pdGPT2 = prompt_product_desc | llm | StrOutputParser()

    ans2 = pdGPT2.invoke({"product": query, "search_info": flattened_list})

    return ans, ans2


tr = (do_embedding(query))

answer = llm_process(tr, query)

print(answer[0], answer[1])
