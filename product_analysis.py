from langchain.prompts import PromptTemplate # for creating the template we feed to llm
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.embeddings import QianfanEmbeddingsEndpoint
# ^ for getting the actual GPT llm and also the embeddor
from langchain_core.output_parsers import StrOutputParser # for converting llm output to something Python understands
import pandas as pd # for making dfs
from IPython.display import display # for displaying the df with styling
import numpy as np # for doing dot product
import requests # for making HTTP request in Python to handle headers, cookies, and authentication
from bs4 import BeautifulSoup # for parsing HTML and XML docs so we can get j the urls / body text directly from a webpage
import re # for using regex funcs like sub and shit

# function to split text since the max length keeps being exceeded
def split_text(text):
    words = text.split() # splits the input string into a list of individual words
    chunks = [] # stores final list of chunks
    current_chunk = [] # temp list for each chunk

    for word in words: # for each word
        if len(" ".join(current_chunk + [word])) <= 400 and word: # if adding this word doesn't exceed 400 words
            current_chunk.append(word) # then it is added
        elif current_chunk:
            chunks.append(" ".join(current_chunk)) # else the current chunk is added to the chunks list as a single string
            current_chunk = [word] # then current_chunk is refreshed into a new chunk

    if current_chunk:
        chunks.append(" ".join(current_chunk)) # then at the end if there's a current chunk left then that's added to the list

    return chunks # returns a list of chunks of max length 400

# does web scraping and text searching to search Baidu -> get search urls -> fetch content from urls -> process text
def baidu_search(query): # query being the context we got
    # defines url for baidu search
    url = "https://www.baidu.com/s" # baidu (DUH)
    
    search_query = {'wd': query} # wd stands for word/query which is what we're searching (the context we're taking in)
    
    headers = { # headers r necessary bc when we as humans make searches we hv these; we need this to mimic a regular web browser's request
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    } # the User-Agent is a commonly used string for web scraping to avoid detection and blocking by webs
    
    # sends the get request to Baidu w our search params and header to do the search for relevant webpages
    response = requests.get(url, params=search_query, headers=headers)
    
    # we need to initialize a Beautiful Soup object to actually parse the HTML so we can just get the urls
    soup = BeautifulSoup(response.text, 'html.parser')

    # making a list to store urls of search results
    results = []
    # so a webpage will be separated in a bunch of divs as you know, and each div has a class that its associated w
    # so when we do soup.find_all('div', class_='result'), we r using the soup object to search across all 'div's for the result class
    # and then we r doing a for each loop of all the result divs and finding the link portion of it
    for item in soup.find_all('div', class_='result'):
        link = item.find('a', href=True) # 'a' is a link notation
        if link:
            results.append(link['href']) # and if it's a link then we add it into the list w the 'href' attribute which is a url

    # now we are making another list based on our results list that takes all the urls there and calls get_page on it to get the acc page
    docs = get_page(results) # so this stores a bunch of pages

    # making another list to store the acc content of the pages in
    content = []
    for doc in docs: # for each doc in the docs list
        page_text = re.sub("\n\n+", "\n", doc) # this regex part is p cool, sub() obviously substitutes something w another thing
        # and in this case we r subbing all the \n\n+ w j one \n which means that anything more than 1 newline will be replaced w a newline
        # so we can just get all the text in a braindead way like this
        if page_text and page_text != "问题反馈": # checks to see if empty string or bad text
            # ^ note that in Python, an empty string of "" returns False
            # page_text = cutoff(page_text)
            content.append(page_text) # and we r adding it to the list
        
    # i = 0
    # for ctnt in content:
    #     i += 1
    #     print("entry ", i,": ",ctnt)

    return content # and now we r just returning a list of all the text from relevant pages

# this is the function that we called in baidu_search to acc get the webpages from the urls
def get_page(urls):
    
    docs = [] # we r making to store the page contents
    # similar process w headers as above
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    # a for each loop of all the urls
    for url in urls:
        
        response = requests.get(url, headers=headers) # this time we r j popping into all the pages
        
        if response.status_code == 200: # status code 200 means that it was successful so if we were let in, then we call beautiful soup
            
            soup = BeautifulSoup(response.content, 'html.parser') # we initiate the soup object again
            # ^ we get the content from the response (the response being the page that we popped into)
            # ^ then we specify we wanna use 'html.parser' to get the info
            paragraphs = soup.find_all('p') # 'p' = praragraph so now we r making a list of all the paragraphs from the parsed html
            # ^ the issue is what if there are text content thats important thats not marked as 'p' or what if 'p' isn't right?
            page_text = "\n".join([p.get_text() for p in paragraphs]) # and we extract all the text from the paragraphs & join them
            chunks = split_text(page_text) # so that it doesn't keep causing errors
            docs.extend(chunks) # and now we add the text into our list of page content
    # so finally we can return a list w all the page contents of j all the text on the page
    return docs

# this is just a function to cut off excess words cuz there's a max that can be embedded
# def cutoff(text):
#     words = text.split() # this splits all the text into a list of individual word strings
#     truncated = " ".join(words[:742]) # and now we r just joining the words tgt (2000 is max)
#     return truncated

# key
qianfan_ak = "DAEEqjuvglLTgQMCXqRvqfUj"
qianfan_sk = "s0AJ849GNB6440lwLWDvGuNEJNrgrbQ3"

# models
llm = QianfanChatEndpoint(model="ERNIE-4.0-8K", streaming=True, qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk, penalty_score=1)
embed = QianfanEmbeddingsEndpoint(model="bge_large_zh", endpoint="bge_large_zh", qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk)

# does the rag search and returns a list (of strings) with all the info from the first few pages of web links (each link's info is concatenated tgt)
def rag_search(query):
    return baidu_search(query) # rn we hv them as diff functions in case baidu doesn't work out

def df_mk(ar1, ar2, ar3, ar4):
    df = pd.DataFrame({ # this is the syntax for making a df which is basically a table in pandas
        "产品描述": ar1, # the quotes has the title of the column
        "产品卖点": ar2, # u can either make an array like [val, val] urself
        "最佳营销卖点": ar3, # or u can use an array that u alr made
        "目标受众": ar4 # and put it into the df like that
    })
        
    return df

def parse_response(llm_output): # this parses the output param that is generated by the llm to find the info we need for the df
        
    # split the output into lines
    lines = llm_output.split('\n') # split function is a python func that turns a string into a list
    # ^ where each newline from the og where the items r separated
    
    # initialize placeholders
    product_description = 'missing description' # this is for debugging in case we find something isn't generated properly
    selling_points = 'missing description' # then we know that whatever has "missing description" didn't have that part
    best_marketing_point = 'missing description'
    target_audience = 'missing description'
        
    # iterate over lines and find the relevant sections
    for line in lines: # for each loop that searches each line in the output for the content we need
        if line.startswith("1. 产品描述："): # so if the line starts with [this] then we store this line into the corresponding var
            product_description = line[8:].strip() # but we strip the title for cleanliness
        elif line.startswith("2. 产品卖点："):
            selling_points = line[8:].strip()
        elif line.startswith("3. 最佳营销卖点："):
            best_marketing_point = line[11:].strip()
        elif line.startswith("4. 目标受众："):
            target_audience = line[8:].strip()
                
    return product_description, selling_points, best_marketing_point, target_audience # and we just return the variables


def call_marketinGPT():

    file_path = "products.txt" # this is the file u wanna open
    # if it was in a diff path, then u would hv to do ../folder/folder/file.txt instead
    
    with open(file_path, 'r', encoding='utf-8') as file: # 'r' means read (DUH), utf-8 encoding is standard
        prod_descr = file.readlines() # "with" makes sure it's closed at the end
    # "as file" basically sets what's opened into the variable "file"
    # prod_descr is a list type var that stores everything that is read from the file
    # readlines reads all the individual lines (broken apart by \n) into the list
    
    prod_descr = [desc.strip() for desc in prod_descr] # prod_descr prior to this would be like ["prod_des1\n", "prod_des2\n"]
    # desc is each individual line stored in prod_descr, using desc.strip for each desc gets rid of \n at the end of each one
    # now, prod_descr is like ["prod_des1", "prod_des2"]
    
    prompt1 = PromptTemplate(
        # template is the prompt that ur using to prompt engineer the GPT
                
        template = """输入一个产品名称后，生成一段简短描述，涵盖其主要卖点、特点和优势。\n\n

        输入格式：\n
        [{prod}]\n\n

        输出格式：\n
        [简短描述，包括卖点、特点和优势]\n""",
        
        input_variables = ["prod"] # here ur telling the gpt that the input variables it uses will be
        # used where {prod} is used in the template
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
        
        input_variables = ["prod", "context"] # here ur telling the gpt that the input variables it uses will be
    )
    
    ar1, ar2, ar3, ar4 = [], [], [], [] # here ur declaring the arrays that the df will be made w
    
    for prod_des in prod_descr: # this is a for-each loop which ensures that each val in the list is used
        
        marketinGPT = prompt1 | llm | StrOutputParser()
        
        ctxt = marketinGPT.invoke({"prod": prod_des})
        
        # do RAG search and embed the context and search results
        ctxt_embed = embed.embed_query(ctxt) # uses the langchain qianfan embedder
        search_results = rag_search(ctxt) # calls the RAG search func to get the list of search results for the context we r looking for
        
        all_chunks = [] # makes a list to store the chunks
        for result in search_results: # for every single result we get
            chunks = split_text(result) # we split the result into a list of chunks
            all_chunks.extend(chunks) # and adds all the chunks to the list separately
            # ^ that's the diff between extend (which adds all the chunks separately) vs append (which would turn it into a list of lists of chunks)
        
        search_embed = [] # makes a list for the embeddings bc the embedding model has a max num of tokens that is exceeded by big chunks
        i1 = 0 # initializes two counters
        i2 = 0
        while i2 < len(all_chunks) - 1: # while loop for until all chunks r processed
            i2 += 1
            search_embed.extend(embed.embed_documents(all_chunks[i1:i2])) # then embeds the search results chunk by chunk
            i1 += 1
        # ^ perhaps not the most efficient so potentially REWORK but effective rn
        
        # calculate dot product and get closest results
        search_embed = np.array(search_embed)
        
        similarity_scores = np.dot(ctxt_embed, search_embed.T) # this will get all the dot products for our searches X context
        filtered_results = [(result, score) for result, score in zip(search_results, similarity_scores) if score > 0.5]
        # ^ gets rid of any useless unrelated results
        max_ctxt = 3 # sets the max num of ctxt to 3 to not overload the model
        if len(filtered_results) < 3: # if there r less than 3 pieces of usable context
            max_ctxt = len(filtered_results) # then we use all of them  
        top_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:max_ctxt] # this sorts the searches for highest matches
        # previously used the zip() function, which combines two lists into a list of tuples made up of the results and their scores
        # sorted() sorts the function (DUH) based on "key=lambda x: x[1]" which means that it sorts on the second element (the scores)
        # reverse=True means that it's sorted in descending order rather than ascending so top scores r on the top
        # [:4] is the slice notation for getting the first up to :"x" index, so this would be the first 4 items
        
        rag_results = " ".join([result[0] for result in top_results]) # turns the top 3 results into one string to be fed into prompt
                
        marketinGPT = prompt2 | llm | StrOutputParser() # this is the setup for the processing pipeline
        # prompt refers to the template ur using to prompt engineer -> this is given to llm
        # llm then takes the text input and generates a response
        # StrOutputParser is an output parser (DUH but also an output parser takes raw output and turns it into a structured format)
        # ^ this turns the llm's output into something Python can easily understand
        # using the '|' operator is basically the chaining part of the processing pipeline
        # ^ this says the output of one component should be used as the input of the next component
        # ^ so the prompt's output is the llm's input, the llm's output is the parser's input
        
        try: # we using a try-except bc who knows if the GPT will output something that is always understandable
                        
            ans = marketinGPT.invoke({"prod": prod_des, "context": rag_results}) # marketinGPT (brilliant name) is the name of the pipeline
            # so when we call it, we r getting an instance of it
            # .invoke(input val) is a method that tells the model to provide a response based on the input val
            # "prod" is the input variable in the prompt, prod_des is the value in the for-each loop
            # this lets prod_des be passed in as the input of the prompt
            # ^ same for rag_result
            
            parsed_response = parse_response(ans) # here we use parse_response to parse the response (DUH)
            
            ar1.append(parsed_response[0]) # here we r appending (adding) the answer to the arrays
            ar2.append(parsed_response[1]) # hopefully everything is right
            ar3.append(parsed_response[2]) # but otherwise it will all be missing descriptions
            ar4.append(parsed_response[3]) # so this tells us if the GPT failed to generate an expected portion
            
        except Exception as e: # this is just the exception portion
            print(f"Couldn't process: {prod_des}. Error: {e}") # we r just saying if we couldn't process any part of the inputs
            ar1.append('could not process')
            ar2.append('could not process')
            ar3.append('could not process')
            ar4.append('could not process')

    return df_mk(ar1, ar2, ar3, ar4) # finally we make the df w the arrays we built

# creating the df
df = call_marketinGPT()

# styling df
styled_df = df.style.set_properties(**{ # so we're just making another df based on the generated df but w styling
    'background-color': 'aliceblue', # the syntax for setting stuff is [what u wanna set]: [val]
    'color': 'black',
    'text-align': 'center',
    'border': '2px solid lightsteelblue !important'
}).set_table_styles([ # below, the selector means it applies to all the following, being thead, which is the table header section (thead)'s table headers (th)
    {'selector': 'thead th', 'props': [('background-color', 'lightslategrey'), ('color', 'white'), ('border', '2px solid darkslategray !important'), ('text-align', 'center')]}
]).hide(axis="index") # ^ marked border as !important to make sure it's done bc it kept on NOT appearing??

# displaying the styled df
display(styled_df)