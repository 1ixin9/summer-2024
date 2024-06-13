from langchain.prompts import PromptTemplate # for creating the template we feed to llm
from langchain_community.chat_models import QianfanChatEndpoint # for getting the actual GPT llm
from langchain_core.output_parsers import StrOutputParser # for converting llm output to something Python understands
import pandas as pd # for making dfs
from IPython.display import display, HTML # for displaying the df with styling
import re # for regular expressions

# key
qianfan_ak = "DAEEqjuvglLTgQMCXqRvqfUj"
qianfan_sk = "s0AJ849GNB6440lwLWDvGuNEJNrgrbQ3"

# model
llm = QianfanChatEndpoint(model="ERNIE-4.0-8K", streaming=True, qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk, penalty_score=1)

def df_mk(ar1, ar2, ar3, ar4):
    df = pd.DataFrame({ # this is the syntax for making a df which is basically a table in pandas
        "产品描述": ar1, # the quotes has the title of the column
        "产品卖点": ar2, # u can either make an array like [val, val] urself
        "最佳营销卖点": ar3, # or u can use an array that u alr made
        "目标受众": ar4 # and put it into the df like that
    })
    return df

def parse_response(response): # this parses the response param
    
    # split the response into lines
    lines = response.split('\n')
    
    # initialize placeholders
    product_description = 'missing description'
    selling_points = 'missing description'
    best_marketing_point = 'missing description'
    target_audience = 'missing description'
    
    # iterate over lines and find the relevant sections
    for line in lines:
        if line.startswith("1. **产品描述**："):
            product_description = line[len("1. **产品描述**："):].strip()
        elif line.startswith("2. **产品卖点**："):
            selling_points = line[len("2. **产品卖点**："):].strip()
        elif line.startswith("3. **最佳营销卖点**："):
            best_marketing_point = line[len("3. **最佳营销卖点**："):].strip()
        elif line.startswith("4. **目标受众**："):
            target_audience = line[len("4. **目标受众**："):].strip()
    
    return product_description, selling_points, best_marketing_point, target_audience


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
    
    prompt = PromptTemplate(
        # template is the prompt that ur using to prompt engineer the GPT
        template="""作为一名零售顾问助手，你的任务是帮助用户分析他们的产品描述，
        并提供该产品的卖点、最佳营销卖点、目标受众以及针对目标受众的营销策略。
        请根据以下格式进行回复，并且仅根据用户提供的信息进行分析和回答：\n\n
            1. **产品描述**：用户提供的产品详细信息。\n
            2. **产品卖点**：根据产品描述，提炼出吸引潜在消费者的关键特点。\n
            3. **最佳营销卖点**：从产品卖点中选择最具市场潜力的特点，并解释为何这个卖点最有吸引力。\n
            4. **目标受众**：根据产品卖点，确定最适合的消费群体。\n\n
            
        以下是一个示例对话：\n
        
        用户：我们有一款新型的可折叠电动自行车，重量轻，电池续航长，适合城市通勤。\n\n
        系统：\n
            1. **产品描述**：新型可折叠电动自行车，重量轻，电池续航长，适合城市通勤。\n
            2. **产品卖点**：轻便设计、长续航电池、便捷的城市通勤工具。\n
            3. **最佳营销卖点**：长续航电池，因为城市通勤用户对续航时间有较高需求，能够减少充电频率。\n
            4. **目标受众**：城市白领、大学生、注重环保和便捷出行的用户。\n\n
            
        请提供您的产品描述：\n
        
        {prod}\n\n

        1. **产品描述**：用户提供的产品描述\n
        2. **产品卖点**：提炼出的产品卖点\n
        3. **最佳营销卖点**：选择的最佳营销卖点及其原因\n
        4. **目标受众**：确定的目标消费群体\n""",
        
        input_variables=["prod"] # here ur telling the gpt that the input variables it uses will be
        # used where {prod} is used in the template
    )
    
    ar1, ar2, ar3, ar4 = [], [], [], [] # here ur declaring the arrays that the df will be made w

    for prod_des in prod_descr: # this is a for-each loop which ensures that each val in the list is used
        
        marketinGPT = prompt | llm | StrOutputParser() # this is the setup for the processing pipeline
        # prompt refers to the template ur using to prompt engineer -> this is given to llm
        # llm then takes the text input and generates a response
        # StrOutputParser is an output parser (DUH but also an output parser takes raw output and turns it into a structured format)
        # ^ this turns the llm's output into something Python can easily understand
        # using the '|' operator is basically the chaining part of the processing pipeline
        # ^ this says the output of one component should be used as the input of the next component
        # ^ so the prompt's output is the llm's input, the llm's output is the parser's input
        
        try: # we using a try-except bc who knows if the GPT will output something that is always understandable
            ans = marketinGPT.invoke({"prod": prod_des}) # marketinGPT (brilliant name) is the name of the pipeline
            # so when we call it, we r getting an instance of it
            # .invoke(input val) is a method that tells the model to provide a response based on the input val
            # "prod" is the input variable in the prompt, prod_des is the value in the for-each loop
            # this lets prod_des be passed in as the input of the prompt
            
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
styled_df = df.style.set_properties(**{
    'background-color': 'white',
    'color': 'black',
    'border-color': 'black'
}).set_table_styles([
    {'selector': 'thead th', 'props': [('background-color', 'black'), ('color', 'white')]}
]).set_caption("产品营销分析")

# displaying the styled df
display(HTML(styled_df.to_html()))