from zai import ZhipuAiClient

# 初始化客户端
client = ZhipuAiClient(api_key="")

# 使用网络搜索工具
response = client.chat.completions.create(
    model='glm-4.7',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '英超联赛最新排名榜，给我讲一下，条理清晰，分行输出，前10名就好'},
    ],
    tools=[
        {
            'type': 'web_search',
            'web_search': {
                'search_query': '英超联赛最新排名榜',
                'search_result': True,
            },
        }
    ],
    temperature=0.5,
    max_tokens=2000,
)

# 获取响应内容并写入文件
text = response.choices[0].message.content
print(type(text))
with open('英超联赛排名.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print("内容已保存到 英超联赛排名.txt 文件中")


