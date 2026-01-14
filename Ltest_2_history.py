from zhipuai import ZhipuAI
import time

client = ZhipuAI(api_key="保密")

def stream_output(stream_data):
    for i in stream_data:
        print(i, end='')
        time.sleep(0.05)

def chat(messages_history):
    response = client.chat.completions.create(
        model="glm-4",
        messages=messages_history,
    )
    return response.choices[0].message.content

# 初始化对话历史
conversation_history = [
    {"role": "system", "content": "你是钢铁侠里的Javis,我是你的boss"}
]

stream_output("请输入你的专属AI名称：")
n = input()

while True:
    stream_output(f'我是{n}，有什么可以帮到你的：')
    user_message = input()
    
    if user_message == '再见':
        print('bye')
        break
    
    # 将用户的输入添加到对话历史中
    conversation_history.append({"role": "user", "content": user_message})
    
    # 获取AI的回复
    ai_response = chat(conversation_history)
    
    # 将AI的回复也添加到对话历史中
    conversation_history.append({"role": "assistant", "content": ai_response})
    
    stream_output(ai_response)
    print('\n')