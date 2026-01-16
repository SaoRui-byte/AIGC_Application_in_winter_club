from zhipuai import ZhipuAI
import time
client = ZhipuAI(api_key="")


def stream_ouput(stream_data):
    for i in stream_data:
        print(i,end='')
        time.sleep(0.05)
        
def chat(message):
    response = client.chat.completions.create(
        model="glm-4",
        messages=[
            {"role": "system", "content": f"你是健康饮食专家"},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content

stream_ouput("请输入你的专属AI名称：")
n = input()


while True:
    stream_ouput(f'我是{n}，有什么可以帮到你的：')
    message = input()
    if(message == '再见'):
        print('bye')
        break

    stream_ouput(chat(message))
    print('\n')


