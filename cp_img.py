import base64
from zai import ZhipuAiClient

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

client = ZhipuAiClient(api_key="")

# # 方式1：使用图像URL
# response = client.chat.completions.create(
#     model="glm-4.6v",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "这张图片里有什么？请详细描述。"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": "https://example.com/image.jpg"
#                     }
#                 }
#             ]
#         }
#     ]
# )

# print(response.choices[0].message.content)

# 方式2：使用base64编码的图像
base64_image = encode_image("D:\壁纸\夏天心.jpg")

response = client.chat.completions.create(
    model="glm-4.6v",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "图片里的女生是哪国人？"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)