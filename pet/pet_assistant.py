import streamlit as st
from zhipuai import ZhipuAI
import os
import base64
import tempfile
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
import numpy as np
from aip import AipSpeech  # 百度语音识别SDK

# 加载环境变量
load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

# 百度语音识别配置
BAIDU_APP_ID = os.getenv("BAIDU_APP_ID")
BAIDU_API_KEY = os.getenv("BAIDU_API_KEY")
BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")
baidu_client = AipSpeech(BAIDU_APP_ID, BAIDU_API_KEY, BAIDU_SECRET_KEY) if BAIDU_APP_ID and BAIDU_API_KEY and BAIDU_SECRET_KEY else None

# 初始化会话状态
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 对话历史
if "uploaded_image_base64" not in st.session_state:
    st.session_state.uploaded_image_base64 = None  # 上传的图片

# ------------------------------
# 核心：sounddevice本地录音（替代webrtc）
# ------------------------------
def record_audio_with_sounddevice(duration=5, samplerate=16000):
    """
    使用sounddevice本地录音
    :param duration: 录音时长（秒）
    :param samplerate: 采样率（适配百度接口）
    :return: 录音的WAV字节流 / None
    """
    try:
        st.info(f"🎤 开始录音 {duration} 秒...请对着麦克风说话！")
        # 开始录音（阻塞式，录满指定时长）
        audio_data = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,  # 单声道
            dtype='int16'  # 16bit格式（百度接口要求）
        )
        sd.wait()  # 等待录音完成
        
        # 将录音转为WAV字节流（无需保存本地文件，直接内存处理）
        wav_buffer = tempfile.SpooledTemporaryFile()
        sf.write(wav_buffer, audio_data, samplerate, format='WAV')
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.read()
        wav_buffer.close()
        
        st.success("✅ 录音完成！正在识别语音内容...")
        return wav_bytes
    except Exception as e:
        st.error(f"❌ 录音失败：{str(e)}")
        st.info("💡 提示：请检查麦克风是否正常，或重新安装sounddevice（pip install sounddevice --upgrade）")
        return None

# ------------------------------
# 百度语音识别（适配sounddevice录音）
# ------------------------------
def baidu_speech_to_text(wav_bytes):
    """将sounddevice录制的WAV字节流转为文字"""
    if not baidu_client:
        st.error("❌ 未配置百度语音参数，请在.env文件中填写BAIDU_APP_ID/API_KEY/SECRET_KEY")
        return ""
    
    try:
        # 提取WAV的纯音频数据（去掉44字节文件头，适配百度PCM格式）
        pcm_data = wav_bytes[44:] if len(wav_bytes) > 44 else wav_bytes
        
        # 调用百度短语音识别接口
        result = baidu_client.asr(pcm_data, 'pcm', 16000, {
            'dev_pid': 1537,  # 普通话识别
        })
        
        # 解析结果
        if result.get("err_no") == 0 and "result" in result and len(result["result"]) > 0:
            return result["result"][0]
        elif result.get("err_no") == 3301:
            st.warning("⚠️ 录音中未检测到有效声音，请靠近麦克风并提高音量")
        else:
            st.error(f"❌ 识别失败：{result.get('err_msg', '未知错误')}（错误码：{result.get('err_no')}）")
        return ""
    except Exception as e:
        st.error(f"❌ 调用百度接口出错：{str(e)}")
        return ""

# ------------------------------
# 智谱AI核心功能（保留原有逻辑）
# ------------------------------
def pet_multimodal_chat(image_base64, user_input, chat_history):
    context = "\n".join([f"{item['role']}: {item['content']}" for item in chat_history])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"""
                    你是专业的宠物医生助手，结合历史对话、当前图片和用户问题，回答简洁精准：
                    历史对话：{context}
                    用户问题：{user_input}
                """},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]
        }
    ]
    try:
        response = client.chat.completions.create(model="glm-4v", messages=messages, temperature=0.3)
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ 多模态请求出错：{str(e)}")
        return "抱歉，暂时无法处理图片请求，请稍后再试。"

def pet_text_chat(user_input, chat_history):
    context = "\n".join([f"{item['role']}: {item['content']}" for item in chat_history])
    prompt = f"""
        你是专业的宠物养护助手，结合历史对话回答用户问题，要个性化：
        历史对话：{context}
        用户问题：{user_input}
    """
    try:
        response = client.chat.completions.create(model="glm-4", messages=[{"role": "user", "content": prompt}], temperature=0.3)
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ 文本请求出错：{str(e)}")
        return "抱歉，暂时无法处理请求，请稍后再试。"

# ------------------------------
# 界面布局（核心：sounddevice录音模块）
# ------------------------------
st.title("🐾 宠物识别与养护助手 | 本地录音版")
st.caption("（大二作业 · sounddevice录音 + 智谱AI + 百度语音识别）")

# 侧边栏功能区
with st.sidebar:
    # 百度语音状态提示
    if baidu_client:
        st.success("✅ 已连接百度语音识别服务")
    else:
        st.error("❌ 未配置百度语音参数")
    
    # 1. 图片上传
    st.subheader("📷 上传宠物照片")
    uploaded_image = st.file_uploader("选择照片（jpg/png）", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image_base64 = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
        st.session_state.uploaded_image_base64 = f"data:image/jpeg;base64,{image_base64}"
        st.image(uploaded_image, caption="已上传的宠物照片", use_column_width=True)
    
    st.divider()
    
    # 2. sounddevice本地录音模块（核心修改）
    st.subheader("🎤 本地语音提问（无浏览器依赖）")
    st.info("💡 操作流程：输入录音时长 → 点击录音 → 说话 → 自动识别 → AI回复")
    
    # 录音时长输入（默认5秒，可自定义）
    record_duration = st.number_input(
        "录音时长（秒）",
        min_value=1, max_value=10, value=5, step=1,
        help="建议3-5秒，过长可能识别不准确"
    )
    
    # 开始录音按钮
    if st.button("▶️ 开始录音并识别", type="primary"):
        # 第一步：本地录音
        wav_bytes = record_audio_with_sounddevice(duration=record_duration)
        if not wav_bytes:
            st.stop()
        
        # 第二步：调用百度识别
        recognized_text = baidu_speech_to_text(wav_bytes)
        if not recognized_text:
            st.stop()
        
        # 第三步：识别成功，自动提交到聊天框
        st.success(f"✅ 语音识别结果：{recognized_text}")
        user_prompt = recognized_text
        
        # 展示用户输入
        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        
        # 调用智谱AI生成回复
        with st.chat_message("assistant"):
            with st.spinner("🤔 正在生成回复..."):
                if st.session_state.uploaded_image_base64:
                    response = pet_multimodal_chat(st.session_state.uploaded_image_base64, user_prompt, st.session_state.chat_history)
                else:
                    response = pet_text_chat(user_prompt, st.session_state.chat_history)
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    st.divider()
    
    # 3. 功能按钮
    if st.button("⏹️ 结束项目", type="primary"):
        st.warning("⚠️ 项目已停止运行！")
        st.info("✅ 请在终端按 Ctrl + C 彻底关闭服务")
        st.stop()
    
    if st.button("🗑️ 清空对话历史"):
        st.session_state.chat_history = []
        st.session_state.uploaded_image_base64 = None
        st.rerun()

# ------------------------------
# 聊天界面（保留文字输入）
# ------------------------------
# 渲染历史对话
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 文字输入框
user_prompt = st.chat_input("输入你的问题（如：它一直挠耳朵怎么办？）")
if user_prompt:
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("正在思考回复..."):
            if st.session_state.uploaded_image_base64:
                response = pet_multimodal_chat(st.session_state.uploaded_image_base64, user_prompt, st.session_state.chat_history)
            else:
                response = pet_text_chat(user_prompt, st.session_state.chat_history)
        st.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})