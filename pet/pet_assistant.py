import streamlit as st
from zhipuai import ZhipuAI
import os
import base64
import tempfile
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
import numpy as np
from aip import AipSpeech  # 百度语音识别/合成SDK
import io
import re  # 文本清洗用

# 加载环境变量
load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

# 百度语音配置（识别+合成共用）
BAIDU_APP_ID = os.getenv("BAIDU_APP_ID")
BAIDU_API_KEY = os.getenv("BAIDU_API_KEY")
BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")
baidu_client = AipSpeech(BAIDU_APP_ID, BAIDU_API_KEY, BAIDU_SECRET_KEY) if BAIDU_APP_ID and BAIDU_API_KEY and BAIDU_SECRET_KEY else None

# 初始化会话状态（新增图片上传的状态跟踪）
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 对话历史
if "uploaded_image_base64" not in st.session_state:
    st.session_state.uploaded_image_base64 = None  # 上传的图片
if "tts_audio_segments" not in st.session_state:
    st.session_state.tts_audio_segments = []  # 存储分段音频字节流
if "last_image_uploaded" not in st.session_state:
    st.session_state.last_image_uploaded = None  # 跟踪最后一次上传的图片标识

# ------------------------------
# 核心1：sounddevice本地录音
# ------------------------------
def record_audio_with_sounddevice(duration=5, samplerate=16000):
    try:
        st.info(f"🎤 开始录音 {duration} 秒...请对着麦克风说话！")
        audio_data = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype='int16'
        )
        sd.wait()
        
        # 转为WAV字节流
        wav_buffer = tempfile.SpooledTemporaryFile()
        sf.write(wav_buffer, audio_data, samplerate, format='WAV')
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.read()
        wav_buffer.close()
        
        st.success("✅ 录音完成！正在识别语音内容...")
        return wav_bytes
    except Exception as e:
        st.error(f"❌ 录音失败：{str(e)}")
        return None

# ------------------------------
# 核心2：百度语音识别（ASR）
# ------------------------------
def baidu_speech_to_text(wav_bytes):
    if not baidu_client:
        st.error("❌ 未配置百度语音参数，请检查.env文件")
        return ""
    
    try:
        pcm_data = wav_bytes[44:] if len(wav_bytes) > 44 else wav_bytes
        result = baidu_client.asr(pcm_data, 'pcm', 16000, {'dev_pid': 1537})
        
        if result.get("err_no") == 0 and "result" in result and len(result["result"]) > 0:
            return result["result"][0]
        elif result.get("err_no") == 3301:
            st.warning("⚠️ 未检测到有效声音，请提高音量")
        else:
            st.error(f"❌ 识别失败：{result.get('err_msg', '未知错误')}")
        return ""
    except Exception as e:
        st.error(f"❌ 调用百度接口出错：{str(e)}")
        return ""

# ------------------------------
# 核心3：百度语音合成（TTS）- 修复所有报错
# ------------------------------
def baidu_text_to_speech(text, per=0):
    """
    百度文字转语音（TTS）- 无ffmpeg/pydub + 修复param err + 兼容旧版Streamlit
    """
    if not baidu_client:
        st.error("❌ 未配置百度语音参数，无法播报语音")
        return None
    
    # 1. 文本清洗（去除特殊字符/换行/多余空格）
    text = re.sub(r'\n+', ' ', text)  # 换行替换为空格
    text = re.sub(r'\s+', ' ', text)  # 多个空格合并为一个
    text = text.strip()               # 去除首尾空格
    
    # 2. 空文本校验
    MAX_SEGMENT_LEN = 500
    if not text or len(text) == 0:
        st.warning("⚠️ 无有效文本可合成语音")
        return None
    
    try:
        # 3. 文本分段（500字/段）
        text_segments = []
        if len(text) <= MAX_SEGMENT_LEN:
            text_segments = [text]
        else:
            st.warning(f"⚠️ 文本过长（{len(text)}字），将分为{len(text)//MAX_SEGMENT_LEN + 1}段播放")
            for i in range(0, len(text), MAX_SEGMENT_LEN):
                segment = text[i:i+MAX_SEGMENT_LEN].strip()
                if segment:  # 跳过空分段
                    text_segments.append(segment)
        
        # 4. 逐段合成语音（修正百度API参数顺序）
        audio_segments = []
        for idx, segment in enumerate(text_segments):
            # 百度TTS正确参数格式
            result = baidu_client.synthesis(
                segment,          # 参数1：要合成的文本
                'zh',             # 参数2：语言（中文）
                1,                # 参数3：客户端类型（固定1）
                {
                    'vol': 5,     # 音量（0-15）
                    'per': per,   # 发音人（0=女声，1=男声，3=情感女声，4=情感男声）
                    'spd': 5,     # 语速（0-9）
                    'pit': 5,     # 音调（0-9）
                    'aue': 3      # 音频格式（3=mp3，兼容前端播放）
                }
            )
            
            # 5. 处理合成结果
            if isinstance(result, dict):
                st.error(f"❌ 第{idx+1}段合成失败：{result.get('err_msg', '未知错误')}")
                return None
            audio_segments.append(result)
            st.info(f"✅ 第{idx+1}段语音合成完成")
        
        st.success("✅ 所有语音段合成完成！可依次播放或合并播放")
        return audio_segments
    
    except Exception as e:
        st.error(f"❌ 语音合成出错：{str(e)}")
        return None

# ------------------------------
# 核心4：前端合并音频（Web Audio API）
# ------------------------------
def merge_audio_frontend(audio_segments):
    """
    将分段音频字节流转为base64，传给前端用Web Audio API合并（带暂停/防重叠功能）
    """
    if not audio_segments or len(audio_segments) == 0:
        return None
    
    # 将每个音频字节流转为base64
    segment_base64_list = [base64.b64encode(seg).decode('utf-8') for seg in audio_segments]
    
    # 生成前端合并音频的JavaScript代码（修复重叠+暂停功能）
    js_code = f"""
    <script>
    // 全局变量管理播放状态
    let audioContext = null;
    let source = null;
    let mergedBuffer = null;
    let isPlaying = false;
    let startTime = 0;
    let pauseTime = 0;

    async function togglePlayback() {{
        if (!audioContext) {{
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            // 首次加载时合并音频
            const buffers = [];
            for (const base64 of {segment_base64_list}) {{
                const response = await fetch(`data:audio/mp3;base64,${{base64}}`);
                const arrayBuffer = await response.arrayBuffer();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                buffers.push(audioBuffer);
            }}
            // 合并所有音频片段
            const totalLength = buffers.reduce((sum, buf) => sum + buf.length, 0);
            mergedBuffer = audioContext.createBuffer(1, totalLength, buffers[0].sampleRate);
            let offset = 0;
            for (const buf of buffers) {{
                mergedBuffer.getChannelData(0).set(buf.getChannelData(0), offset);
                offset += buf.length;
            }}
        }}

        if (isPlaying) {{
            // 暂停播放
            source.stop();
            pauseTime = audioContext.currentTime - startTime;
            isPlaying = false;
            document.getElementById('mergePlayBtn').innerText = '▶️ 继续播放完整语音';
        }} else {{
            // 开始/继续播放
            if (source && source.state === 'running') {{
                source.stop();
            }}
            source = audioContext.createBufferSource();
            source.buffer = mergedBuffer;
            source.connect(audioContext.destination);
            source.start(0, pauseTime);
            startTime = audioContext.currentTime - pauseTime;
            isPlaying = true;
            document.getElementById('mergePlayBtn').innerText = '⏸️ 暂停播放';

            // 播放结束后重置状态
            source.onended = () => {{
                isPlaying = false;
                pauseTime = 0;
                document.getElementById('mergePlayBtn').innerText = '🎧 播放合并后的完整语音';
            }};
        }}
    }}
    </script>
    <button id="mergePlayBtn" onclick="togglePlayback()" style="padding: 8px 16px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">
    🎧 播放合并后的完整语音
    </button>
    """
    return js_code

# ------------------------------
# 核心5：智谱AI对话/多模态识别
# ------------------------------
def pet_multimodal_chat(image_base64, user_input, chat_history):
    context = "\n".join([f"{item['role']}: {item['content']}" for item in chat_history])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"""
                    你是专业的宠物医生助手，回答简洁精准：
                    历史对话：{context}
                    用户问题：{user_input}
                    请先识别品种，再分析健康状态，最后给个性化养护建议。
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
# 界面布局（修复图片上传bug）
# ------------------------------
st.title("🐾 宠物识别与养护助手 | 语音交互版")
st.caption("（大二作业 · 百度语音识别+合成 + 智谱AI | 无ffmpeg依赖）")

# 侧边栏功能区
with st.sidebar:
    # 百度语音状态
    if baidu_client:
        st.success("✅ 已连接百度语音识别/合成服务")
    else:
        st.error("❌ 未配置百度语音参数")
    
    # 图片上传（核心修复：添加key + 强制更新状态）
    st.subheader("📷 上传宠物照片")
    uploaded_image = st.file_uploader(
        "选择照片（jpg/png）", 
        type=["jpg", "png", "jpeg"],
        key="pet_image_uploader",  # 关键：添加唯一key，确保组件状态跟踪
        help="上传新图片会自动替换旧图片，无需清空对话"
    )
    
    # 修复：检测新图片上传并强制更新session_state
    if uploaded_image:
        # 生成唯一标识（文件名+大小），判断是否是新图片
        image_identifier = f"{uploaded_image.name}_{uploaded_image.size}"
        if image_identifier != st.session_state.last_image_uploaded:
            image_base64 = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
            st.session_state.uploaded_image_base64 = f"data:image/jpeg;base64,{image_base64}"
            st.session_state.last_image_uploaded = image_identifier  # 更新最后上传的标识
            st.success("✅ 新图片已上传并生效！")
        st.image(uploaded_image, caption="当前上传的宠物照片", use_column_width=True)
    else:
        # 无图片时重置状态
        st.session_state.uploaded_image_base64 = None
        st.session_state.last_image_uploaded = None
        st.info("请上传宠物照片以启用图片识别功能")
    
    # 新增：清空图片按钮（可选）
    if st.button("🗑️ 清空当前图片", key="clear_image"):
        st.session_state.uploaded_image_base64 = None
        st.session_state.last_image_uploaded = None
        st.rerun()  # 刷新界面
    
    st.divider()
    
    # 本地录音模块
    st.subheader("🎤 本地语音提问")
    record_duration = st.number_input("录音时长（秒）", min_value=1, max_value=10, value=5, step=1, key="record_duration")
    
    # 语音播报发音人选择
    st.subheader("🔊 语音播报设置")
    voice_type = st.selectbox(
        "选择发音人",
        options=["女声（默认）", "男声", "情感女声", "情感男声"],
        index=0,
        key="voice_type",
        help="不同发音人效果不同，可按需选择"
    )
    per_map = {"女声（默认）":0, "男声":1, "情感女声":3, "情感男声":4}
    selected_per = per_map[voice_type]
    
    # 开始录音按钮
    if st.button("▶️ 开始录音并识别", type="primary", key="record_btn"):
        # 录音 → 识别 → 对话 → 合成语音
        wav_bytes = record_audio_with_sounddevice(duration=record_duration)
        if not wav_bytes:
            st.stop()
        
        recognized_text = baidu_speech_to_text(wav_bytes)
        if not recognized_text:
            st.stop()
        
        st.success(f"✅ 语音识别结果：{recognized_text}")
        user_prompt = recognized_text
        
        # 展示用户输入（修复：去掉key参数）
        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        
        # 调用智谱AI生成回复（使用最新的图片）
        with st.chat_message("assistant"):
            with st.spinner("🤔 正在生成回复..."):
                if st.session_state.uploaded_image_base64:
                    response = pet_multimodal_chat(st.session_state.uploaded_image_base64, user_prompt, st.session_state.chat_history)
                else:
                    response = pet_text_chat(user_prompt, st.session_state.chat_history)
            st.markdown(response)
            
            # 生成语音（修复后）
            tts_audio_segments = baidu_text_to_speech(response, per=selected_per)
            if tts_audio_segments:
                st.session_state.tts_audio_segments = tts_audio_segments
                # 生成前端合并播放的按钮
                merge_js = merge_audio_frontend(tts_audio_segments)
                if merge_js:
                    st.components.v1.html(merge_js, height=50)
                # 保留分段播放按钮
                for idx, audio_bytes in enumerate(tts_audio_segments):
                    st.caption(f"🎧 语音播报 - 第{idx+1}段")
                    st.audio(audio_bytes, format='audio/mp3', start_time=0)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    st.divider()
    
    # 功能按钮
    if st.button("⏹️ 结束项目", type="primary", key="stop_btn"):
        st.warning("⚠️ 项目已停止运行！")
        st.info("✅ 请在终端按 Ctrl + C 彻底关闭服务")
        st.stop()
    
    if st.button("🗑️ 清空对话历史", key="clear_chat"):
        st.session_state.chat_history = []
        st.session_state.tts_audio_segments = []
        # 保留图片状态（可选：如需清空图片，取消下面注释）
        # st.session_state.uploaded_image_base64 = None
        # st.session_state.last_image_uploaded = None
        st.rerun()

# ------------------------------
# 聊天界面（修复：去掉所有st.chat_message的key参数）
# ------------------------------
# 渲染历史对话
for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 助手回复展示语音播放按钮
        if msg["role"] == "assistant" and st.session_state.tts_audio_segments:
            # 生成前端合并播放的按钮
            merge_js = merge_audio_frontend(st.session_state.tts_audio_segments)
            if merge_js:
                st.components.v1.html(merge_js, height=50)
            # 保留分段播放按钮
            for seg_idx, audio_bytes in enumerate(st.session_state.tts_audio_segments):
                st.caption(f"🎧 语音播报 - 第{seg_idx+1}段")
                st.audio(audio_bytes, format='audio/mp3')

# 文字输入框（添加key）
user_prompt = st.chat_input("输入你的问题（如：它一直挠耳朵怎么办？）", key="chat_input")
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
        
        # 生成语音（修复后）
        tts_audio_segments = baidu_text_to_speech(response, per=selected_per if 'selected_per' in locals() else 0)
        if tts_audio_segments:
            st.session_state.tts_audio_segments = tts_audio_segments
            # 生成前端合并播放的按钮
            merge_js = merge_audio_frontend(tts_audio_segments)
            if merge_js:
                st.components.v1.html(merge_js, height=50)
            # 保留分段播放按钮
            for idx, audio_bytes in enumerate(tts_audio_segments):
                st.caption(f"🎧 语音播报 - 第{idx+1}段")
                st.audio(audio_bytes, format='audio/mp3', start_time=0)
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})