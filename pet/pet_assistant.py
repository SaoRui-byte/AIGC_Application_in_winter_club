import streamlit as st
from zhipuai import ZhipuAI
import os
import base64
import tempfile
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
import numpy as np
from aip import AipSpeech
import io
import re

# 加载环境变量
load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

# 百度语音配置
BAIDU_APP_ID = os.getenv("BAIDU_APP_ID")
BAIDU_API_KEY = os.getenv("BAIDU_API_KEY")
BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")
baidu_client = AipSpeech(BAIDU_APP_ID, BAIDU_API_KEY, BAIDU_SECRET_KEY) if all([BAIDU_APP_ID, BAIDU_API_KEY, BAIDU_SECRET_KEY]) else None

# 初始化会话状态（新增新图片上传标志）
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  
if "uploaded_image_base64" not in st.session_state:
    st.session_state.uploaded_image_base64 = None
if "tts_audio_segments" not in st.session_state:
    st.session_state.tts_audio_segments = []
if "last_image_uploaded" not in st.session_state:
    st.session_state.last_image_uploaded = None
# 新增：跟踪是否刚上传了新图片
if "is_new_image_uploaded" not in st.session_state:
    st.session_state.is_new_image_uploaded = False

# ------------------------------
# 辅助函数：意图检测
# ------------------------------
def detect_intent(user_input):
    """检测用户输入的意图，判断是回溯历史还是当前图片提问"""
    # 回溯历史的关键词（新增"上一个问题"等精准关键词）
    history_keywords = ["之前", "刚才", "之前问的", "那只", "之前的", "之前说的", 
                       "之前的问题", "上一个问题", "上一个", "刚才问的"]
    # 当前图片的关键词
    current_image_keywords = ["这只", "这是什么", "它", "这张", "当前", "现在"]
    
    # 检查是否包含回溯历史的关键词
    if any(keyword in user_input for keyword in history_keywords):
        return "history"
    # 检查是否包含当前图片的关键词
    elif any(keyword in user_input for keyword in current_image_keywords):
        return "current_image"
    else:
        return "default"

# ------------------------------
# 1. 本地录音功能
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
# 2. 百度语音识别（ASR）
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
# 3. 百度语音合成（TTS）
# ------------------------------
def baidu_text_to_speech(text, per=0):
    if not baidu_client:
        st.error("❌ 未配置百度语音参数，无法播报语音")
        return None
    
    # 文本清洗
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    if not text:
        st.warning("⚠️ 无有效文本可合成语音")
        return None
    
    MAX_SEGMENT_LEN = 500
    text_segments = [text[i:i+MAX_SEGMENT_LEN].strip() for i in range(0, len(text), MAX_SEGMENT_LEN) if text[i:i+MAX_SEGMENT_LEN].strip()]
    
    try:
        audio_segments = []
        for idx, segment in enumerate(text_segments):
            result = baidu_client.synthesis(
                segment,
                'zh',
                1,
                {
                    'vol': 5,
                    'per': per,
                    'spd': 5,
                    'pit': 5,
                    'aue': 3
                }
            )
            
            if isinstance(result, dict):
                st.error(f"❌ 第{idx+1}段合成失败：{result.get('err_msg', '未知错误')}")
                return None
            audio_segments.append(result)
        
        st.success(f"✅ 语音合成完成（共{len(audio_segments)}段）")
        return audio_segments
    except Exception as e:
        st.error(f"❌ 语音合成出错：{str(e)}")
        return None

# ------------------------------
# 4. 前端音频合并播放
# ------------------------------
def merge_audio_frontend(audio_segments):
    if not audio_segments:
        return None
    
    segment_base64_list = [base64.b64encode(seg).decode('utf-8') for seg in audio_segments]
    
    js_code = f"""
    <script>
    let audioContext = null;
    let source = null;
    let mergedBuffer = null;
    let isPlaying = false;
    let startTime = 0;
    let pauseTime = 0;

    async function togglePlayback() {{
        if (!audioContext) {{
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const buffers = [];
            for (const base64 of {segment_base64_list}) {{
                const response = await fetch(`data:audio/mp3;base64,${{base64}}`);
                const arrayBuffer = await response.arrayBuffer();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                buffers.push(audioBuffer);
            }}
            const totalLength = buffers.reduce((sum, buf) => sum + buf.length, 0);
            mergedBuffer = audioContext.createBuffer(1, totalLength, buffers[0].sampleRate);
            let offset = 0;
            for (const buf of buffers) {{
                mergedBuffer.getChannelData(0).set(buf.getChannelData(0), offset);
                offset += buf.length;
            }}
        }}

        if (isPlaying) {{
            source.stop();
            pauseTime = audioContext.currentTime - startTime;
            isPlaying = false;
            document.getElementById('mergePlayBtn').innerText = '▶️ 继续播放完整语音';
        }} else {{
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
# 5. 智谱AI对话
# ------------------------------
def pet_multimodal_chat(image_base64, user_input, chat_history, use_history=True):
    messages = [
        {"role": "system", "content": "你是专业的宠物专家，精通动物品种和动物医疗方面知识，回答要简洁精准。如果用户提问涉及品种识别，请先识别品种，再回答问题；如果用户判断错误，要指出并解释。"}
    ]
    
    # 根据use_history决定是否添加历史对话
    if use_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_input},
            {"type": "image_url", "image_url": {"url": image_base64}}
        ]
    })
    
    try:
        response = client.chat.completions.create(
            model="glm-4v",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ 多模态请求出错：{str(e)}")
        return "抱歉，暂时无法处理图片请求，请稍后再试。"

# 【核心修改】新增exclude_last_user参数，排除当前提问，只传更早的历史
def pet_text_chat(user_input, chat_history, use_history=True, exclude_last_user=False):
    messages = [
        {"role": "system", "content": "你是专业的宠物养护助手，结合历史对话回答用户问题，回答要个性化、简洁实用。如果用户问上一个问题/之前的问题是什么，请准确引用历史对话内容回答。"}
    ]
    
    # 根据use_history决定是否添加历史对话
    if use_history:
        # 排除最后一轮用户提问（当前的回溯提问），只传更早的历史
        if exclude_last_user and len(chat_history) >= 2:
            history_to_use = chat_history[:-1]  # 去掉最后一条（当前用户提问）
        else:
            history_to_use = chat_history
        
        for msg in history_to_use:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": user_input})
    
    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ 文本请求出错：{str(e)}")
        return "抱歉，暂时无法处理请求，请稍后再试。"

# ------------------------------
# 6. 界面布局（核心：自动切换逻辑）
# ------------------------------
st.title("🐾 宠物识别与养护助手 ")

# 侧边栏
with st.sidebar:
    # 百度语音状态
    if baidu_client:
        st.success("✅ 已连接百度语音识别/合成服务")
    else:
        st.error("❌ 未配置百度语音参数")
    
    # 图片上传
    st.subheader("📷 上传宠物照片")
    uploaded_image = st.file_uploader(
        "选择照片（jpg/png）", 
        type=["jpg", "png", "jpeg"],
        key="pet_image_uploader",
        help="上传新照片会自动触发「只看当前照片，不参考历史」模式"
    )
    
    # 检测新图片上传（核心：设置新图片标志）
    if uploaded_image:
        image_identifier = f"{uploaded_image.name}_{uploaded_image.size}"
        if image_identifier != st.session_state.last_image_uploaded:
            image_base64 = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
            st.session_state.uploaded_image_base64 = f"data:image/jpeg;base64,{image_base64}"
            st.session_state.last_image_uploaded = image_identifier
            # 关键：标记为刚上传新图片
            st.session_state.is_new_image_uploaded = True
            st.success("✅ 新图片已上传！AI将仅参考当前照片回答，不使用历史对话")
        st.image(uploaded_image, caption="当前上传的宠物照片", use_column_width=True)
    else:
        st.session_state.uploaded_image_base64 = None
        st.session_state.last_image_uploaded = None
        st.info("请上传宠物照片以启用图片识别功能")
    
    st.divider()
    
    # 语音设置
    st.subheader("🎤 本地语音提问")
    record_duration = st.number_input("录音时长（秒）", min_value=1, max_value=10, value=5, step=1, key="record_duration")
    
    st.subheader("🔊 语音播报设置")
    voice_type = st.selectbox(
        "选择发音人",
        options=["女声（默认）", "男声", "情感女声", "情感男声"],
        index=0,
        key="voice_type"
    )
    per_map = {"女声（默认）":0, "男声":1, "情感女声":3, "情感男声":4}
    selected_per = per_map[voice_type]
    
    # 录音按钮
    if st.button("▶️ 开始录音并识别", type="primary", key="record_btn"):
        wav_bytes = record_audio_with_sounddevice(duration=record_duration)
        if not wav_bytes:
            st.stop()
        
        recognized_text = baidu_speech_to_text(wav_bytes)
        if not recognized_text:
            st.stop()
        
        st.success(f"✅ 语音识别结果：{recognized_text}")
        user_prompt = recognized_text
        
        # 添加用户语音输入到对话历史
        with st.chat_message("user"):
            st.markdown(f"🎤 语音输入：{user_prompt}")
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        
        # 自动判断模式
        intent = detect_intent(user_prompt)
        # 刚上传新图片 → 只看图片，不看历史
        if st.session_state.is_new_image_uploaded:
            use_image = True
            use_history = False
            # 重置新图片标志
            st.session_state.is_new_image_uploaded = False
        elif intent == "history":
            # 回溯历史 → 只看历史，不看图片
            use_image = False
            use_history = True
        elif intent == "current_image":
            # 当前图片提问 → 看图片+历史
            use_image = True
            use_history = True
        else:
            # 默认模式
            use_image = True if st.session_state.uploaded_image_base64 else False
            use_history = True
        
        # 生成AI回复
        with st.chat_message("assistant"):
            with st.spinner("🤔 正在生成回复..."):
                if use_image and st.session_state.uploaded_image_base64:
                    response = pet_multimodal_chat(st.session_state.uploaded_image_base64, user_prompt, st.session_state.chat_history, use_history)
                else:
                    # 【核心修改】回溯历史时，排除当前提问
                    exclude_last = True if intent == "history" else False
                    response = pet_text_chat(user_prompt, st.session_state.chat_history, use_history, exclude_last)
            st.markdown(response)
            
            # 语音合成
            tts_audio_segments = baidu_text_to_speech(response, per=selected_per)
            if tts_audio_segments:
                st.session_state.tts_audio_segments = tts_audio_segments
                merge_js = merge_audio_frontend(st.session_state.tts_audio_segments)
                if merge_js:
                    st.components.v1.html(merge_js, height=50)
                for idx, audio_bytes in enumerate(st.session_state.tts_audio_segments):
                    st.caption(f"🎧 语音播报 - 第{idx+1}段")
                    st.audio(audio_bytes, format='audio/mp3', start_time=0)
        
        # 添加AI回复到对话历史
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
        st.rerun()

# ------------------------------
# 聊天界面
# ------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 修复：使用全局状态的tts_audio_segments，而非局部变量
        if msg["role"] == "assistant" and st.session_state.tts_audio_segments:
            merge_js = merge_audio_frontend(st.session_state.tts_audio_segments)
            if merge_js:
                st.components.v1.html(merge_js, height=50)
            for seg_idx, audio_bytes in enumerate(st.session_state.tts_audio_segments):
                st.caption(f"🎧 语音播报 - 第{seg_idx+1}段")
                st.audio(audio_bytes, format='audio/mp3')

# 文字输入框
user_prompt = st.chat_input("输入你的问题（如：它一直挠耳朵怎么办？）", key="chat_input")
if user_prompt:
    # 添加文字输入到对话历史
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    
    # 自动判断模式
    intent = detect_intent(user_prompt)
    # 刚上传新图片 → 只看图片，不看历史
    if st.session_state.is_new_image_uploaded:
        use_image = True
        use_history = False
        # 重置新图片标志
        st.session_state.is_new_image_uploaded = False
    elif intent == "history":
        # 回溯历史 → 只看历史，不看图片
        use_image = False
        use_history = True
    elif intent == "current_image":
        # 当前图片提问 → 看图片+历史
        use_image = True
        use_history = True
    else:
        # 默认模式
        use_image = True if st.session_state.uploaded_image_base64 else False
        use_history = True
    
    # 生成AI回复
    with st.chat_message("assistant"):
        with st.spinner("正在思考回复..."):
            if use_image and st.session_state.uploaded_image_base64:
                response = pet_multimodal_chat(st.session_state.uploaded_image_base64, user_prompt, st.session_state.chat_history, use_history)
            else:
                # 【核心修改】回溯历史时，传入exclude_last_user=True，排除当前提问
                exclude_last = True if intent == "history" else False
                response = pet_text_chat(user_prompt, st.session_state.chat_history, use_history, exclude_last)
        st.markdown(response)
        
        # 语音合成
        selected_per = per_map.get(st.session_state.get("voice_type", "女声（默认）"), 0)
        tts_audio_segments = baidu_text_to_speech(response, per=selected_per)
        if tts_audio_segments:
            st.session_state.tts_audio_segments = tts_audio_segments
            merge_js = merge_audio_frontend(st.session_state.tts_audio_segments)
            if merge_js:
                st.components.v1.html(merge_js, height=50)
            for idx, audio_bytes in enumerate(st.session_state.tts_audio_segments):
                st.caption(f"🎧 语音播报 - 第{idx+1}段")
                st.audio(audio_bytes, format='audio/mp3', start_time=0)
    
    # 添加AI回复到对话历史
    st.session_state.chat_history.append({"role": "assistant", "content": response})