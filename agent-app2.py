from dotenv import load_dotenv
import os
import re
import wave
from io import BytesIO
from datetime import datetime
from typing import Dict, Any

import numpy as np
import noisereduce as nr
import streamlit as st

from groq import Groq
from gtts import gTTS

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from tavily import TavilyClient
from langgraph.checkpoint.memory import InMemorySaver

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =========================================================
# ENVIRONMENT
# .env içerisindeki API anahtarlarını yükler.
# =========================================================

load_dotenv()


# =========================================================
# STREAMLIT
# =========================================================

st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")


# =========================================================
# GROQ CLIENT
# Whisper ses -> yazı işlemi için kullanılır.
# Aynı GROQ_API_KEY ile çalışır.
# =========================================================

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =========================================================
# LLM
# AI'ın yazıya dönüştürülmüş kullanıcı mesajını analiz edip
# normal cevap oluşturmasını sağlar.
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


# =========================================================
# TAVILY
# Güncel internet bilgisi gerektiğinde agent tarafından kullanılır.
# =========================================================

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Güncel bilgiler için internette arama yapar."""

    try:
        return tavily_client.search(query=query, max_results=5, search_depth="advanced")
    except Exception as e:
        return {"error": f"Web search failed: {str(e)}"}


# =========================================================
# PDF FONT
# Türkçe karakterlerin PDF'de düzgün görünmesini sağlar.
# =========================================================

def register_pdf_fonts():
    font_paths = [
        (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf")
    ]

    for regular_path, bold_path in font_paths:
        if os.path.exists(regular_path):
            pdfmetrics.registerFont(TTFont("AgentFont", regular_path))
            pdfmetrics.registerFont(TTFont("AgentFontBold", bold_path if os.path.exists(bold_path) else regular_path))
            return "AgentFont", "AgentFontBold"

    raise FileNotFoundError("Türkçe karakter destekleyen PDF fontu bulunamadı.")


PDF_FONT, PDF_FONT_BOLD = register_pdf_fonts()


# =========================================================
# AGENT
# LangGraph memory + Groq + Tavily burada birleştirilir.
# =========================================================

@st.cache_resource
def get_agent():
    memory = InMemorySaver()

    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""Sen yardımsever, teknik ve güvenilir bir AI asistansın.

Kurallar:

1. Güncel bilgi gerekiyorsa web_search tool'unu kullan.
2. Güncel bilgi gerekmiyorsa gereksiz yere web_search kullanma.
3. Kullanıcının önceki mesajlarını dikkate al.
4. Teknik sorularda somut ve uygulanabilir cevaplar ver.
5. Emin olmadığın bilgileri kesin gerçek gibi sunma.
6. Kod verirken çalışabilir ve düzenli kod yaz.
7. Kullanıcı Türkçe yazıyorsa Türkçe cevap ver.
8. Cevapları gereksiz yere uzatma.
9. Kod örneklerini Markdown code block içerisinde göster.
10. Sesli mesajdan gelen metni normal kullanıcı mesajı gibi değerlendir.
11. Web search kullandıysan mümkün olduğunda kaynakları cevap içerisinde belirt.""",
        checkpointer=memory
    )


agent = get_agent()


# =========================================================
# AUDIO ANALYSIS
# Ses dosyasının gerçekten konuşma içerip içermediğini
# AI'a göndermeden önce lokal olarak kontrol eder.
#
# Bu bölüm LLM token tüketmez.
# =========================================================

def analyze_audio_signal(audio_bytes: bytes):
    try:
        audio_buffer = BytesIO(audio_bytes)

        with wave.open(audio_buffer, "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())

        if not frames:
            return False, None, "Ses verisi bulunamadı."

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        if len(audio) == 0:
            return False, None, "Ses verisi boş."

        # RMS ses seviyesini hesaplar.
        rms = np.sqrt(np.mean(np.square(audio)))

        # Çok düşük seviyedeki kayıtları konuşma olarak kabul etmiyoruz.
        if rms < 350:
            return False, None, "Konuşma algılanmadı. Ses çok düşük veya boş."

        # Normalizasyon.
        max_value = np.max(np.abs(audio))

        if max_value > 0:
            audio = audio / max_value

        # Lokal gürültü azaltma.
        # Bu işlem LLM çağrısı yapmadan bilgisayarda gerçekleştirilir.
        try:
            reduced_audio = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                stationary=False,
                prop_decrease=0.8
            )
        except Exception:
            reduced_audio = audio

        # Gürültü azaltma sonrasında tekrar ses kontrolü.
        reduced_rms = np.sqrt(np.mean(np.square(reduced_audio)))

        if reduced_rms < 0.015:
            return False, None, "Konuşma algılanmadı. Kayıt sessiz."

        # 16-bit PCM'e geri dönüştür.
        reduced_audio = np.clip(reduced_audio, -1, 1)
        reduced_audio = (reduced_audio * 32767).astype(np.int16)

        output = BytesIO()

        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(reduced_audio.tobytes())

        output.seek(0)

        return True, output.getvalue(), None

    except Exception as e:
        return False, None, f"Ses analiz edilemedi: {str(e)}"


# =========================================================
# SPEECH TO TEXT
# Temizlenmiş ses dosyasını Whisper Large V3 Turbo'ya gönderir.
#
# Burada LLM olan GPT-OSS-120B henüz çalışmaz.
# Önce yalnızca konuşmayı yazıya çeviriyoruz.
# =========================================================

def transcribe_audio(audio_bytes: bytes) -> str:
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "voice_message.wav"

    transcription = groq_client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3-turbo",
        language="tr",
        response_format="json",
        temperature=0
    )

    return transcription.text.strip()


# =========================================================
# TTS
# AI cevabını Türkçe MP3'e dönüştürür.
# =========================================================

def clean_tts_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"```[\s\S]*?```", "Kod bloğu burada gösterilmiştir.", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def create_audio(text: str) -> bytes:
    cleaned_text = clean_tts_text(text)

    if not cleaned_text:
        return b""

    audio_buffer = BytesIO()

    tts = gTTS(text=cleaned_text, lang="tr", slow=False)
    tts.write_to_fp(audio_buffer)

    audio_buffer.seek(0)

    return audio_buffer.getvalue()


# =========================================================
# PDF
# =========================================================

def clean_pdf_text(text: str) -> str:
    if not text:
        return ""

    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def markdown_to_pdf_elements(text: str, styles):
    elements = []
    lines = text.split("\n")
    code_block = False
    code_lines = []

    for line in lines:
        if line.strip().startswith("```"):
            if not code_block:
                code_block = True
                code_lines = []
            else:
                code_block = False

                if code_lines:
                    elements.append(Preformatted("\n".join(code_lines), styles["AgentCode"]))
                    elements.append(Spacer(1, 10))

                code_lines = []

            continue

        if code_block:
            code_lines.append(line)
            continue

        if not line.strip():
            elements.append(Spacer(1, 6))
            continue

        if line.startswith("# "):
            elements.append(Paragraph(clean_pdf_text(line[2:]), styles["AgentH1"]))
            continue

        if line.startswith("## "):
            elements.append(Paragraph(clean_pdf_text(line[3:]), styles["AgentH2"]))
            continue

        if line.startswith("### "):
            elements.append(Paragraph(clean_pdf_text(line[4:]), styles["AgentH3"]))
            continue

        if line.strip().startswith("- "):
            content = clean_pdf_text(line.strip()[2:])
            content = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", content)
            elements.append(Paragraph("• " + content, styles["AgentBody"]))
            continue

        numbered_match = re.match(r"^\s*(\d+)\.\s+(.*)", line)

        if numbered_match:
            number = numbered_match.group(1)
            content = clean_pdf_text(numbered_match.group(2))
            content = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", content)
            elements.append(Paragraph(f"{number}. {content}", styles["AgentBody"]))
            continue

        content = clean_pdf_text(line)
        content = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", content)
        content = re.sub(r"`([^`]+)`", r"<font name='AgentFontBold'>\1</font>", content)

        elements.append(Paragraph(content, styles["AgentBody"]))

    if code_block and code_lines:
        elements.append(Preformatted("\n".join(code_lines), styles["AgentCode"]))

    return elements


def create_pdf(title: str, content: str = "", conversation=None) -> bytes:
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=45, leftMargin=45,
        topMargin=50, bottomMargin=50, title=title, author="AI Agent"
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle("AgentTitle", fontName=PDF_FONT_BOLD, fontSize=20, leading=26, spaceAfter=10, alignment=TA_LEFT))
    styles.add(ParagraphStyle("AgentDate", fontName=PDF_FONT, fontSize=8, leading=12, spaceAfter=20))
    styles.add(ParagraphStyle("AgentH1", fontName=PDF_FONT_BOLD, fontSize=17, leading=22, spaceBefore=15, spaceAfter=10))
    styles.add(ParagraphStyle("AgentH2", fontName=PDF_FONT_BOLD, fontSize=14, leading=19, spaceBefore=13, spaceAfter=8))
    styles.add(ParagraphStyle("AgentH3", fontName=PDF_FONT_BOLD, fontSize=12, leading=17, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle("AgentRole", fontName=PDF_FONT_BOLD, fontSize=11, leading=16, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle("AgentBody", fontName=PDF_FONT, fontSize=10, leading=16, spaceAfter=6, alignment=TA_LEFT))
    styles.add(ParagraphStyle("AgentCode", fontName=PDF_FONT, fontSize=8, leading=11, leftIndent=10, rightIndent=10, spaceBefore=5, spaceAfter=10))

    story = [
        Paragraph(clean_pdf_text(title), styles["AgentTitle"]),
        Paragraph(f"Oluşturulma zamanı: {datetime.now().strftime('%d.%m.%Y %H:%M')}", styles["AgentDate"])
    ]

    if conversation:
        story.append(Paragraph("Konuşma Geçmişi", styles["AgentH1"]))

        for message in conversation:
            role_name = "Kullanıcı" if message.get("role") == "user" else "AI Agent"
            story.append(Paragraph(role_name, styles["AgentRole"]))
            story.extend(markdown_to_pdf_elements(message.get("text", ""), styles))
            story.append(Spacer(1, 8))
    else:
        story.extend(markdown_to_pdf_elements(content, styles))

    document.build(story)
    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# AGENT STREAM
# Agent cevabını parça parça Streamlit'e aktarır.
# =========================================================

def stream_response(prompt: str, config: dict):
    try:
        for chunk, metadata in agent.stream(
            {"messages": [HumanMessage(content=prompt)]},
            config,
            stream_mode="messages"
        ):
            if isinstance(chunk, AIMessage) and chunk.content:
                yield chunk.content

    except Exception as e:
        yield f"Agent çalışırken hata oluştu:\n\n{str(e)}"


# =========================================================
# SESSION STATE
# history içerisinde artık AI cevaplarının ses dosyalarını da tutuyoruz.
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
# HEADER
# =========================================================

st.title("🤖 Hafızalı AI Agent")
st.caption("LangChain + LangGraph + Groq + Tavily + PDF + Voice")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Agent")
    st.write("Model: `openai/gpt-oss-120b`")
    st.write("Memory: `InMemorySaver`")
    st.write("Web Search: `Tavily`")
    st.write("Speech-to-Text: `Whisper Large V3 Turbo`")
    st.write("Text-to-Speech: `gTTS Turkish`")

    st.divider()

    if st.button("🗑️ Yeni Sohbet", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()

    st.subheader("📄 PDF")

    if st.session_state.history:
        conversation_pdf = create_pdf(
            title="AI Agent Konuşma Geçmişi",
            conversation=st.session_state.history
        )

        st.download_button(
            label="📥 Tüm Konuşmayı PDF İndir",
            data=conversation_pdf,
            file_name=f"ai_agent_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# =========================================================
# CHAT HISTORY
# Önceki mesajları ve AI cevaplarının PDF / ses seçeneklerini gösterir.
# =========================================================

for index, msg in enumerate(st.session_state.history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

        if msg["role"] == "assistant":
            pdf_data = create_pdf(title="AI Agent Cevabı", content=msg["text"])

            st.download_button(
                label="📄 Bu cevabı PDF indir",
                data=pdf_data,
                file_name=f"agent_answer_{index}.pdf",
                mime="application/pdf",
                key=f"pdf_history_{index}"
            )

            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/mp3")

                st.download_button(
                    label="🎵 MP3 indir",
                    data=msg["audio"],
                    file_name=f"agent_answer_{index}.mp3",
                    mime="audio/mpeg",
                    key=f"audio_history_{index}"
                )


# =========================================================
# VOICE INPUT
#
# Kullanıcı buradan mikrofonla konuşur.
#
# ÖNEMLİ:
# Ses önce lokal olarak analiz edilir.
# Boş/gürültülü kayıt ise Whisper'a bile gönderilmez.
# Böylece gereksiz API maliyeti oluşmaz.
# =========================================================

st.subheader("🎤 Sesli Mesaj")

audio_input = st.audio_input("Konuşmak için mikrofona basın")


# =========================================================
# VOICE PROCESSING
# Ses geldiyse:
#
# 1. Lokal ses analizi
# 2. Gürültü azaltma
# 3. Boş ses kontrolü
# 4. Whisper
# 5. Yazıya dönüştürme
# 6. Agent
# 7. TTS
# =========================================================

if audio_input:

    audio_bytes = audio_input.getvalue()

    with st.spinner("🎧 Ses analiz ediliyor..."):
        has_speech, cleaned_audio, audio_error = analyze_audio_signal(audio_bytes)

    if not has_speech:
        st.warning(f"🔇 {audio_error}")

    else:
        with st.spinner("🗣️ Konuşma yazıya dönüştürülüyor..."):

            try:
                transcript = transcribe_audio(cleaned_audio)
            except Exception as e:
                transcript = ""
                st.error(f"Ses yazıya dönüştürülemedi: {str(e)}")

        # Whisper herhangi bir metin çıkaramadıysa AI'a gönderme.
        if not transcript or len(transcript.strip()) < 2:
            st.warning("🔇 Konuşma algılanamadı. AI'a istek gönderilmedi.")

        else:
            # Kullanıcının sesinden çıkan metni göster.
            st.chat_message("user").markdown(f"🎤 {transcript}")

            # Metni normal HumanMessage gibi agent'a gönder.
            st.session_state.history.append({
                "role": "user",
                "text": f"🎤 {transcript}"
            })

            config = {
                "configurable": {
                    "thread_id": "streamlit_session"
                }
            }

            with st.chat_message("assistant"):
                answer = st.write_stream(
                    stream_response(transcript, config)
                )

            # AI cevabını Türkçe sese çevir.
            try:
                audio_data = create_audio(answer)
            except Exception as e:
                audio_data = None
                st.warning(f"🔊 AI cevabı sese dönüştürülemedi: {str(e)}")

            st.session_state.history.append({
                "role": "assistant",
                "text": answer,
                "audio": audio_data
            })

            # PDF
            pdf_data = create_pdf(
                title="AI Agent Cevabı",
                content=answer
            )

            st.download_button(
                label="📄 Cevabı PDF İndir",
                data=pdf_data,
                file_name=f"agent_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                key=f"voice_pdf_{datetime.now().timestamp()}"
            )

            # AI ses cevabı
            if audio_data:
                st.audio(audio_data, format="audio/mp3")

                st.download_button(
                    label="🎵 Cevabı MP3 İndir",
                    data=audio_data,
                    file_name=f"agent_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                    mime="audio/mpeg",
                    key=f"voice_audio_{datetime.now().timestamp()}"
                )


# =========================================================
# TEXT INPUT
# Normal yazılı mesajlar için klasik chat input.
# =========================================================

if prompt := st.chat_input("Mesajınızı yazın..."):

    st.session_state.history.append({
        "role": "user",
        "text": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    config = {
        "configurable": {
            "thread_id": "streamlit_session"
        }
    }

    with st.chat_message("assistant"):
        answer = st.write_stream(
            stream_response(prompt, config)
        )

    try:
        audio_data = create_audio(answer)
    except Exception as e:
        audio_data = None
        st.warning(f"🔊 Ses oluşturulamadı: {str(e)}")

    st.session_state.history.append({
        "role": "assistant",
        "text": answer,
        "audio": audio_data
    })

    # PDF
    pdf_data = create_pdf(
        title="AI Agent Cevabı",
        content=answer
    )

    st.download_button(
        label="📄 Cevabı PDF İndir",
        data=pdf_data,
        file_name=f"agent_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        key=f"pdf_new_{datetime.now().timestamp()}"
    )

    # Ses
    if audio_data:
        st.audio(audio_data, format="audio/mp3")

        st.download_button(
            label="🎵 Cevabı MP3 İndir",
            data=audio_data,
            file_name=f"agent_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
            mime="audio/mpeg",
            key=f"audio_new_{datetime.now().timestamp()}"
        )