import os
import tempfile
import hashlib

import streamlit as st

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool

from faster_whisper import WhisperModel
from streamlit_mic_recorder import mic_recorder


# =========================================================
# SAYFA AYARLARI
# =========================================================

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="centered"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

/* ========================================================
   GENEL
======================================================== */

.stApp {
    background: #ffffff;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}


/* ========================================================
   BAŞLIK
======================================================== */

.main-title {
    text-align: center;
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.7px;
    margin-bottom: 4px;
}

.subtitle {
    text-align: center;
    color: #888;
    font-size: 14px;
    margin-bottom: 30px;
}


/* ========================================================
   CHAT MESAJLARI
======================================================== */

[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 10px 15px;
    margin-bottom: 10px;
}


/* Kullanıcı */

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-user"]
) {
    background: #f3f4f6;
}


/* Assistant */

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-assistant"]
) {
    background: #ffffff;
}


/* ========================================================
   DÜZENLE BUTONU
======================================================== */

.edit-area {
    margin-top: 3px;
}

.edit-area button {
    font-size: 12px !important;
    padding: 2px 8px !important;
    min-height: 28px !important;
    border-radius: 8px !important;
}


/* ========================================================
   DÜZENLEME PANELİ
======================================================== */

.edit-title {
    font-size: 18px;
    font-weight: 600;
    margin-top: 25px;
    margin-bottom: 10px;
}


/* ========================================================
   SES ALANI
======================================================== */

.voice-section {
    margin-top: 10px;
    margin-bottom: 8px;
}

.voice-title {
    text-align: center;
    color: #999;
    font-size: 12px;
    margin-bottom: 5px;
}


/* ========================================================
   BUTONLAR
======================================================== */

button {
    border-radius: 10px !important;
}


/* ========================================================
   FOOTER
======================================================== */

.footer-text {
    text-align: center;
    color: #aaa;
    font-size: 12px;
    margin-top: 30px;
}


/* ========================================================
   KOLOM BOŞLUKLARI
======================================================== */

[data-testid="column"] {
    padding: 0 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PSİKOLOJİ BİLGİ TOOL'U
# =========================================================

@tool
def psychology_knowledge(topic: str) -> str:
    """
    Psikolojiyle ilgili kavramlar hakkında referans bilgi sağlar.
    """

    knowledge = {

        "bağlanma": """
Bağlanma kuramı, insanların yakın ilişkilerde nasıl bağ
kurduğunu, yakınlık ve ayrılık durumlarına nasıl tepki
verdiğini anlamaya yönelik psikolojik bir çerçevedir.

Literatürde güvenli, kaygılı ve kaçıngan bağlanma gibi
örüntülerden söz edilir.

Bu kavramlar kişinin davranışlarını anlamak için kullanılabilir;
ancak tek başına klinik tanı anlamına gelmez.
""",

        "kaygılı bağlanma": """
Kaygılı bağlanma örüntüsünde kişi ilişkilerde reddedilme,
terk edilme veya yeterince sevilmeme konusunda daha fazla
kaygı yaşayabilir.

Yakınlık ihtiyacı ile ilişkinin kaybedileceğine dair endişe
aynı anda görülebilir.

Ancak tek bir davranış kişinin bağlanma stilini kesin olarak
belirlemez.
""",

        "kaçıngan bağlanma": """
Kaçıngan bağlanma örüntüsünde kişi yakın ilişkilerde aşırı
yakınlıktan rahatsız olabilir ve bağımsızlığa daha fazla
önem verebilir.

Bu durum kişiden kişiye farklı biçimlerde ortaya çıkabilir.
""",

        "yalnızlık": """
Yalnızlık yalnızca fiziksel olarak yalnız olmak değildir.

Kişinin sahip olduğu sosyal ilişkiler ile ihtiyaç duyduğu
veya arzuladığı ilişkiler arasındaki algılanan farkla da
ilişkili olabilir.

Sosyal izolasyon ve yalnızlık aynı kavram değildir.
""",

        "ego": """
Ego kavramı farklı psikoloji ekollerinde farklı anlamlarda
kullanılır.

Freud'un yapısal kuramında ego; id'in dürtüleri,
süperegonun talepleri ve dış gerçekliğin koşulları arasında
düzenleme yapan yapı olarak ele alınır.

Günlük dilde kullanılan ego kavramı bu teknik anlamdan
farklı olabilir.
""",

        "travma": """
Travma, kişinin ciddi tehdit, korku, çaresizlik veya
başa çıkma kapasitesini aşan deneyimlerle ilişkili
psikolojik etkilerini ifade etmek için kullanılabilir.

Travmatik bir deneyim ile travma sonrası bir ruhsal
bozukluk aynı şey değildir.
""",

        "savunma mekanizmaları": """
Savunma mekanizmaları, kişinin rahatsız edici duygu,
düşünce veya çatışmalarla başa çıkmasında rol oynayabilen
psikolojik süreçlerdir.

Bastırma, yansıtma, inkâr, rasyonalizasyon ve yer değiştirme
gibi farklı kavramlardan söz edilir.

Bunların kullanılması tek başına psikolojik bir bozukluk
anlamına gelmez.
"""
    }

    topic = topic.lower().strip()

    return knowledge.get(
        topic,
        f"""
'{topic}' konusu için yerel psikoloji bilgi tabanında
özel bir kayıt bulunamadı.

Genel psikoloji bilgisiyle açıklama yapabilirsin.
Kesin psikolojik teşhis koyma.
"""
    )


# =========================================================
# WHISPER
# =========================================================

@st.cache_resource
def load_whisper():

    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )


def transcribe_audio(audio_bytes):

    whisper = load_whisper()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_file:

        temp_file.write(audio_bytes)
        temp_path = temp_file.name

    try:

        segments, info = whisper.transcribe(
            temp_path,
            language="tr",
            vad_filter=True
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        )

        return text.strip()

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# =========================================================
# GENEL AI AGENT
# =========================================================

@st.cache_resource
def create_agent_with_ollama():

    llm = ChatOllama(
        model="qwen3:1.7b",
        temperature=0
    )

    return create_agent(

        model=llm,

        tools=[
            psychology_knowledge
        ],

        system_prompt="""
Sen genel amaçlı, bilgili ve yardımcı bir AI Agent'sın.

Kullanıcının sorularını konu sınırlaması olmadan cevapla.

Yazılım, Python, C#, .NET, veritabanları, yapay zeka,
LLM, Agent sistemleri, LangChain, teknoloji, bilim,
matematik, tarih, günlük yaşam ve diğer genel konularda
yardımcı olabilirsin.

PSİKOLOJİ:

Psikoloji sorularında daha derin, kapsamlı ve kavramsal
cevaplar vermeye çalış.

Bağlanma, yalnızlık, travma, kişilik, duygular,
motivasyon, bilişsel süreçler, davranış, ilişkiler,
savunma mekanizmaları ve benzeri konuları açıklayabilirsin.

Psikoloji sorularında:

- Kavramları açıkla.
- Gerekirse farklı psikolojik yaklaşımları karşılaştır.
- Örnekler kullan.
- Bir davranışın birden fazla olası açıklaması
  olabileceğini belirt.
- Kullanıcının anlattığı durumdan kesin teşhis çıkarma.
- Klinik tanı koyma.
- Gerektiğinde belirsizliği belirt.

PSİKOLOJİ DIŞINDA:

Soruyu psikolojiye bağlamaya çalışma.

Kullanıcı Python soruyorsa Python,
Agent soruyorsa Agent,
LangChain soruyorsa LangChain,
veritabanı soruyorsa veritabanı hakkında cevap ver.

AGENT DAVRANIŞI:

Bir tool kullanmak gerekiyorsa uygun tool'u kullan.

Tool sonucunu aldıktan sonra sonucu değerlendir ve
kullanıcıya doğal bir cevap oluştur.

Tool kullanmayı gerektirmeyen sorularda doğrudan cevap ver.

Kullanıcının dilini takip et.
Türkçe sorulara Türkçe cevap ver.

Cevapları anlaşılır ve doğal tut.
Basit soruları gereksiz yere uzatma.
Karmaşık sorularda ise yeterli teknik ayrıntı ver.

Sen yalnızca psikoloji agent'ı değilsin.
Genel amaçlı bir AI Agent'sın.
"""
    )


agent = create_agent_with_ollama()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_audio_hash" not in st.session_state:
    st.session_state.processed_audio_hash = None

if "editing_message" not in st.session_state:
    st.session_state.editing_message = None


# =========================================================
# BAŞLIK
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🤖 AI Agent'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ollama · Qwen3 · LangChain · Whisper'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MESAJLARI GÖSTER
# =========================================================

for index, message in enumerate(
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # -------------------------------------------------
        # SADECE KULLANICI MESAJLARINDA DÜZENLE
        # -------------------------------------------------

        if message["role"] == "user":

            if st.button(
                "✏️ Düzenle",
                key=f"edit_message_{index}"
            ):

                st.session_state.editing_message = index

                st.rerun()


# =========================================================
# DÜZENLEME MODU
# =========================================================

if st.session_state.editing_message is not None:

    edit_index = (
        st.session_state.editing_message
    )

    old_text = (
        st.session_state.messages[
            edit_index
        ]["content"]
    )

    st.markdown(
        '<div class="edit-title">'
        '✏️ Mesajı düzenle'
        '</div>',
        unsafe_allow_html=True
    )

    edited_text = st.text_area(
        "Mesaj",
        value=old_text,
        height=110,
        key=f"edit_text_{edit_index}",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # İPTAL
    # -----------------------------------------------------

    with col1:

        if st.button(
            "İptal",
            use_container_width=True,
            key="cancel_edit"
        ):

            st.session_state.editing_message = None

            st.rerun()

    # -----------------------------------------------------
    # YENİDEN GÖNDER
    # -----------------------------------------------------

    with col2:

        if st.button(
            "↻ Yeniden Gönder",
            use_container_width=True,
            key="resend_edit"
        ):

            edited_text = edited_text.strip()

            if edited_text:

                # -----------------------------------------
                # MESAJI DEĞİŞTİR
                # -----------------------------------------

                st.session_state.messages[
                    edit_index
                ]["content"] = edited_text

                # -----------------------------------------
                # SONRAKİ MESAJLARI SİL
                # -----------------------------------------

                st.session_state.messages = (
                    st.session_state.messages[
                        :edit_index + 1
                    ]
                )

                # -----------------------------------------
                # AGENT'I TEKRAR ÇALIŞTIR
                # -----------------------------------------

                with st.spinner(
                    "🤖 Yeniden düşünüyor..."
                ):

                    response = agent.invoke({

                        "messages":
                            st.session_state.messages

                    })

                    output = response[
                        "messages"
                    ][-1].content

                # -----------------------------------------
                # YENİ CEVABI EKLE
                # -----------------------------------------

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": output

                })

                # -----------------------------------------
                # EDIT MODUNU KAPAT
                # -----------------------------------------

                st.session_state.editing_message = None

                st.rerun()


# =========================================================
# NORMAL CHAT
# =========================================================

if st.session_state.editing_message is None:

    # =====================================================
    # SESLİ GİRİŞ
    # =====================================================

    st.markdown(
        """
        <div class="voice-section">
            <div class="voice-title">
                🎤 Sesli mesaj
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    audio = mic_recorder(
        start_prompt="🎤 Konuş",
        stop_prompt="⏹️ Durdur",
        just_once=True,
        use_container_width=True,
        key="voice_recorder"
    )


    # =====================================================
    # SESİ İŞLE
    # =====================================================

    if audio:

        audio_bytes = audio.get("bytes")

        if audio_bytes:

            current_audio_hash = hashlib.sha256(
                audio_bytes
            ).hexdigest()

            # Aynı ses kaydının tekrar çalışmasını engelle
            if (
                current_audio_hash
                != st.session_state.processed_audio_hash
            ):

                st.session_state.processed_audio_hash = (
                    current_audio_hash
                )

                with st.spinner(
                    "🎧 Ses yazıya çevriliyor..."
                ):

                    try:

                        voice_text = transcribe_audio(
                            audio_bytes
                        )

                        voice_text = voice_text.strip()

                        if voice_text:

                            # ---------------------------------
                            # USER MESAJI
                            # ---------------------------------

                            st.session_state.messages.append({

                                "role": "user",

                                "content": voice_text

                            })

                            with st.chat_message("user"):

                                st.markdown(
                                    voice_text
                                )


                            # ---------------------------------
                            # AGENT
                            # ---------------------------------

                            with st.chat_message(
                                "assistant"
                            ):

                                with st.spinner(
                                    "🤖 Düşünüyor..."
                                ):

                                    response = agent.invoke({

                                        "messages":
                                            st.session_state.messages

                                    })

                                    output = response[
                                        "messages"
                                    ][-1].content

                                st.markdown(output)


                            # ---------------------------------
                            # MEMORY
                            # ---------------------------------

                            st.session_state.messages.append({

                                "role": "assistant",

                                "content": output

                            })


                    except Exception as e:

                        st.error(
                            f"Ses işlenirken hata oluştu: {e}"
                        )


    # =====================================================
    # CHAT INPUT
    # =====================================================

    prompt = st.chat_input(
        "Mesajını yaz..."
    )


    # =====================================================
    # YAZILI MESAJ
    # =====================================================

    if prompt:

        prompt = prompt.strip()

        if prompt:

            # ---------------------------------------------
            # USER
            # ---------------------------------------------

            st.session_state.messages.append({

                "role": "user",

                "content": prompt

            })

            with st.chat_message("user"):

                st.markdown(prompt)


            # ---------------------------------------------
            # AGENT
            # ---------------------------------------------

            with st.chat_message("assistant"):

                with st.spinner(
                    "🤖 Düşünüyor..."
                ):

                    response = agent.invoke({

                        "messages":
                            st.session_state.messages

                    })

                    output = response[
                        "messages"
                    ][-1].content

                st.markdown(output)


            # ---------------------------------------------
            # MEMORY
            # ---------------------------------------------

            st.session_state.messages.append({

                "role": "assistant",

                "content": output

            })


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer-text">'
    'Local AI · Qwen3 1.7B · LangChain · Whisper'
    '</div>',
    unsafe_allow_html=True
)