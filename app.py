import streamlit as st
import os
from datetime import date, datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MindEase · Nara",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════
# GLOBAL CSS — Calm Forest · Organic Minimal
# Font: Fraunces (display) + Plus Jakarta Sans (body)
# ════════════════════════════════════════════════════════════
st.markdown(
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;1,300;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>

/* ── Root palette ── */
:root {
  --teal:        #2D8A7A;
  --teal-dark:   #1A5C52;
  --teal-lt:     #E0F5F1;
  --sage:        #5C7A6B;
  --sage-mid:    #B8D4C2;
  --sage-lt:     #EAF2ED;
  --cream:       #FAF8F3;
  --warm:        #F2EDE4;
  --text:        #1E2D27;
  --text-muted:  #6B8076;
  --gold:        #C8956A;
  --gold-lt:     #FBF0E8;
  --danger:      #B84A3A;
  --danger-lt:   #FDF2F0;
  --white:       #FFFFFF;
  --border:      rgba(92,122,107,0.15);
  --shadow:      0 8px 40px rgba(26,92,82,0.08);
  --shadow-sm:   0 2px 12px rgba(26,92,82,0.06);
}

/* ── Base ── */
html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  color: var(--text) !important;
}
.stApp {
  background: var(--cream) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--white) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 0 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 4px 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--sage-mid); border-radius: 4px; }

/* ── Input box ── */
[data-testid="stChatInput"] {
  border-radius: 20px !important;
  border: 1.5px solid var(--sage-mid) !important;
  background: var(--white) !important;
  box-shadow: var(--shadow-sm) !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px rgba(45,138,122,0.12) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--teal) !important; }

/* ── Buttons ── */
.stButton > button {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 500 !important;
  border-radius: 12px !important;
  transition: all .2s ease !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--teal-dark), var(--sage)) !important;
  border: none !important;
  color: white !important;
  box-shadow: 0 3px 12px rgba(45,138,122,0.28) !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 5px 18px rgba(45,138,122,0.38) !important;
}

/* ── Select slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--teal) !important;
  border-color: var(--teal) !important;
}

/* ── Success / info ── */
.stSuccess {
  background: var(--sage-lt) !important;
  border-left-color: var(--teal) !important;
  color: var(--teal-dark) !important;
}

/* ── Title area ── */
.main-header {
  background: linear-gradient(135deg, var(--teal-dark) 0%, var(--sage) 100%);
  padding: 24px 40px 20px;
  position: relative;
  overflow: hidden;
}
.main-header::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 200px; height: 200px;
  background: rgba(255,255,255,0.05);
  border-radius: 50%;
}
.main-header::after {
  content: '';
  position: absolute;
  bottom: -40px; left: 40%;
  width: 140px; height: 140px;
  background: rgba(255,255,255,0.04);
  border-radius: 50%;
}

/* ── Chat bubbles ── */
.bubble-user {
  background: linear-gradient(135deg, var(--teal) 0%, var(--sage) 100%);
  color: white;
  border-radius: 18px 18px 4px 18px;
  padding: 12px 18px;
  font-size: 14px;
  line-height: 1.7;
  box-shadow: 0 3px 12px rgba(45,138,122,0.22);
  max-width: 78%;
  margin-left: auto;
  font-family: 'Plus Jakarta Sans', sans-serif;
}
.bubble-nara {
  background: var(--white);
  color: var(--text);
  border-radius: 18px 18px 18px 4px;
  padding: 12px 18px;
  font-size: 14px;
  line-height: 1.7;
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--border);
  max-width: 82%;
  font-family: 'Plus Jakarta Sans', sans-serif;
}
.bubble-crisis {
  background: var(--danger-lt);
  border: 1px solid rgba(184,74,58,0.2);
  color: var(--danger);
  border-radius: 18px 18px 18px 4px;
  padding: 12px 18px;
  font-size: 14px;
  line-height: 1.7;
  max-width: 82%;
  font-family: 'Plus Jakarta Sans', sans-serif;
}
.bubble-ts {
  font-size: 10px;
  color: var(--text-muted);
  margin-top: 4px;
  font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Welcome card ── */
.welcome-card {
  background: var(--white);
  border-radius: 20px;
  border: 1px solid var(--border);
  padding: 40px;
  text-align: center;
  box-shadow: var(--shadow);
  max-width: 560px;
  margin: 40px auto;
}
.welcome-icon {
  font-size: 52px;
  margin-bottom: 16px;
  display: block;
}
.welcome-title {
  font-family: 'Fraunces', serif;
  font-style: italic;
  font-size: 28px;
  color: var(--teal-dark);
  margin-bottom: 8px;
  line-height: 1.2;
}
.welcome-sub {
  font-size: 14px;
  color: var(--text-muted);
  line-height: 1.7;
  margin-bottom: 24px;
  font-weight: 300;
}
.starter-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 20px;
}
.starter-chip {
  background: var(--sage-lt);
  color: var(--teal-dark);
  border: 1.5px solid var(--sage-mid);
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  text-align: left;
  font-family: 'Plus Jakarta Sans', sans-serif;
  transition: all .18s ease;
}
.starter-chip:hover {
  background: var(--teal);
  color: white;
  border-color: var(--teal);
}

/* ── Sidebar sections ── */
.sidebar-header {
  background: linear-gradient(135deg, var(--teal-dark), var(--sage));
  padding: 20px 20px 16px;
  position: relative;
  overflow: hidden;
}
.sidebar-header::before {
  content: '';
  position: absolute;
  top: -30px; right: -30px;
  width: 100px; height: 100px;
  background: rgba(255,255,255,0.07);
  border-radius: 50%;
}
.sidebar-section {
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
}
.sidebar-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--sage);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 12px;
  font-family: 'Plus Jakarta Sans', sans-serif;
}
.stat-row {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}
.stat-card {
  flex: 1;
  background: var(--sage-lt);
  border-radius: 12px;
  padding: 12px 10px;
  text-align: center;
  border: 1px solid var(--sage-mid);
}
.stat-val {
  font-size: 20px;
  font-weight: 600;
  color: var(--teal-dark);
  font-family: 'Plus Jakarta Sans', sans-serif;
  line-height: 1;
}
.stat-lbl {
  font-size: 10px;
  color: var(--text-muted);
  margin-top: 3px;
  font-family: 'Plus Jakarta Sans', sans-serif;
}
.insight-box {
  background: var(--white);
  border-radius: 12px;
  border-left: 3px solid var(--teal);
  padding: 10px 14px;
  font-size: 12.5px;
  color: var(--text);
  line-height: 1.6;
  font-style: italic;
  margin-top: 10px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--teal);
}
.disclaimer-box {
  background: var(--danger-lt);
  border-left: 3px solid var(--danger);
  padding: 10px 14px;
  border-radius: 0 10px 10px 0;
  font-size: 12px;
  color: var(--danger);
  line-height: 1.6;
  font-family: 'Plus Jakarta Sans', sans-serif;
}
.dot-pulse {
  display: inline-block;
  width: 7px; height: 7px;
  background: #7AE6A0;
  border-radius: 50%;
  margin-right: 6px;
  animation: pulse 2s ease infinite;
}
@keyframes pulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:0.6; transform:scale(1.35); }
}
</style>
, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════
MOOD_LEVELS = {
    1: ("Sangat Buruk", "😞", "#E24B4A"),
    2: ("Kurang Baik",  "😔", "#EF9F27"),
    3: ("Biasa Saja",   "😐", "#888780"),
    4: ("Cukup Baik",   "🙂", "#5DCAA5"),
    5: ("Sangat Baik",  "😊", "#1D9E75"),
}

CRISIS_KEYWORDS = [
    "bunuh diri", "menyakiti diri", "tidak mau hidup",
    "ingin mati", "mati saja", "sudah tidak kuat", "hopeless"
]

STARTER_PROMPTS = [
    "😔 Aku lagi stres banget", "😰 Aku sering cemas belakangan ini",
    "😴 Aku susah tidur terus",  "💭 Aku butuh teman ngobrol",
    "🧠 Ajari aku teknik relaksasi", "💚 Apa itu kesehatan mental?"
]

# ════════════════════════════════════════════════════════════
# API KEY
# ════════════════════════════════════════════════════════════
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    api_key = os.environ.get("GOOGLE_API_KEY", "")

if not api_key:
    st.error("❌ GOOGLE_API_KEY tidak ditemukan! Tambahkan di Streamlit Secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# ════════════════════════════════════════════════════════════
# RAG INIT (cached)
# ════════════════════════════════════════════════════════════
@st.cache_resource
def init_rag():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    # Build ChromaDB dari knowledge_base/ jika belum ada
    chroma_path = "./chroma_db"
    if not os.path.exists(chroma_path) or not os.listdir(chroma_path):
        with st.spinner("⏳ Membangun knowledge base... (hanya sekali)"):
            loader = DirectoryLoader(
                "knowledge_base/",
                glob="*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50,
                separators=["\n\n", "\n", ".", " "]
            )
            chunks = splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=chroma_path
            )
    else:
        vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        max_output_tokens=1024,
        convert_system_message_to_human=True,
        google_api_key=api_key
    )

    qa_prompt = PromptTemplate(
        template=
Kamu adalah Nara, teman virtual yang hangat dan empatik dari MindEase.
Berbicara santai dalam Bahasa Indonesia, seperti teman sebaya yang peduli.

Konteks dari knowledge base:
---------------------
{context}
---------------------

Pertanyaan: {question}

Panduan menjawab:
1. Validasi perasaan pengguna terlebih dahulu
2. Jawab dengan informatif menggunakan konteks di atas
3. Berikan saran praktis yang bisa langsung dicoba
4. Tutup dengan hangat dan ajukan pertanyaan ringan
5. JANGAN mendiagnosa kondisi mental
6. Jika krisis, sertakan hotline: 119 ext 8

Jawaban Nara:,
        input_variables=["context", "question"]
    )

    return llm, retriever, qa_prompt

llm, retriever, qa_prompt = init_rag()

# ════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════
if "messages"     not in st.session_state: st.session_state.messages     = []
if "mood_history" not in st.session_state: st.session_state.mood_history = {}
if "memory"       not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=20, memory_key="chat_history",
        return_messages=True, output_key="answer"
    )

# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════
def get_mood_context() -> str:
    today = str(date.today())
    if today in st.session_state.mood_history:
        m = st.session_state.mood_history[today]
        return f"[Mood hari ini: {m['emoji']} {m['label']}] "
    return ""

def get_weekly_insight() -> str:
    today = date.today()
    week  = [str(today - timedelta(days=i)) for i in range(6, -1, -1)]
    data  = {d: st.session_state.mood_history[d] for d in week if d in st.session_state.mood_history}
    if not data:
        return ""
    avg = sum(v["level"] for v in data.values()) / len(data)
    if avg >= 4.0: return f"Minggu yang luar biasa! Rata-rata mood {avg:.1f}/5 💚"
    if avg >= 3.0: return f"Minggu cukup baik. Rata-rata mood {avg:.1f}/5 🌿"
    if avg >= 2.0: return f"Minggu yang lumayan berat. Rata-rata {avg:.1f}/5 💙"
    return f"Minggu yang sangat berat. Rata-rata {avg:.1f}/5 — pertimbangkan bicara dengan profesional 🤝"

def generate_response(prompt: str) -> str:
    if any(kw in prompt.lower() for kw in CRISIS_KEYWORDS):
        return (
            "Hey, aku sangat khawatir mendengar itu. 💙\n\n"
            "Kamu tidak sendirian dan perasaanmu sangat valid.\n\n"
            "Tolong segera hubungi:\n"
            "📞 **Hotline Kemenkes: 119 ext 8** (24 jam, gratis)\n"
            "💬 Into The Light: intothelightid.org\n\n"
            "Aku di sini bersamamu. Mau cerita lebih?"
        )
    mood_ctx = get_mood_context()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=False,
        verbose=False
    )
    result = chain.invoke({"question": mood_ctx + prompt})
    return result["answer"]

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:

    # Header
    st.markdown(
    <div class='sidebar-header'>
      <div style='display:flex; align-items:center; gap:12px; position:relative;'>
        <div style='width:42px; height:42px; background:rgba(255,255,255,0.15);
                    border-radius:50%; display:flex; align-items:center;
                    justify-content:center; font-size:20px;
                    border:2px solid rgba(255,255,255,0.25);'>🌿</div>
        <div>
          <div style='font-family:"Fraunces",serif; font-style:italic; font-size:18px;
                      color:white; line-height:1;'>MindEase</div>
          <div style='font-size:11px; color:rgba(255,255,255,0.7); margin-top:3px;
                      font-family:"Plus Jakarta Sans",sans-serif; font-weight:300;'>
            <span class='dot-pulse'></span>Nara siap menemanimu
          </div>
        </div>
      </div>
    </div>
    , unsafe_allow_html=True)

    # Mood Tracker
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-label'>🎭 Mood Hari Ini</div>", unsafe_allow_html=True)

    mood_level = st.select_slider(
        "Perasaanmu hari ini:",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{MOOD_LEVELS[x][1]}  {MOOD_LEVELS[x][0]}",
        label_visibility="collapsed"
    )

    mood_note = st.text_input(
        "Catatan",
        placeholder="e.g. kurang tidur, habis olahraga...",
        label_visibility="collapsed"
    )

    if st.button("💾 Simpan Mood", use_container_width=True, type="primary"):
        today = str(date.today())
        st.session_state.mood_history[today] = {
            "level": mood_level,
            "label": MOOD_LEVELS[mood_level][0],
            "emoji": MOOD_LEVELS[mood_level][1],
            "catatan": mood_note,
            "waktu": datetime.now().strftime("%H:%M")
        }
        st.success(f"{MOOD_LEVELS[mood_level][1]} Tersimpan — {MOOD_LEVELS[mood_level][0]}!")

    st.markdown("</div>", unsafe_allow_html=True)

    # Grafik mood
    if st.session_state.mood_history:
        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-label'>📊 Tren 7 Hari</div>", unsafe_allow_html=True)

        tanggals = sorted(st.session_state.mood_history.keys())[-7:]
        levels   = [st.session_state.mood_history[t]["level"] for t in tanggals]
        colors   = [MOOD_LEVELS[l][2] for l in levels]
        avg      = sum(levels) / len(levels)

        # Stat cards
        today_mood = st.session_state.mood_history.get(str(date.today()), {})
        today_emoji = today_mood.get("emoji", "—")
        st.markdown(f
        <div class='stat-row'>
          <div class='stat-card'>
            <div class='stat-val'>{avg:.1f}</div>
            <div class='stat-lbl'>rata-rata</div>
          </div>
          <div class='stat-card'>
            <div class='stat-val'>{len(tanggals)}</div>
            <div class='stat-lbl'>hari tercatat</div>
          </div>
          <div class='stat-card'>
            <div class='stat-val'>{today_emoji}</div>
            <div class='stat-lbl'>hari ini</div>
          </div>
        </div>
        , unsafe_allow_html=True)

        # Mini chart
        fig, ax = plt.subplots(figsize=(3.8, 2.2))
        fig.patch.set_facecolor("#EAF2ED")
        ax.set_facecolor("#EAF2ED")

        ax.bar(range(len(tanggals)), levels, color=colors, width=0.55, alpha=0.75, zorder=2)
        ax.plot(range(len(tanggals)), levels, "o-",
                color="#1A5C52", linewidth=2, markersize=5,
                markerfacecolor="white", markeredgewidth=2, markeredgecolor="#1A5C52", zorder=3)
        ax.axhline(y=avg, color="#5C7A6B", linestyle="--", linewidth=1, alpha=0.6)

        ax.set_xticks(range(len(tanggals)))
        ax.set_xticklabels([t[8:] for t in tanggals], fontsize=7.5, color="#2C3E35")
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["😞","😔","😐","🙂","😊"], fontsize=9)
        ax.set_ylim(0.5, 5.5)
        ax.grid(axis="y", color="#B8D4C2", alpha=0.5, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color("#B8D4C2")
            spine.set_linewidth(0.6)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig)
        plt.close()

        # Weekly insight
        insight = get_weekly_insight()
        if insight:
            st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown(
    <div style='padding:16px 20px;'>
      <div class='disclaimer-box'>
        ⚠️ <strong>MindEase</strong> adalah alat bantu edukasi, bukan pengganti
        psikolog profesional.<br><br>
        Darurat: <strong>119 ext 8</strong>
      </div>
    </div>
    , unsafe_allow_html=True)

    # Reset chat
    st.markdown("<div style='padding:0 20px 20px;'>", unsafe_allow_html=True)
    if st.button("🗑️ Reset Percakapan", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory   = ConversationBufferWindowMemory(
            k=20, memory_key="chat_history",
            return_messages=True, output_key="answer"
        )
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════

# Header bar
st.markdown(
<div class='main-header'>
  <div style='display:flex; align-items:center; gap:16px; position:relative;'>
    <div style='width:50px; height:50px; background:rgba(255,255,255,0.15);
                border-radius:50%; display:flex; align-items:center;
                justify-content:center; font-size:24px;
                border:2px solid rgba(255,255,255,0.25); flex-shrink:0;'>🌿</div>
    <div>
      <div style='font-family:"Fraunces",serif; font-style:italic; font-size:26px;
                  color:white; line-height:1.1;'>MindEase</div>
      <div style='font-size:13px; color:rgba(255,255,255,0.72); margin-top:3px;
                  font-family:"Plus Jakarta Sans",sans-serif; font-weight:300;'>
        <span class='dot-pulse'></span>
        Halo! Aku <strong style='font-weight:600;'>Nara</strong>,
        teman virtual kesehatan mentalmu 💙
      </div>
    </div>
    <div style='margin-left:auto; display:flex; gap:8px; position:relative;'>
      <span style='background:rgba(255,255,255,0.12); color:rgba(255,255,255,0.82);
                   border:1px solid rgba(255,255,255,0.2); border-radius:999px;
                   padding:4px 12px; font-size:11px;
                   font-family:"Plus Jakarta Sans",sans-serif;'>💬 Curhat bebas</span>
      <span style='background:rgba(255,255,255,0.12); color:rgba(255,255,255,0.82);
                   border:1px solid rgba(255,255,255,0.2); border-radius:999px;
                   padding:4px 12px; font-size:11px;
                   font-family:"Plus Jakarta Sans",sans-serif;'>📚 Berbasis RAG</span>
    </div>
  </div>
</div>
, unsafe_allow_html=True)

# Chat container
chat_container = st.container()

with chat_container:
    # Welcome screen jika belum ada pesan
    if not st.session_state.messages:
        chips_html = "".join([
            f"<div class='starter-chip'>{s}</div>"
            for s in STARTER_PROMPTS
        ])
        st.markdown(f
        <div class='welcome-card'>
          <span class='welcome-icon'>🌿</span>
          <div class='welcome-title'>Halo, aku Nara!</div>
          <div class='welcome-sub'>
            Teman virtual kesehatan mentalmu di MindEase.<br>
            Aku di sini untuk mendengarkan dan menemanimu — tanpa menghakimi. 💙
          </div>
          <div style='font-family:"Plus Jakarta Sans",sans-serif; font-size:12px;
                      font-weight:600; color:#5C7A6B; text-transform:uppercase;
                      letter-spacing:0.05em; margin-bottom:12px;'>
            Mulai dari sini:
          </div>
          <div class='starter-grid'>{chips_html}</div>
        </div>
        , unsafe_allow_html=True)
    else:
        # Tampilkan riwayat chat
        for msg in st.session_state.messages:
            ts = msg.get("time", "")
            if msg["role"] == "user":
                col1, col2 = st.columns([1, 3])
                with col2:
                    st.markdown(
                        f"<div class='bubble-user'>{msg['content']}</div>"
                        f"<div class='bubble-ts' style='text-align:right;'>{ts}</div>",
                        unsafe_allow_html=True
                    )
            else:
                col1, col2 = st.columns([3, 1])
                with col1:
                    is_crisis = any(kw in msg["content"].lower() for kw in CRISIS_KEYWORDS)
                    cls = "bubble-crisis" if is_crisis else "bubble-nara"
                    content = msg["content"].replace("\n", "<br>")
                    st.markdown(
                        f"<div class='{cls}'>{content}</div>"
                        f"<div class='bubble-ts'>{ts}</div>",
                        unsafe_allow_html=True
                    )

# Input
st.markdown("<div style='padding:12px 0;'></div>", unsafe_allow_html=True)

if prompt := st.chat_input("Ceritakan apa yang kamu rasakan..."):
    ts_now = datetime.now().strftime("%H:%M")

    # Simpan pesan user
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "time": ts_now
    })

    # Generate respons
    with st.spinner("Nara sedang mengetik..."):
        try:
            response = generate_response(prompt)
        except Exception as e:
            response = f"Maaf, ada gangguan teknis. Coba lagi ya! 🙏"

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "time": datetime.now().strftime("%H:%M")
    })

    st.rerun()
