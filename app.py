
import streamlit as st
import os
from datetime import date, datetime
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import matplotlib.pyplot as plt

# ---- Konfigurasi Halaman ----
st.set_page_config(
    page_title="MindEase — Teman Curhat Kesehatan Mental",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Styling Custom ----
st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .chat-message-user {
        background: #EEEDFE; border-radius: 18px 18px 4px 18px;
        padding: 12px 16px; margin: 8px 0; color: #3C3489;
    }
    .chat-message-minda {
        background: #E1F5EE; border-radius: 18px 18px 18px 4px;
        padding: 12px 16px; margin: 8px 0; color: #085041;
    }
    .disclaimer {
        background: #FAEEDA; border-left: 4px solid #EF9F27;
        padding: 10px 14px; border-radius: 0 8px 8px 0;
        font-size: 13px; color: #633806;
    }
</style>
""", unsafe_allow_html=True)

# ---- API Key ----
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    api_key = os.environ.get("GOOGLE_API_KEY", "")

if not api_key:
    st.error("❌ GOOGLE_API_KEY tidak ditemukan! Tambahkan di Streamlit Secrets.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# ---- Inisialisasi RAG (cached agar tidak reload tiap interaksi) ----
@st.cache_resource
def init_rag():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0.7,
        max_output_tokens=1024,
        convert_system_message_to_human=True
    )

    QA_PROMPT = PromptTemplate(
        template="""
Kamu adalah Minda, teman virtual hangat dari MindEase. Berbicara santai dalam Bahasa Indonesia.

Konteks dari knowledge base:
---------------------
{context}
---------------------

Pertanyaan: {question}

Jawab dengan: validasi perasaan → informasi berguna → saran praktis → tutup hangat.
JANGAN mendiagnosa. Jika krisis, berikan hotline 119 ext 8.

Jawaban Minda:""",
        input_variables=["context", "question"]
    )

    return llm, retriever, QA_PROMPT

llm, retriever, qa_prompt = init_rag()

# ---- Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=20, memory_key="chat_history",
        return_messages=True, output_key="answer"
    )
if "mood_history" not in st.session_state:
    st.session_state.mood_history = {}

# ---- Sidebar ----
with st.sidebar:
    st.image("https://via.placeholder.com/150x60/534AB7/FFFFFF?text=MindEase", use_column_width=True)
    st.markdown("### 🎭 Mood Tracker Hari Ini")

    MOOD_LEVELS = {
        1: ("Sangat Buruk", "😞", "#E24B4A"),
        2: ("Kurang Baik",  "😔", "#EF9F27"),
        3: ("Biasa Saja",   "😐", "#888780"),
        4: ("Cukup Baik",   "🙂", "#5DCAA5"),
        5: ("Sangat Baik",  "😊", "#1D9E75"),
    }

    mood_level = st.select_slider(
        "Bagaimana perasaanmu hari ini?",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{MOOD_LEVELS[x][1]} {MOOD_LEVELS[x][0]}"
    )
    mood_note = st.text_input("Catatan (opsional)", placeholder="e.g. kurang tidur, habis olahraga...")

    if st.button("Simpan Mood", use_container_width=True, type="primary"):
        today = str(date.today())
        st.session_state.mood_history[today] = {
            "level": mood_level,
            "label": MOOD_LEVELS[mood_level][0],
            "emoji": MOOD_LEVELS[mood_level][1],
            "catatan": mood_note
        }
        st.success(f"{MOOD_LEVELS[mood_level][1]} Mood disimpan!")

    if st.session_state.mood_history:
        st.markdown("---")
        st.markdown("### 📊 Tren Mood")
        tanggals = sorted(st.session_state.mood_history.keys())[-7:]  # 7 hari terakhir
        levels = [st.session_state.mood_history[t]["level"] for t in tanggals]

        fig, ax = plt.subplots(figsize=(4, 2.5))
        colors = [MOOD_LEVELS[l][2] for l in levels]
        ax.bar(range(len(tanggals)), levels, color=colors, width=0.6, alpha=0.85)
        ax.plot(range(len(tanggals)), levels, "o-", color="#534AB7", linewidth=1.5, markersize=5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_xticks(range(len(tanggals)))
        ax.set_xticklabels([t[8:] for t in tanggals], fontsize=8)
        ax.set_ylim(0.5, 5.5)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
    ⚠️ MindEase adalah alat bantu edukasi, bukan pengganti psikolog profesional.<br>
    Darurat: <strong>119 ext 8</strong>
    </div>
    """, unsafe_allow_html=True)

# ---- Main Chat Area ----
st.title("🧠 MindEase")
st.caption("Teman curhat kesehatan mental kamu — Hai, aku Minda! 💙")

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🧠"):
            st.write(msg["content"])

# Input chat
if prompt := st.chat_input("Ceritakan apa yang kamu rasakan..."):
    # Simpan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate respons
    CRISIS_KEYWORDS = ["bunuh diri", "menyakiti diri", "tidak mau hidup", "ingin mati"]

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Minda sedang berpikir..."):
            if any(kw in prompt.lower() for kw in CRISIS_KEYWORDS):
                response = """Hey, aku sangat khawatir mendengar itu. 💙

Kamu tidak sendirian. Tolong hubungi:
📞 **Hotline Kemenkes: 119 ext 8** (24 jam, gratis)

Aku di sini bersamamu."""
            else:
                # Tambah konteks mood
                today = str(date.today())
                mood_ctx = ""
                if today in st.session_state.mood_history:
                    m = st.session_state.mood_history[today]
                    mood_ctx = f"[Mood hari ini: {m["emoji"]} {m["label"]}] "

                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=st.session_state.memory,
                    combine_docs_chain_kwargs={"prompt": qa_prompt},
                    return_source_documents=False,
                    verbose=False
                )
                result = chain.invoke({"question": mood_ctx + prompt})
                response = result["answer"]

            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
