import pickle
import re
import random
from datetime import datetime

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from portfolio_data import ABOUT, SKILLS, PROJECTS, EXPERIENCE, EDUCATION, CERTIFICATES, CONTACT
from tech_info import TECH_INFO

# ── NLTK setup ─────────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english")) - {"not", "no", "what", "how", "where", "which", "who"}


def preprocess(text: str) -> str:
    text   = str(text).lower().strip()
    text   = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


# ── Load trained model ─────────────────────────────────────────────────────────
with open("trainedmodel/intent_model.pkl", "rb") as f:
    MODEL = pickle.load(f)

print(f"✅ Model loaded | Intents: {list(MODEL.classes_)}")

CONFIDENCE_THRESHOLD = 0.32


# ── Response Builder ───────────────────────────────────────────────────────────
def build_response(intent: str, user_message: str) -> str:
    msg  = user_message.lower()
    hour = datetime.now().hour
    time_greet = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")

    # ── greeting ──────────────────────────────────────────────────────────────
    if intent == "greeting":
        return (
            f"{time_greet}! 👋 I'm Aniket's AI portfolio assistant.\n"
            "I can tell you about his skills, projects, experience, certificates, and more.\n"
            "What would you like to explore? 😊"
        )

    # ── about ─────────────────────────────────────────────────────────────────
    if intent == "about":
        passions = ", ".join(ABOUT["passion"])
        exploring = ", ".join(ABOUT["currently_exploring"])
        return (
            f"👨‍💻 About Aniket Das\n\n"
            f"{ABOUT['summary']}\n\n"
            f"🔥 Passions   : {passions}\n"
            f"🚀 Exploring  : {exploring}"
        )

    # ── skills ────────────────────────────────────────────────────────────────
    if intent == "skills":
        return (
            "Here's Aniket's complete tech stack:\n\n"
            f"🎨 Frontend  : {', '.join(SKILLS['frontend'])}\n"
            f"⚙️  Backend   : {', '.join(SKILLS['backend'])}\n"
            f"🗄️  Database  : {', '.join(SKILLS['database'])}\n"
            f"🛠️  Tools     : {', '.join(SKILLS['tools'])}\n"
            f"💻 Languages : {', '.join(SKILLS['languages'])}\n\n"
            "Ask me about any specific technology for more details!"
        )

    # ── projects ──────────────────────────────────────────────────────────────
    if intent == "projects":
        # Check if asking about a specific project
        for proj in PROJECTS:
            keywords = [proj["short"]] + proj["short"].split()
            if any(kw in msg for kw in keywords):
                features = "\n".join([f"  • {f}" for f in proj["features"]])
                return (
                    f"🔹 {proj['name']}\n\n"
                    f"📝 {proj['description']}\n\n"
                    f"🛠️  Tech Stack : {', '.join(proj['tech'])}\n\n"
                    f"✨ Features:\n{features}\n\n"
                    f"🌐 Live Demo  : {proj['live']}\n"
                    f"📂 Source     : {proj['source']}\n"
                    f"📌 Status     : {proj['status']}"
                )
        # General projects list
        lines = "\n".join([
            f"  {i+1}. {p['name']} ({p['status']})"
            for i, p in enumerate(PROJECTS)
        ])
        return (
            f"Aniket has built {len(PROJECTS)} projects:\n\n{lines}\n\n"
            "Ask me about any specific project for full details, live link, and source code!"
        )

    # ── experience ────────────────────────────────────────────────────────────
    if intent == "experience":
        result = "💼 Work Experience\n\n"
        for exp in EXPERIENCE:
            resp = "\n".join([f"    • {r}" for r in exp["responsibilities"]])
            result += (
                f"🏢 {exp['role']} @ {exp['company']} ({exp['duration']}) — {exp['type']}\n"
                f"  Responsibilities:\n{resp}\n\n"
            )
        return result.strip()

    # ── education ─────────────────────────────────────────────────────────────
    if intent == "education":
        edu = EDUCATION
        return (
            f"🎓 Education\n\n"
            f"📘 Degree      : {edu['degree']}\n"
            f"🏫 Institution : {edu['institution']}\n"
            f"📍 Location    : {edu['location']}\n"
            f"📅 Duration    : {edu['year']}\n"
            f"⭐ CGPA        : {edu['cgpa']}\n\n"
            "He combines strong academics with real-world full-stack and ML project experience!"
        )

    # ── certificates ──────────────────────────────────────────────────────────
    if intent == "certificates":
        lines = "\n".join([
            f"  {i+1}. {c['title']} — {c['date']} ({c['type']})"
            for i, c in enumerate(CERTIFICATES)
        ])
        return (
            f"🏆 Certificates & Achievements ({len(CERTIFICATES)} total)\n\n"
            f"{lines}\n\n"
            "He has participated in multiple hackathons, completed industry internships, "
            "and earned course certificates in Java, Python, AI/ML, and Web Development!"
        )

    # ── contact ───────────────────────────────────────────────────────────────
    if intent == "contact":
        return (
            "📬 Here's how to reach Aniket:\n\n"
            f"📧 Email    : {CONTACT['email']}\n"
            f"💼 LinkedIn : {CONTACT['linkedin']}\n"
            f"🐙 GitHub   : {CONTACT['github']}\n"
            f"📞 Phone    : {CONTACT['phone']}\n\n"
            "Feel free to reach out for collaborations, freelance work, or job opportunities!"
        )

    # ── hiring ────────────────────────────────────────────────────────────────
    if intent == "hiring":
        return (
            "✅ Aniket is currently open to new opportunities!\n\n"
            "He's interested in:\n"
            "  • Full-stack development roles (MERN)\n"
            "  • Freelance web development projects\n"
            "  • ML/AI integrated web applications\n"
            "  • Internships and growth-oriented positions\n\n"
            f"📧 Reach him at: {CONTACT['email']}\n"
            f"💼 LinkedIn    : {CONTACT['linkedin']}"
        )

    # ── technology ────────────────────────────────────────────────────────────
    if intent == "technology":
        for tech, info in TECH_INFO.items():
            if tech in msg:
                return f"🛠️ {info}"
        return (
            "Aniket's main technologies are React, Node.js, Express.js, MongoDB, "
            "Python, Flask, Django, Tailwind CSS, Socket.io, WebRTC, Firebase, and more.\n\n"
            "Ask me about any specific technology for a detailed answer!"
        )

    # ── thanks ────────────────────────────────────────────────────────────────
    if intent == "thanks":
        return random.choice([
            "You're very welcome! 😊 Feel free to ask anything else about Aniket's work.",
            "Happy to help! 😊 Is there anything else you'd like to know?",
            "Glad I could help! Ask me anything else about Aniket. 😊",
        ])

    # ── goodbye ───────────────────────────────────────────────────────────────
    if intent == "goodbye":
        return "Goodbye! 👋 It was great chatting. Don't hesitate to reach out to Aniket — have a wonderful day!"

    # ── help ──────────────────────────────────────────────────────────────────
    if intent == "help":
        return (
            "Here's what I can help you with:\n\n"
            "  👨‍💻  About Aniket\n"
            "  🛠️   Skills & tech stack\n"
            "  📁   Project details (with live links!)\n"
            "  💼   Work experience\n"
            "  🎓   Education\n"
            "  🏆   Certificates & achievements\n"
            "  📬   Contact information\n"
            "  ✅   Job / freelance availability\n"
            "  🔧   Specific technology questions\n\n"
            "Just ask naturally — I'll understand! 😊"
        )

    # ── fallback ──────────────────────────────────────────────────────────────
    return random.choice([
        "Hmm, I'm not sure about that. Try asking about Aniket's skills, projects, or experience! 🤔",
        "I didn't quite get that. You can ask about his tech stack, projects, education, or contact info!",
        "Could you rephrase that? I'm best at answering questions about Aniket's portfolio. 😊",
    ])


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(title="Aniket Portfolio Chatbot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 🔒 Replace with your Vercel URL in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply:      str
    intent:     str
    confidence: float


@app.get("/")
def root():
    return {"status": "✅ Aniket's Portfolio Chatbot API is running!", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_msg = req.message.strip()
    if not user_msg:
        return ChatResponse(reply="Please say something! 😊", intent="empty", confidence=1.0)

    processed  = preprocess(user_msg)
    proba      = MODEL.predict_proba([processed])[0]
    confidence = float(proba.max())
    intent     = MODEL.classes_[proba.argmax()]

    if confidence < CONFIDENCE_THRESHOLD:
        intent = "fallback"

    reply = build_response(intent, user_msg)
    return ChatResponse(reply=reply, intent=intent, confidence=round(confidence, 3))