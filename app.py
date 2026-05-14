"""
E-commerce Review Intelligence System
A premium AI-powered dashboard for customer review analysis.
"""

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import streamlit as st
import string
import re
import time
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Review Intelligence · AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  NLTK / SPACY SETUP
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_nlp_resources():
    """Download NLTK data and load spaCy model once."""
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet",   quiet=True)
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True, capture_output=True,
        )
        nlp_model = spacy.load("en_core_web_sm")
    return nlp_model, set(stopwords.words("english"))

nlp, stop_words = load_nlp_resources()

# ─────────────────────────────────────────────
#  TRAINING DATA
# ─────────────────────────────────────────────
TRAIN_REVIEWS = [
    # ── POSITIVE ─────────────────────────────
    ("I absolutely love this product it works perfectly and arrived quickly", "Positive"),
    ("Amazing quality exceeded all my expectations will buy again", "Positive"),
    ("Great packaging fast shipping very happy with the purchase", "Positive"),
    ("Excellent customer service resolved my issue immediately", "Positive"),
    ("Perfect fit beautiful design highly recommend to everyone", "Positive"),
    ("The product is very good and I really liked it", "Positive"),
    ("This is really good I loved it so much", "Positive"),
    ("Very good product I am really happy with it", "Positive"),
    ("I really liked this product it is very good", "Positive"),
    ("Absolutely fantastic product highly recommend it to everyone", "Positive"),
    ("Best purchase I have ever made incredibly satisfied", "Positive"),
    ("Superb quality product arrived on time very impressed", "Positive"),
    ("Outstanding performance exceeded every expectation I had", "Positive"),
    ("Wonderful product works exactly as described love it", "Positive"),
    ("Really happy with this purchase great value for money", "Positive"),
    ("Brilliant product very well made and delivered fast", "Positive"),
    ("I love this item it is exactly what I wanted", "Positive"),
    ("Incredible quality so satisfied with this purchase", "Positive"),
    ("Top quality product and excellent service very pleased", "Positive"),
    ("Very satisfied with the product it works great", "Positive"),
    ("Good product does exactly what it is supposed to do", "Positive"),
    ("I am very happy with this item it is perfect", "Positive"),
    ("Great product very pleased with the quality and delivery", "Positive"),
    ("Loved the product it is well made and durable", "Positive"),
    ("Highly satisfied with the purchase would definitely buy again", "Positive"),
    ("The item looks great and works even better than expected", "Positive"),
    ("So happy with this order the product is excellent", "Positive"),
    ("Really nice product very good build quality", "Positive"),
    ("This product is amazing I am completely in love with it", "Positive"),
    ("Delighted with the purchase everything was perfect", "Positive"),
    ("Awesome product arrived quickly and works perfectly", "Positive"),
    ("The best product I have bought this year highly recommend", "Positive"),
    ("Very well made item great quality very happy", "Positive"),
    ("I enjoyed using this product it is very good", "Positive"),
    ("Excellent item delivered on time great quality overall", "Positive"),

    # ── NEGATIVE ─────────────────────────────
    ("Terrible product broke after one day of use", "Negative"),
    ("Very late delivery and the package was damaged", "Negative"),
    ("Worst quality ever complete waste of money", "Negative"),
    ("Customer service was rude and unhelpful never buying again", "Negative"),
    ("Item was broken packaging was awful very disappointed", "Negative"),
    ("Horrible experience product stopped working after two days", "Negative"),
    ("Total waste of money do not buy this product", "Negative"),
    ("Very disappointed the item does not work at all", "Negative"),
    ("Damaged product received very poor packaging terrible service", "Negative"),
    ("Worst purchase ever product quality is extremely poor", "Negative"),
    ("Awful quality broke immediately extremely disappointed", "Negative"),
    ("Terrible delivery took three weeks and arrived broken", "Negative"),
    ("Do not buy this product it is a complete scam", "Negative"),
    ("Very poor quality not worth the money at all", "Negative"),
    ("Really bad product stopped working the very first day", "Negative"),
    ("Disgusting quality product looks nothing like the picture", "Negative"),
    ("Completely useless item waste of time and money", "Negative"),
    ("Terrible customer service no refund given very upset", "Negative"),
    ("Product arrived damaged and customer support ignored me", "Negative"),
    ("Worst experience ever product is cheap and broken", "Negative"),
    ("I hate this product it does not work properly", "Negative"),
    ("Very bad quality returned it immediately", "Negative"),
    ("Not at all satisfied with this product very poor", "Negative"),
    ("Broken on arrival complete disappointment do not recommend", "Negative"),
    ("Really terrible quality extremely unhappy with purchase", "Negative"),

    # ── NEUTRAL ──────────────────────────────
    ("Product is okay nothing special but does its job", "Neutral"),
    ("Average quality for the price shipping was normal", "Neutral"),
    ("It is fine not great not bad meets basic expectations", "Neutral"),
    ("Decent product arrived on time packaging was acceptable", "Neutral"),
    ("The item works as described standard quality nothing more", "Neutral"),
    ("Product is alright it does what it is supposed to do", "Neutral"),
    ("Neither good nor bad just an average product", "Neutral"),
    ("Okay product nothing extraordinary about it", "Neutral"),
    ("It does the job but nothing impressive about it", "Neutral"),
    ("Average experience delivery was normal product is basic", "Neutral"),
    ("Fairly standard product meets expectations nothing special", "Neutral"),
    ("Product is usable but not outstanding in any way", "Neutral"),
    ("It is an average item for an average price", "Neutral"),
    ("Decent enough nothing to complain about but not great", "Neutral"),
    ("Works fine for basic use not the best quality though", "Neutral"),
]

POSITIVE_WORDS = {
    "excellent", "amazing", "love", "great", "best", "perfect",
    "fantastic", "wonderful", "superb", "outstanding", "awesome",
    "brilliant", "incredible", "happy", "delighted", "satisfied",
    "good", "liked", "enjoy", "pleased", "nice", "impressed",
}

# ─────────────────────────────────────────────
#  BUILD MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_model():
    """Train TF-IDF + Logistic Regression on expanded dataset."""
    texts  = [r[0] for r in TRAIN_REVIEWS]
    labels = [r[1] for r in TRAIN_REVIEWS]
    tfidf_vec = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = tfidf_vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, C=1.5, random_state=42)
    clf.fit(X, labels)
    return tfidf_vec, clf

tfidf, model = build_model()

# ─────────────────────────────────────────────
#  CORE NLP FUNCTIONS
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Preprocess raw review text."""
    text  = text.lower()
    text  = text.translate(str.maketrans("", "", string.punctuation))
    text  = re.sub(r"\d+", "", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    doc   = nlp(" ".join(words))
    return " ".join([token.lemma_ for token in doc])


def detect_fake_review(text: str) -> str:
    """Heuristic fake-review detector."""
    words = text.split()
    if len(words) < 3:
        return "Fake Review"
    if len(set(words)) < len(words) / 2:
        return "Fake Review"
    if sum(1 for w in words if w in POSITIVE_WORDS) >= 3:
        return "Fake Review"
    return "Real Review"


def classify_issue(text: str, sentiment: str) -> list:
    """Detect issue categories from negative/neutral reviews."""
    if sentiment == "Positive":
        return ["No Issue"]
    text   = text.lower()
    issues = []
    if any(k in text for k in ["delivery", "late", "shipping", "delay"]):
        issues.append("Delivery")
    if any(k in text for k in ["broken", "quality", "bad", "defect", "cheap"]):
        issues.append("Product Quality")
    if any(k in text for k in ["package", "packaging", "box", "wrap"]):
        issues.append("Packaging")
    if any(k in text for k in ["payment", "charge", "refund", "bill"]):
        issues.append("Payment")
    if any(k in text for k in ["support", "service", "customer", "rude", "helpful"]):
        issues.append("Customer Service")
    return issues if issues else ["No Issue"]


def predict_review(review: str) -> dict:
    """Full prediction pipeline; returns a result dict."""
    cleaned   = clean_text(review)
    vector    = tfidf.transform([cleaned])
    sentiment = model.predict(vector)[0]
    fake      = detect_fake_review(cleaned)
    issues    = classify_issue(cleaned, sentiment)
    return {
        "sentiment":   sentiment,
        "fake_status": fake,
        "issues":      issues,
    }

# ─────────────────────────────────────────────
#  CUSTOM CSS  — light, clean, professional
# ─────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

        html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

        /* ── Page background: light slate ─── */
        .stApp { background: #f0f4f8; color: #1e293b; }
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding: 0 2.5rem 4rem 2.5rem; max-width: 1100px; }

        /* ── HERO BANNER ─────────────────── */
        .hero-banner {
            background: linear-gradient(120deg, #1e40af 0%, #2563eb 45%, #0ea5e9 100%);
            border-radius: 20px;
            padding: 3rem 3rem 2.8rem;
            margin: 1.8rem 0 2rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 8px 40px rgba(37,99,235,0.25);
        }
        .hero-banner::before {
            content: '';
            position: absolute; inset: 0;
            background-image: radial-gradient(rgba(255,255,255,0.12) 1px, transparent 1px);
            background-size: 24px 24px;
            pointer-events: none;
        }
        .hero-banner::after {
            content: '';
            position: absolute;
            top: -60px; right: -60px;
            width: 300px; height: 300px;
            background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 70%);
            pointer-events: none;
        }
        .hero-eyebrow {
            font-family: 'Space Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.78);
            margin-bottom: 0.8rem;
        }
        .hero-title {
            font-size: clamp(1.7rem, 3.2vw, 2.7rem);
            font-weight: 800;
            color: #ffffff;
            line-height: 1.15;
            margin-bottom: 0.7rem;
        }
        .hero-sub {
            font-size: 0.98rem;
            color: rgba(255,255,255,0.82);
            max-width: 500px;
            line-height: 1.65;
        }
        .hero-badges { display:flex; gap:0.55rem; flex-wrap:wrap; margin-top:1.6rem; }
        .badge {
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.18);
            color: #ffffff;
            border: 1px solid rgba(255,255,255,0.28);
        }

        /* ── SECTION HEADING ─────────────── */
        .sec-heading {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.9rem;
        }
        .sec-number {
            width: 30px; height: 30px;
            border-radius: 50%;
            background: linear-gradient(135deg, #2563eb, #0ea5e9);
            color: #fff;
            font-size: 0.78rem;
            font-weight: 700;
            display: flex; align-items: center; justify-content: center;
            flex-shrink: 0;
            box-shadow: 0 3px 10px rgba(37,99,235,0.35);
        }
        .sec-title {
            font-size: 1rem;
            font-weight: 700;
            color: #1e293b;
        }

        /* ── WHITE ROUNDED CARD ──────────── */
        .card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 1.8rem 2rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 2px 16px rgba(0,0,0,0.06);
        }

        /* ── TEXTAREA ────────────────────── */
        .stTextArea > div > div > textarea {
            background: #f8fafc !important;
            border: 1.5px solid #cbd5e1 !important;
            border-radius: 12px !important;
            color: #1e293b !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 0.95rem !important;
            line-height: 1.7 !important;
            padding: 1rem 1.2rem !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }
        .stTextArea > div > div > textarea:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
            background: #fff !important;
        }
        .stTextArea > div > div > textarea::placeholder { color: #94a3b8 !important; }

        /* ── BUTTONS ─────────────────────── */
        div[data-testid="column"] .stButton > button {
            width: 100%;
            border-radius: 10px;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 700;
            font-size: 0.88rem;
            padding: 0.65rem 1.2rem;
            border: none;
            cursor: pointer;
            transition: all 0.22s ease;
        }
        div[data-testid="column"]:nth-child(1) .stButton > button {
            background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
            color: #ffffff;
            box-shadow: 0 4px 16px rgba(37,99,235,0.3);
        }
        div[data-testid="column"]:nth-child(1) .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(37,99,235,0.45);
        }
        div[data-testid="column"]:nth-child(2) .stButton > button {
            background: #f1f5f9;
            color: #64748b;
            border: 1.5px solid #e2e8f0 !important;
        }
        div[data-testid="column"]:nth-child(2) .stButton > button:hover {
            background: #e2e8f0;
            color: #334155;
        }

        /* ── SENTIMENT ───────────────────── */
        .sentiment-row { display:flex; align-items:center; gap:1.2rem; margin-bottom:0.5rem; }
        .sent-emoji    { font-size:3rem; line-height:1; }
        .sent-label    { font-size:2rem; font-weight:800; line-height:1; }
        .sent-desc     { font-size:0.82rem; color:#64748b; margin-top:0.25rem; }

        /* ── AUTHENTICITY ────────────────── */
        .auth-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 1.1rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }
        .auth-real { background:#dcfce7; border:1.5px solid #86efac; color:#16a34a; }
        .auth-fake { background:#fee2e2; border:1.5px solid #fca5a5; color:#dc2626; }
        .auth-desc { font-size:0.78rem; color:#64748b; margin-top:0.6rem; line-height:1.5; }

        /* ── ISSUE PILLS ─────────────────── */
        .issue-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.42rem 1rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            margin: 0.25rem 0.25rem 0.25rem 0;
        }
        .pill-warn { background:#fff7ed; border:1.5px solid #fdba74; color:#ea580c; }
        .pill-ok   { background:#f0fdf4; border:1.5px solid #86efac; color:#16a34a; }

        /* ── DIVIDER ─────────────────────── */
        .soft-divider { border:none; border-top:1.5px solid #e2e8f0; margin:1.6rem 0; }

        /* ── SPINNER ─────────────────────── */
        .stSpinner > div { border-top-color: #2563eb !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "result"      not in st.session_state: st.session_state.result      = None
if "review_text" not in st.session_state: st.session_state.review_text = ""
if "analyzed"    not in st.session_state: st.session_state.analyzed    = False

# ─────────────────────────────────────────────
#  INJECT CSS
# ─────────────────────────────────────────────
inject_css()

# ─────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-eyebrow">🧠 &nbsp; Powered by NLP + Machine Learning</div>
        <div class="hero-title">E-Commerce Review<br>Intelligence System</div>
        <div class="hero-sub">
            Instantly decode customer sentiment, detect authenticity, and surface
            actionable issue signals from any customer review.
        </div>
        <div class="hero-badges">
            <span class="badge">Sentiment Analysis</span>
            <span class="badge">Fake Detection</span>
            <span class="badge">Issue Classification</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  SECTION 1 — INPUT  (inside rounded card)
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="sec-heading">
        <div class="sec-number">1</div>
        <div class="sec-title">Enter Customer Review</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="card">', unsafe_allow_html=True)

review_input = st.text_area(
    label="",
    value=st.session_state.review_text,
    height=160,
    placeholder="Paste or type a customer review here…  e.g. 'The package arrived late and the product was completely broken. Very disappointed.'",
    key="review_area",
    label_visibility="collapsed",
)

col_btn1, col_btn2, col_space = st.columns([2, 1, 4])

with col_btn1:
    analyze_clicked = st.button("⚡  Analyze Review", use_container_width=True)

with col_btn2:
    clear_clicked = st.button("✕  Clear", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Button logic ─────────────────────────────
if clear_clicked:
    st.session_state.result      = None
    st.session_state.review_text = ""
    st.session_state.analyzed    = False
    st.rerun()

if analyze_clicked:
    if not review_input.strip():
        st.warning("⚠️  Please enter a review before analyzing.")
    else:
        st.session_state.review_text = review_input
        with st.spinner("Analyzing review…"):
            time.sleep(0.5)
            result = predict_review(review_input)
        st.session_state.result   = result
        st.session_state.analyzed = True
        st.rerun()

# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────
if st.session_state.analyzed and st.session_state.result:
    r = st.session_state.result

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # ── SECTION 2 — SENTIMENT & AUTHENTICITY ─
    st.markdown(
        """
        <div class="sec-heading">
            <div class="sec-number">2</div>
            <div class="sec-title">Sentiment &amp; Authenticity</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    SENT_COLOR = {"Positive": "#16a34a", "Negative": "#dc2626", "Neutral": "#2563eb"}
    SENT_EMOJI = {"Positive": "😊",      "Negative": "😠",      "Neutral": "😐"}
    SENT_DESC  = {
        "Positive": "The review expresses a positive customer experience.",
        "Negative": "The review expresses a negative customer experience.",
        "Neutral":  "The review is neither clearly positive nor negative.",
    }

    sent       = r["sentiment"]
    sent_color = SENT_COLOR.get(sent, "#2563eb")
    sent_emoji = SENT_EMOJI.get(sent, "🤔")
    sent_desc  = SENT_DESC.get(sent, "")

    is_fake    = r["fake_status"] == "Fake Review"
    badge_cls  = "auth-fake" if is_fake else "auth-real"
    badge_icon = "⚠️" if is_fake else "✅"
    badge_text = r["fake_status"]
    auth_desc  = (
        "Heuristic flags detected: unusual word repetition, excessive positive keywords, or very short content."
        if is_fake else
        "Review passes all authenticity checks — no suspicious patterns detected."
    )

    st.markdown(
        f"""
        <div class="card">
            <div class="sentiment-row">
                <div class="sent-emoji">{sent_emoji}</div>
                <div>
                    <div class="sent-label" style="color:{sent_color};">{sent}</div>
                    <div class="sent-desc">{sent_desc}</div>
                </div>
            </div>
            <div style="margin-top:1.3rem;padding-top:1.3rem;border-top:1.5px solid #f1f5f9;">
                <div style="font-size:0.72rem;font-weight:700;color:#94a3b8;
                            letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.5rem;">
                    Authenticity Check
                </div>
                <span class="auth-badge {badge_cls}">{badge_icon}&nbsp; {badge_text}</span>
                <div class="auth-desc">{auth_desc}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── SECTION 3 — ISSUE DETECTION ──────────
    st.markdown(
        """
        <div class="sec-heading">
            <div class="sec-number">3</div>
            <div class="sec-title">Issue Detection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ISSUE_ICONS = {
        "Delivery":         "🚚",
        "Product Quality":  "📦",
        "Packaging":        "🎁",
        "Payment":          "💳",
        "Customer Service": "🎧",
        "No Issue":         "✅",
    }

    pill_html = ""
    for issue in r["issues"]:
        icon     = ISSUE_ICONS.get(issue, "⚠️")
        pill_cls = "pill-ok" if issue == "No Issue" else "pill-warn"
        pill_html += f'<span class="issue-pill {pill_cls}">{icon}&nbsp; {issue}</span>'

    no_issue = r["issues"] == ["No Issue"]
    summary  = (
        "No specific issues were detected in this review."
        if no_issue else
        "The following issue categories were identified in this review:"
    )

    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:0.88rem;color:#64748b;margin-bottom:0.9rem;">{summary}</div>
            <div>{pill_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-top:3rem;text-align:center;">
        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                    letter-spacing:0.18em;text-transform:uppercase;color:#94a3b8;">
            E-Commerce Review Intelligence System &nbsp;·&nbsp; NLP Demo Build
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)