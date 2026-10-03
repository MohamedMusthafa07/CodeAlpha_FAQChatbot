"""Streamlit UI for the CodeAlpha FAQ Chatbot.

All NLP/business logic lives in the `chatbot` package; this file only
handles presentation and session state.
"""
from __future__ import annotations

import html
import logging
from collections import Counter
from datetime import datetime

import streamlit as st

st.set_page_config(
    page_title="AI FAQ Assistant",
    page_icon="\U0001F916",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger("faq_chatbot")

try:
    from chatbot import (
        DEFAULT_DATA_PATH,
        DEFAULT_THRESHOLD,
        MAX_INPUT_CHARS,
        DatasetError,
        FAQMatcher,
        InvalidQuestionError,
        load_faqs,
    )
except ImportError as import_exc:  # missing scikit-learn or broken install
    logger.error("Import failure: %s", import_exc)
    st.error(
        "A required dependency is missing. Activate your virtual environment, run "
        "`pip install -r requirements.txt`, and restart the app."
    )
    st.stop()

EXAMPLE_QUESTIONS = (
    "I forgot my password. How do I change it?",
    "My payment failed but money was deducted",
    "How can I get my certificate?",
    "What is the internship duration?",
)

CSS = """
<style>
:root {
  --cyan: #22d3ee; --blue: #3b82f6; --muted: #8aa4c8; --text: #e6f1ff;
  --panel: rgba(15, 28, 52, 0.55); --border: rgba(34, 211, 238, 0.22);
}
.stApp {
  background:
    radial-gradient(1100px 520px at 50% -8%, rgba(34, 211, 238, 0.13), transparent 60%),
    linear-gradient(180deg, #050b18 0%, #02050d 100%);
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: rgba(6, 14, 30, 0.94); border-right: 1px solid var(--border); }
.block-container { padding-top: 1.2rem; max-width: 1050px; }

.hud-header {
  display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap;
  padding: 14px 20px; border: 1px solid var(--border); border-radius: 16px;
  background: var(--panel); backdrop-filter: blur(12px);
}
.hud-title { font-size: 1.25rem; font-weight: 700; letter-spacing: 0.22em; color: var(--cyan);
  text-shadow: 0 0 14px rgba(34, 211, 238, 0.5); }
.status { display: flex; gap: 8px; flex-wrap: wrap; font-size: 0.74rem; color: var(--muted); }
.chip { padding: 4px 10px; border: 1px solid var(--border); border-radius: 999px;
  background: rgba(34, 211, 238, 0.06); white-space: nowrap; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #34d399;
  box-shadow: 0 0 8px #34d399; margin-right: 6px; animation: blink 2s ease-in-out infinite; }

.core-wrap { text-align: center; padding: 26px 0 10px; }
.orb { position: relative; width: 104px; height: 104px; margin: 0 auto 22px; border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #a5f3fc, #22d3ee 35%, #0e4f8a 70%, #06162e);
  box-shadow: 0 0 40px rgba(34, 211, 238, 0.55), 0 0 90px rgba(59, 130, 246, 0.35);
  animation: pulse 3.2s ease-in-out infinite; }
.orb::before, .orb::after { content: ""; position: absolute; border-radius: 50%;
  border: 1px solid rgba(34, 211, 238, 0.45); }
.orb::before { inset: -14px; border-style: dashed; animation: spin 14s linear infinite; }
.orb::after { inset: -28px; border-color: rgba(59, 130, 246, 0.28); animation: spin 24s linear infinite reverse; }
.core-label { letter-spacing: 0.32em; font-size: 0.82rem; color: var(--cyan); font-weight: 600; }
.core-sub { letter-spacing: 0.22em; font-size: 0.68rem; color: var(--muted); margin-top: 4px; }

.chat-area { margin: 8px 0 14px; }
.msg { display: flex; margin: 12px 0; }
.msg.user { justify-content: flex-end; }
.bubble { max-width: 80%; padding: 12px 16px; border-radius: 16px; line-height: 1.55;
  font-size: 0.95rem; border: 1px solid var(--border); color: var(--text); word-wrap: break-word; }
.msg.user .bubble { background: linear-gradient(135deg, rgba(59, 130, 246, 0.35), rgba(34, 211, 238, 0.16));
  border-bottom-right-radius: 4px; }
.msg.bot .bubble { background: var(--panel); border-bottom-left-radius: 4px; backdrop-filter: blur(10px); }
.meta { margin-top: 9px; font-size: 0.72rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px; }
.tag { padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border); }
.tag.warn { border-color: rgba(251, 191, 36, 0.55); color: #fbbf24; }
.scorebar { height: 3px; background: rgba(255, 255, 255, 0.08); border-radius: 3px; margin-top: 8px; overflow: hidden; }
.scorebar > span { display: block; height: 100%; background: linear-gradient(90deg, #3b82f6, #22d3ee); }
.related { margin-top: 8px; font-size: 0.78rem; color: var(--muted); }
.empty-hint { text-align: center; color: var(--muted); font-size: 0.9rem; padding: 6px 0 4px; }

div[data-testid="stForm"] { border: 1px solid var(--border); border-radius: 16px; background: var(--panel); }
.stTextArea textarea { border-radius: 12px; }
.hud-footer { margin-top: 28px; padding: 14px 0 6px; text-align: center; font-size: 0.74rem;
  letter-spacing: 0.14em; color: var(--muted); border-top: 1px solid var(--border); }

@keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.05); } }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
@media (prefers-reduced-motion: reduce) {
  .orb, .orb::before, .orb::after, .dot { animation: none; }
}
@media (max-width: 640px) { .bubble { max-width: 94%; } }
</style>
"""


def _html(markup: str) -> str:
    """Strip indentation and blank lines so Markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


def _safe(text: str) -> str:
    """Escape user/FAQ text for inline HTML (also neutralises Markdown math with $)."""
    return html.escape(text).replace("$", "&#36;").replace("\n", "<br>")


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


# --------------------------------------------------------------------------
# Data / matcher (cached)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _build_matcher(path_str: str, modified_ns: int) -> FAQMatcher:
    """Build the TF-IDF index once. `modified_ns` refreshes the cache if the JSON changes."""
    return FAQMatcher(load_faqs(path_str), threshold=DEFAULT_THRESHOLD)


def get_matcher() -> FAQMatcher:
    try:
        modified_ns = DEFAULT_DATA_PATH.stat().st_mtime_ns
    except OSError as exc:
        raise DatasetError(
            f"FAQ dataset not found. Expected file: {DEFAULT_DATA_PATH}"
        ) from exc
    return _build_matcher(str(DEFAULT_DATA_PATH), modified_ns)


# --------------------------------------------------------------------------
# Session state + callbacks
# --------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("notice", None)


def process_question(question: str) -> None:
    """Match a question and append the exchange to the conversation history."""
    st.session_state.notice = None
    threshold = st.session_state.get("threshold", DEFAULT_THRESHOLD)
    try:
        result = get_matcher().match(question, threshold=threshold)
    except InvalidQuestionError as exc:
        st.session_state.notice = ("warning", str(exc))
        return
    except DatasetError as exc:
        st.session_state.notice = ("error", str(exc))
        return
    except Exception:  # noqa: BLE001 - never show tracebacks to users
        logger.exception("Unexpected error while answering a question")
        st.session_state.notice = (
            "error",
            "Something went wrong while processing your question. Please try again.",
        )
        return

    stamp = _timestamp()
    st.session_state.messages.append({"role": "user", "text": result.question, "time": stamp})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": result.answer,
            "time": stamp,
            "matched": result.matched,
            "score": result.score,
            "category": result.category,
            "matched_question": result.matched_question,
            "suggestions": [q for q, _ in result.suggestions],
        }
    )


def handle_send() -> None:
    process_question(st.session_state.get("question_input", ""))


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.notice = None


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def render_header(faq_count: int | None) -> None:
    count_chip = f'<span class="chip">{faq_count} FAQs LOADED</span>' if faq_count else ""
    st.markdown(
        _html(
            f"""
            <div class="hud-header">
              <div class="hud-title">AI FAQ ASSISTANT</div>
              <div class="status">
                <span class="chip"><span class="dot"></span>ONLINE</span>
                <span class="chip">OFFLINE NLP</span>
                <span class="chip">TF-IDF + COSINE</span>
                {count_chip}
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_core(ready: bool) -> None:
    label = "AI CORE ONLINE" if ready else "AI CORE OFFLINE"
    sub = "NLP ENGINE READY" if ready else "KNOWLEDGE BASE UNAVAILABLE"
    st.markdown(
        _html(
            f"""
            <div class="core-wrap">
              <div class="orb"></div>
              <div class="core-label">{label}</div>
              <div class="core-sub">{sub}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def _message_html(msg: dict) -> str:
    text = _safe(msg["text"])
    stamp = _safe(msg["time"])
    if msg["role"] == "user":
        return (
            f'<div class="msg user"><div class="bubble">{text}'
            f'<div class="meta"><span class="tag">You</span><span class="tag">{stamp}</span></div>'
            f"</div></div>"
        )

    score = float(msg.get("score", 0.0))
    width = max(0.0, min(100.0, score * 100))
    tags = [f'<span class="tag">{stamp}</span>']
    related = ""
    if msg.get("matched"):
        tags.insert(0, f'<span class="tag">Category: {_safe(msg.get("category") or "-")}</span>')
        tags.insert(1, f'<span class="tag">Match score: {score:.2f}</span>')
        if msg.get("matched_question"):
            related = f'<div class="related">Matched FAQ: {_safe(msg["matched_question"])}</div>'
    else:
        tags.insert(0, '<span class="tag warn">Low match</span>')
        tags.insert(1, f'<span class="tag">Match score: {score:.2f}</span>')
        if msg.get("suggestions"):
            items = " | ".join(_safe(q) for q in msg["suggestions"])
            related = f'<div class="related">Closest FAQs: {items}</div>'
    return (
        f'<div class="msg bot"><div class="bubble">{text}'
        f'<div class="scorebar"><span style="width:{width:.0f}%"></span></div>'
        f'{related}<div class="meta">{"".join(tags)}</div></div></div>'
    )


def render_chat(messages: list[dict]) -> None:
    if not messages:
        st.markdown(
            '<div class="empty-hint">Ask me anything about accounts, courses, payments, '
            "certificates or internships. Try an example from the sidebar.</div>",
            unsafe_allow_html=True,
        )
        return
    body = "".join(_message_html(m) for m in messages)
    st.markdown(f'<div class="chat-area">{body}</div>', unsafe_allow_html=True)


def render_sidebar(matcher: FAQMatcher | None) -> None:
    with st.sidebar:
        st.markdown("### System Status")
        if matcher is None:
            st.error("Knowledge base unavailable")
            return
        counts = Counter(faq.category for faq in matcher.faqs)
        st.success("AI core online - NLP engine ready")
        st.metric("FAQs in knowledge base", len(matcher.faqs))
        st.caption("Categories")
        st.markdown(
            "".join(
                f'<span class="chip" style="display:inline-block;margin:2px 4px 2px 0">'
                f"{html.escape(cat)} &middot; {n}</span>"
                for cat, n in sorted(counts.items())
            ),
            unsafe_allow_html=True,
        )
        st.divider()
        st.slider(
            "Match score threshold",
            min_value=0.05,
            max_value=0.90,
            value=DEFAULT_THRESHOLD,
            step=0.01,
            key="threshold",
            help="Answers below this cosine-similarity score trigger the fallback response.",
        )
        st.caption(f"Current threshold: {st.session_state.get('threshold', DEFAULT_THRESHOLD):.2f}")
        st.divider()
        st.markdown("### Try an example")
        for index, example in enumerate(EXAMPLE_QUESTIONS):
            st.button(example, key=f"example_{index}", on_click=process_question, args=(example,))
        st.button("Clear conversation", key="sidebar_clear", on_click=clear_chat)
        st.divider()
        with st.expander("How it works"):
            st.markdown(
                "1. Your question is validated and cleaned.\n"
                "2. Stop words are removed and words are lightly stemmed.\n"
                "3. The question is converted to a TF-IDF vector.\n"
                "4. Cosine similarity compares it with every FAQ.\n"
                "5. If the best **match score** passes the threshold, that FAQ's answer is shown; "
                "otherwise a safe fallback is returned.\n\n"
                "The match score is a similarity measure, not a probability."
            )


def render_input() -> None:
    with st.form("chat_form", clear_on_submit=True):
        st.text_area(
            "Ask a question",
            key="question_input",
            height=90,
            max_chars=MAX_INPUT_CHARS,
            placeholder="Type your question here, e.g. I forgot my password. How do I change it?",
            label_visibility="collapsed",
        )
        send_col, clear_col, _ = st.columns([1, 1, 4])
        with send_col:
            st.form_submit_button("Send", on_click=handle_send, type="primary")
        with clear_col:
            st.form_submit_button("Clear", on_click=clear_chat)


def render_footer() -> None:
    st.markdown(
        '<div class="hud-footer">CODEALPHA INTERNSHIP &nbsp;|&nbsp; FAQ CHATBOT '
        "&nbsp;|&nbsp; PYTHON + NLP</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    matcher: FAQMatcher | None = None
    load_error: str | None = None
    try:
        matcher = get_matcher()
    except DatasetError as exc:
        load_error = str(exc)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to initialise the chatbot")
        load_error = "The chatbot could not start. Please check the application logs."

    render_sidebar(matcher)
    render_header(len(matcher.faqs) if matcher else None)
    render_core(matcher is not None)

    if matcher is None:
        st.error(load_error or "The FAQ knowledge base could not be loaded.")
        st.info("Fix `data/faqs.json` (valid JSON with a non-empty `faqs` list) and refresh the page.")
        render_footer()
        return

    render_chat(st.session_state.messages)

    notice = st.session_state.get("notice")
    if notice:
        level, message = notice
        (st.warning if level == "warning" else st.error)(message)

    render_input()
    render_footer()


main()
