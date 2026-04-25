import uuid, os, json, tempfile
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

# ── Import ALL RAG functions from the unified engine module ──────
from final_infera import (
    # Session helpers
    new_session, silent_ingest, MAX_FILE_SIZE_MB,
    # Core RAG pipeline
    planning_agent, query_system, generate_answer,
    detect_conflicts, check_completeness, compute_confidence,
    conf_color, conf_label,
    # Web search
    web_search_answer, web_verify_answer, web_supplement_incomplete,
    # Personal / smalltalk
    is_personal_query, handle_personal_query,
    # Conversation context
    build_conversation_context,
    # Confluence
    fetch_confluence_by_id, fetch_confluence_by_title,
    fetch_confluence_space, ingest_doc_data_list,
    # Email
    send_email_smtp,
    # Language
    LANGUAGE_OPTIONS, LANGUAGE_NAMES, detect_language,
    # Model preload
    get_model,
    normalize_math_in_text,
    extract_math_blocks,
    # NEW: Multimedia and image ingestion (Branch 2 + Branch 3)
    process_image_with_llm,
    ingest_new_source,
)

# ================================================================
# FLASK APP SETUP
# ================================================================
app = Flask(__name__, static_folder=".", static_url_path="")
app.secret_key = "infera-secret-key-" + str(uuid.uuid4())[:8]
CORS(app)

# ── In-memory session store (keyed by session_id cookie) ────────
# Structure:  { user_id: { "profile": {...}, "sessions": {sid: session_dict}, "session_order": [sid,...], "active_session": sid, "settings": {...} } }
_users = {}

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPABASE_URL = "https://uvkdrgbgjbugucouikrp.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_db_connection():
    return psycopg2.connect(
        host="db.uvkdrgbgjbugucouikrp.supabase.co",
        database="infera",
        user="postgres",
        password=os.getenv("DB_PASSWORD"),
        port="5432"
    )


def _get_user_id():
    """Get or create a persistent user ID from the cookie-based Flask session."""
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())[:12]
    return session["user_id"]


def _get_user_store():
    """Get or initialise the full store for the current user."""
    uid = _get_user_id()
    if uid not in _users:
        sid = str(uuid.uuid4())
        _users[uid] = {
            "profile": {"name": "User", "email": "", "phone": ""},
            "sessions": {sid: new_session()},
            "session_order": [sid],
            "active_session": sid,
            "settings": {"response_language": "English"},
        }
    return _users[uid]


def _active_sess(store):
    """Return the active session dict, creating one if needed."""
    sid = store["active_session"]
    if sid not in store["sessions"]:
        sid = str(uuid.uuid4())
        store["sessions"][sid] = new_session()
        store["session_order"].insert(0, sid)
        store["active_session"] = sid
    return store["sessions"][sid], sid


# ================================================================
# SERVE FRONTEND
# ================================================================
@app.route("/")
def index():
    return send_from_directory(".", "final_frontend.html")


@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.json

        gmail = data.get("gmail")
        password = data.get("password")

        result = supabase.table("credentials") \
            .select("*") \
            .eq("gmail", gmail) \
            .execute()

        if not result.data:
            return jsonify({
                "ok": False,
                "error": "Login credentials wrong"
            })

        user = result.data[0]

        if user["password"] != password:
            return jsonify({
                "ok": False,
                "error": "Login credentials wrong"
            })

        user = result.data[0]

        session["gmail"] = user["gmail"]
        session["name"] = user["name"]

        return jsonify({
            "ok": True,
            "name": user["name"],
            "gmail": user["gmail"],
            "username": user["username"]
        })

    except Exception as e:
        print("LOGIN ERROR:", str(e))
        return jsonify({
            "ok": False,
            "error": "Server error occurred"
        })


@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.json

    gmail = data.get("gmail")
    name = data.get("name")
    password = data.get("password")

    existing = supabase.table("credentials") \
        .select("*") \
        .eq("gmail", gmail) \
        .execute()

    if existing.data:
        return jsonify({"ok": False, "error": "User already exists"})

    username = gmail.split("@")[0] + "@infera.ai"

    supabase.table("credentials").insert({
        "gmail": gmail,
        "name": name,
        "password": password,
        "username": username
    }).execute()

    return jsonify({"ok": True,
        "message": "Signup successful"
    })

# ================================================================
# SESSIONS
# ================================================================
@app.route("/api/session/new", methods=["POST"])
def api_new_session():
    data = request.get_json(force=True)
    incognito_chat = data.get("incognito_chat", False)
    store = _get_user_store()

    sid = str(uuid.uuid4())

    gmail = session.get("gmail")

    if not incognito_chat:
        supabase.table("chat_sessions").insert({
            "id": sid,
            "gmail": gmail,
            "title": "Untitled Chat"
        }).execute()

    store["sessions"][sid] = new_session()
    store["session_order"].insert(0, sid)
    store["active_session"] = sid

    return jsonify({"ok": True, "session_id": sid})


@app.route("/api/sessions", methods=["GET"])
def api_list_sessions():
    gmail = session.get("gmail")

    result = supabase.table("chat_sessions") \
        .select("*") \
        .eq("gmail", gmail) \
        .order("created_at", desc=True) \
        .execute()

    sessions = []

    for row in result.data:
        sessions.append({
            "id": row["id"],
            "title": row["title"],
            "is_active": False
        })

    return jsonify(sessions)


@app.route("/api/session/<sid>/activate", methods=["POST"])
def api_activate_session(sid):
    store = _get_user_store()
    if sid in store["sessions"]:
        store["active_session"] = sid
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Session not found"}), 404


@app.route("/api/session/<sid>/delete", methods=["DELETE"])
def api_delete_session(sid):
    store = _get_user_store()
    if sid not in store["sessions"]:
        return jsonify({"ok": False, "error": "Session not found"}), 404

    # Remove session
    del store["sessions"][sid]
    if sid in store["session_order"]:
        store["session_order"].remove(sid)

    # If we deleted the active session, switch to another or create new
    if store["active_session"] == sid:
        if store["session_order"]:
            store["active_session"] = store["session_order"][0]
        else:
            new_sid = str(uuid.uuid4())
            store["sessions"][new_sid] = new_session()
            store["session_order"].insert(0, new_sid)
            store["active_session"] = new_sid

    return jsonify({"ok": True, "active_session": store["active_session"]})


@app.route("/api/session/<sid>/messages", methods=["GET"])
def api_get_messages(sid):
    result = supabase.table("chat_messages") \
        .select("*") \
        .eq("session_id", sid) \
        .order("timestamp") \
        .execute()

    session_info = supabase.table("chat_sessions") \
        .select("file_names") \
        .eq("id", sid) \
        .execute()

    file_names = []
    if session_info.data:
        file_names = session_info.data[0]["file_names"]

    return jsonify({
        "ok": True,
        "messages": result.data,
        "file_names": file_names
    })


# ================================================================
# FILE UPLOAD
# ================================================================
@app.route("/api/upload", methods=["POST"])
def api_upload():
    store = _get_user_store()
    sess, sid = _active_sess(store)

    files = request.files.getlist("files")
    if not files:
        return jsonify({"ok": False, "error": "No files provided"}), 400

    file_tuples = []
    rejected = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in (".pdf", ".docx", ".txt", ".html", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
            rejected.append(f"{f.filename} (unsupported format)")
            continue
        file_bytes = f.read()
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            rejected.append(f"{f.filename} (exceeds {MAX_FILE_SIZE_MB}MB)")
            continue
        file_tuples.append((f.filename, file_bytes, ext))

    result = silent_ingest(file_tuples, sess)
    ingested = result["ingested"]
    ingestion_failures = result["failures"]

    supabase.table("chat_sessions").update({
        "file_names": sess["file_names"]
    }).eq("id", sid).execute()

    return jsonify({
        "ok": True,
        "ingested": ingested,
        "rejected": rejected,
        "ingestion_failures": ingestion_failures,
        "file_names": sess.get("file_names", []),
        "rag_ready": sess.get("rag_ready", False),
        "detected_languages": sess.get("detected_languages", []),
    })


# ================================================================
# CONFLUENCE
# ================================================================
@app.route("/api/confluence", methods=["POST"])
def api_confluence():
    store = _get_user_store()
    sess, sid = _active_sess(store)
    data = request.get_json(force=True)

    base_url = data.get("base_url", "").strip()
    email    = data.get("email", "").strip()
    token    = data.get("token", "").strip()
    mode     = data.get("mode", "")  # "page_id", "title", "space"

    if not all([base_url, email, token, mode]):
        return jsonify({"ok": False, "error": "Missing required fields"}), 400

    try:
        if mode == "page_id":
            page_id = data.get("page_id", "").strip()
            if not page_id:
                return jsonify({"ok": False, "error": "Page ID required"}), 400
            doc_data = fetch_confluence_by_id(base_url, page_id, email, token)
            if doc_data["source_file"] not in sess.get("file_names", []):
                ingest_doc_data_list([doc_data], sess)
                sess.setdefault("file_names", []).append(doc_data["source_file"])
                return jsonify({"ok": True, "message": f"Indexed: {doc_data['source_file']}",
                                "file_names": sess["file_names"], "rag_ready": sess.get("rag_ready", False)})
            return jsonify({"ok": True, "message": "Already indexed"})

        elif mode == "title":
            space_key = data.get("space_key", "").strip()
            title     = data.get("title", "").strip()
            if not all([space_key, title]):
                return jsonify({"ok": False, "error": "Space key and title required"}), 400
            doc_data = fetch_confluence_by_title(base_url, space_key, title, email, token)
            if doc_data["source_file"] not in sess.get("file_names", []):
                ingest_doc_data_list([doc_data], sess)
                sess.setdefault("file_names", []).append(doc_data["source_file"])
                return jsonify({"ok": True, "message": f"Indexed: {doc_data['source_file']}",
                                "file_names": sess["file_names"], "rag_ready": sess.get("rag_ready", False)})
            return jsonify({"ok": True, "message": "Already indexed"})

        elif mode == "space":
            space_key  = data.get("space_key", "").strip()
            max_pages  = int(data.get("max_pages", 20))
            if not space_key:
                return jsonify({"ok": False, "error": "Space key required"}), 400
            doc_data_list = fetch_confluence_space(base_url, space_key, email, token, max_pages=max_pages)
            already = set(sess.get("file_names", []))
            new_docs = [d for d in doc_data_list if d["source_file"] not in already]
            if new_docs:
                ingest_doc_data_list(new_docs, sess)
                for d in new_docs:
                    if d["source_file"] not in sess.get("file_names", []):
                        sess.setdefault("file_names", []).append(d["source_file"])
                return jsonify({"ok": True, "message": f"Indexed {len(new_docs)} pages",
                                "file_names": sess["file_names"], "rag_ready": sess.get("rag_ready", False)})
            return jsonify({"ok": True, "message": "All pages already indexed"})

        else:
            return jsonify({"ok": False, "error": f"Unknown mode: {mode}"}), 400

    except (RuntimeError, ValueError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400

# ================================================================
# AUDIO INGESTION  (from Branch 3)
# ================================================================
@app.route("/api/ingest_audio", methods=["POST"])
def api_ingest_audio():
    store = _get_user_store()
    sess, sid = _active_sess(store)
    file = request.files.get("file")
    if not file:
        return jsonify({"ok": False, "error": "No file provided"}), 400
    allowed_audio_exts = [".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac", ".wma", ".opus", ".amr",
                          ".mpeg", ".mpga", ".mp2", ".mp2a", ".m2a", ".m3a", ".mpa", ".mp4a"]
    ext = Path(file.filename).suffix.lower()
    mime = (file.content_type or "").lower()
    if ext not in allowed_audio_exts and not mime.startswith("audio/"):
        return jsonify({"ok": False, "error": f"Unsupported audio format: {ext}"}), 400
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        doc_data = ingest_new_source(tmp_path, file.filename)
        os.remove(tmp_path)
        if doc_data["source_file"] not in sess.get("file_names", []):
            ingest_doc_data_list([doc_data], sess)
            sess.setdefault("file_names", []).append(doc_data["source_file"])
        return jsonify({"ok": True, "message": f"Processed {file.filename}",
                        "file_names": sess.get("file_names", []), "rag_ready": sess.get("rag_ready", False)})
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"ok": False, "error": str(e)}), 500


# ================================================================
# VIDEO INGESTION  (from Branch 3)
# ================================================================
@app.route("/api/ingest_video", methods=["POST"])
def api_ingest_video():
    store = _get_user_store()
    sess, sid = _active_sess(store)
    file = request.files.get("file")
    if not file:
        return jsonify({"ok": False, "error": "No file provided"}), 400
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".3gp"]:
        return jsonify({"ok": False, "error": "Unsupported video format"}), 400
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        doc_data = ingest_new_source(tmp_path, file.filename)
        os.remove(tmp_path)
        if doc_data["source_file"] not in sess.get("file_names", []):
            ingest_doc_data_list([doc_data], sess)
            sess.setdefault("file_names", []).append(doc_data["source_file"])
        return jsonify({"ok": True, "message": f"Processed {file.filename}",
                        "file_names": sess.get("file_names", []), "rag_ready": sess.get("rag_ready", False)})
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"ok": False, "error": str(e)}), 500


# ================================================================
# URL / YOUTUBE INGESTION  (from Branch 3)
# ================================================================
@app.route("/api/ingest_url", methods=["POST"])
def api_ingest_url():
    store = _get_user_store()
    sess, sid = _active_sess(store)
    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"ok": False, "error": "No URL provided"}), 400
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    from urllib.parse import urlparse as _urlparse
    parsed = _urlparse(url)
    domain = parsed.netloc or url
    if "youtube.com" in domain or "youtu.be" in domain:
        display_name = f"[YouTube] {url}"
    else:
        path = parsed.path.strip("/")
        display_name = f"[Web] {domain}" + (f"/{path[:40]}" if path else "")
    try:
        doc_data = ingest_new_source(url, display_name)
        if doc_data["source_file"] not in sess.get("file_names", []):
            ingest_doc_data_list([doc_data], sess)
            sess.setdefault("file_names", []).append(doc_data["source_file"])
        n_chunks = len(sess.get("chunks", []))
        return jsonify({"ok": True,
                        "message": f"Indexed {display_name} ({n_chunks} chunks ready)",
                        "file_names": sess.get("file_names", []),
                        "rag_ready": sess.get("rag_ready", False)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

# ================================================================
# SETTINGS
# ================================================================
@app.route("/api/set_language", methods=["POST"])
def api_set_language():
    store = _get_user_store()
    data = request.get_json(force=True)
    lang = data.get("language", "English")
    if lang in LANGUAGE_OPTIONS:
        store["settings"]["response_language"] = lang
        return jsonify({"ok": True, "language": lang})
    return jsonify({"ok": False, "error": "Unsupported language"}), 400


# ================================================================
# MAIN QUERY ENDPOINT
# ================================================================
@app.route("/api/query", methods=["POST"])
def api_query():
    store = _get_user_store()
    sess, sid = _active_sess(store)

    data = request.get_json(force=True)
    user_query = data.get("query", "").strip()
    incognito_chat = data.get("incognito_chat", False)
    thinking_mode = data.get("thinking_mode", False)

    if thinking_mode:
        print("🧠 THINKING MODE ACTIVE")
    else:
        print("⚡ NORMAL MODE")

    if not user_query:
        return jsonify({"ok": False, "error": "Empty query"}), 400

    gmail = session.get("gmail")

    existing = supabase.table("chat_sessions") \
        .select("id") \
        .eq("id", sid) \
        .execute()

    if not existing.data and not incognito_chat:
        title = user_query[:50] if user_query else "Untitled Chat"

        supabase.table("chat_sessions").insert({
            "id": sid,
            "gmail": gmail,
            "title": title
        }).execute()

    response_lang = store["settings"].get("response_language", "English")

    # Record user message
    sess["messages"].append({"role": "user", "content": user_query})

    if not incognito_chat:
        supabase.table("chat_messages").insert({
            "session_id": sid,
            "role": "user",
            "content": user_query
        }).execute()

    # Set title from first user query
    user_qs = [m["content"] for m in sess["messages"] if m["role"] == "user"]
    if len(user_qs) == 1:
        sess["title"] = user_query[:60]

        if not incognito_chat:
            supabase.table("chat_sessions").update({
                "title": user_query[:60]
            }).eq("id", sid).execute()

    email_intent = any(kw in user_query.lower()
                       for kw in ["send email", "email this", "mail this", "send to", "email answer"])

    # ── GATE 1: No documents ──────────────────────────────────────
    if not sess["rag_ready"]:

        # Allow personal / general questions without docs
        if is_personal_query(user_query):
            personal_ans = handle_personal_query(user_query, response_lang)

            assistant_msg = {
                "role": "assistant",
                "content": personal_ans,
                "citations": [],
                "web_sources": [],
                "confidence_score": 99,
                "confidence_breakdown": {},
                "supplement": "",
                "show_email": False
            }

            sess["messages"].append(assistant_msg)
            if not incognito_chat:
                supabase.table("chat_messages").insert({
                    "session_id": sid,
                    "role": "assistant",
                    "content": assistant_msg["content"]
                }).execute()

            return jsonify({
                "ok": True,
                "type": "personal",
                "answer": personal_ans,
                "confidence_score": 99,
                "confidence_breakdown": {}
            })

        msg = ("⚠️ No documents loaded yet. Please upload a PDF, DOCX, TXT, or HTML file "
            "— or connect a Confluence page — using the sidebar.")

        assistant_msg = {
            "role": "assistant",
            "content": msg,
            "citations": [],
            "confidence_score": 0,
            "confidence_breakdown": {}
        }

        sess["messages"].append(assistant_msg)
        if not incognito_chat:
            supabase.table("chat_messages").insert({
                "session_id": sid,
                "role": "assistant",
                "content": assistant_msg["content"]
            }).execute()

        return jsonify({
            "ok": True,
            "type": "no_docs",
            "message": msg
        })

    # ── GATE 2: Personal / smalltalk ─────────────────────────────
    if is_personal_query(user_query):
        personal_ans = handle_personal_query(user_query, response_lang)
        assistant_msg = {"role": "assistant", "content": personal_ans,
                         "citations": [], "web_sources": [],
                         "confidence_score": 99, "confidence_breakdown": {},
                         "supplement": "", "show_email": False}
        sess["messages"].append(assistant_msg)
        if not incognito_chat:
            supabase.table("chat_messages").insert({
                "session_id": sid,
                "role": "assistant",
                "content": assistant_msg["content"]
            }).execute()
        return jsonify({
            "ok": True, "type": "personal",
            "answer": personal_ans,
            "confidence_score": 99,
            "confidence_breakdown": {},
        })

    # ── MAIN PROCESSING ───────────────────────────────────────────
    plan            = planning_agent(user_query, sess["messages"][:-1], sess["file_names"])
    query_type      = plan.get("query_type", "direct")
    rewritten_query = plan.get("rewritten", user_query)
    sub_queries     = plan.get("sub_queries") or [rewritten_query]

    ret_chunks, citations, doc_has_garbled, retrieval_score = query_system(
        sub_queries, sess["index"], sess["chunks"], sess["embeddings"],
        thinking_mode=thinking_mode, session=sess
    )
    doc_has_garbled = doc_has_garbled or sess.get("has_garbled_math", False)

    # ── BRANCH A: Found in documents ─────────────────────────────
    if ret_chunks:
        answer, unique_citations = generate_answer(
            rewritten_query, ret_chunks, citations,
            query_type, doc_has_garbled, response_lang,
            thinking_mode=thinking_mode
        )
        conflict_info      = detect_conflicts(rewritten_query, ret_chunks, unique_citations)
        completeness       = check_completeness(rewritten_query, answer)
        completeness_score = completeness.get("completeness_score", 0.7)

        # Check for conflict requiring user confirmation
        if conflict_info and conflict_info.get("severity") in ("medium", "high"):
            return jsonify({
                "ok": True, "type": "conflict_prompt",
                "answer": answer,
                "citations": unique_citations,
                "conflict_info": {
                    "conflict_summary": conflict_info.get("conflict_summary", ""),
                    "severity": conflict_info.get("severity", "medium"),
                },
                "retrieval_score": retrieval_score,
                "completeness_score": completeness_score,
                "rewritten_query": rewritten_query,
            })

        supplement_text, web_sources_supp = "", []
        if not completeness.get("is_complete") and completeness.get("needs_web_supplement"):
            supplement_text, web_sources_supp = web_supplement_incomplete(
                rewritten_query, answer, completeness.get("missing_aspects", [])
            )

        verify_result, web_sources_ver = None, []
        if retrieval_score < 0.35:
            verify_result, web_sources_ver = web_verify_answer(rewritten_query, answer)

        web_verified    = bool(verify_result or web_sources_supp)
        all_web_sources = web_sources_ver or web_sources_supp
        n_unique        = len(set(c["file"] for c in unique_citations))
        conf_score, conf_break = compute_confidence(
            retrieval_score, len(ret_chunks), n_unique,
            completeness_score, web_verified=web_verified,
            conflict_detected=bool(conflict_info), query_type=query_type,
        )

        _msg_id = str(uuid.uuid4())[:6]
        assistant_msg = {
            "role": "assistant", "content": answer,
            "citations": unique_citations, "web_sources": all_web_sources,
            "confidence_score": conf_score, "confidence_breakdown": conf_break,
            "supplement": supplement_text, "show_email": email_intent, "_msg_id": _msg_id,
        }
        if conflict_info and conflict_info.get("severity") == "low":
            assistant_msg["conflict_info"] = {
                "conflict_summary": conflict_info.get("conflict_summary", ""),
                "severity": "low",
            }
        sess["messages"].append(assistant_msg)
        if not incognito_chat:
            supabase.table("chat_messages").insert({
                "session_id": sid,
                "role": "assistant",
                "content": assistant_msg["content"]
            }).execute()

        answer = normalize_math_in_text(answer)
        return jsonify({
            "ok": True, "type": "doc_answer",
            "answer": answer,
            "citations": unique_citations,
            "web_sources": all_web_sources,
            "confidence_score": conf_score,
            "confidence_breakdown": conf_break,
            "confidence_color": conf_color(conf_score),
            "confidence_label": conf_label(conf_score),
            "supplement": supplement_text,
            "conflict_info": assistant_msg.get("conflict_info"),
            "email_intent": email_intent,
        })

    # ── BRANCH B: Not found in documents ─────────────────────────
    context_block, prior_web_src = build_conversation_context(sess["messages"][:-1])

    # B1: Follow-up with prior context
    if context_block:
        from final_infera import client as groq_client

        fu_prompt = f"""You are Infera, an AI research assistant with conversation memory.
The user has a follow-up question. Their documents have no direct information about it,
but previous answers from this conversation (shown below) do.
Use the context fully. Use your own knowledge only if context is insufficient, and say so.
Never say "I cannot access documents" — just use what is provided.
{"IMPORTANT: Respond entirely in " + response_lang + "." if response_lang != "English" else ""}

=== Previous Conversation Answers ===
{context_block}
=== End ===

Follow-up: {rewritten_query}

Answer completely and clearly:"""
        try:
            fu_resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": fu_prompt}],
                max_tokens=850,
            )
            fu_answer = fu_resp.choices[0].message.content
        except Exception as ex:
            fu_answer = f"Error generating follow-up: {ex}"

        conf_s, conf_b = compute_confidence(
            0.50, 0, len(prior_web_src), 0.72, web_verified=bool(prior_web_src),
        )
        _msg_id = str(uuid.uuid4())[:6]
        assistant_msg = {
            "role": "assistant", "content": fu_answer,
            "citations": [], "web_sources": prior_web_src,
            "confidence_score": conf_s, "confidence_breakdown": conf_b,
            "supplement": "", "show_email": email_intent, "_msg_id": _msg_id,
        }
        sess["messages"].append(assistant_msg)
        if not incognito_chat:
            supabase.table("chat_messages").insert({
                "session_id": sid,
                "role": "assistant",
                "content": assistant_msg["content"]
            }).execute()

        return jsonify({
            "ok": True, "type": "followup_answer",
            "answer": fu_answer,
            "citations": [],
            "web_sources": prior_web_src,
            "confidence_score": conf_s,
            "confidence_breakdown": conf_b,
            "confidence_color": conf_color(conf_s),
            "confidence_label": conf_label(conf_s),
        })

    # B2: No context at all — prompt user for web search
    return jsonify({
        "ok": True, "type": "web_search_prompt",
        "rewritten_query": rewritten_query,
        "message": "Could not find relevant information in your documents.",
    })


# ================================================================
# WEB SEARCH (user confirmed)
# ================================================================
@app.route("/api/web_search", methods=["POST"])
def api_web_search():
    store = _get_user_store()
    sess, sid = _active_sess(store)
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    incognito_chat = data.get("incognito_chat", False)

    if not query:
        return jsonify({"ok": False, "error": "No query"}), 400

    web_ans, web_sources, from_trusted = web_search_answer(query)
    web_ans = normalize_math_in_text(web_ans)
    conf_score, conf_break = compute_confidence(
        0.55, 0, len(web_sources), 0.75,
        web_verified=bool(web_sources), from_trusted_web=from_trusted,
    )

    _msg_id = str(uuid.uuid4())[:6]
    assistant_msg = {
        "role": "assistant", "content": web_ans,
        "citations": [], "web_sources": web_sources,
        "confidence_score": conf_score, "confidence_breakdown": conf_break,
        "show_email": True, "_msg_id": _msg_id,
    }
    sess["messages"].append(assistant_msg)
    if not incognito_chat:
        supabase.table("chat_messages").insert({
            "session_id": sid,
            "role": "assistant",
            "content": assistant_msg["content"]
        }).execute()

    return jsonify({
        "ok": True, "type": "web_answer",
        "answer": web_ans,
        "web_sources": web_sources,
        "confidence_score": conf_score,
        "confidence_breakdown": conf_break,
        "confidence_color": conf_color(conf_score),
        "confidence_label": conf_label(conf_score),
    })


# ================================================================
# WEB SEARCH DECLINED
# ================================================================
@app.route("/api/web_search_decline", methods=["POST"])
def api_web_search_decline():
    store = _get_user_store()
    sess, sid = _active_sess(store)
    data = request.get_json(force=True)
    incognito_chat = data.get("incognito_chat", False)

    no_reply = (
        "Understood \u2014 I won't search the web for this one.\n\n"
        "Since the answer isn't in your uploaded documents and you've chosen not to use "
        "web search, I'm unable to provide a response for this query right now.\n\n"
        "That said, if you change your mind, simply ask the same question again and "
        "choose **Yes** when I offer to search the web \u2014 I'll find the best answer "
        "from trusted sources right away.\n\n"
        "Is there anything else I can help you with from your documents? \U0001f4c4"
    )
    assistant_msg = {
        "role": "assistant", "content": no_reply,
        "citations": [], "web_sources": [],
        "confidence_score": 0, "confidence_breakdown": {}, "show_email": False,
    }
    sess["messages"].append(assistant_msg)
    if not incognito_chat:
        supabase.table("chat_messages").insert({
            "session_id": sid,
            "role": "assistant",
            "content": assistant_msg["content"]
        }).execute()

    return jsonify({"ok": True, "type": "declined", "answer": no_reply})


# ================================================================
# CONFLICT RESOLUTION (user chose to verify with web)
# ================================================================
@app.route("/api/web_verify", methods=["POST"])
def api_web_verify():
    store = _get_user_store()
    sess, sid = _active_sess(store)
    data = request.get_json(force=True)
    incognito_chat = data.get("incognito_chat", False)

    query            = data.get("query", "")
    doc_answer       = data.get("doc_answer", "")
    citations_data   = data.get("citations", [])
    retrieval_score  = data.get("retrieval_score", 0.5)
    completeness_sc  = data.get("completeness_score", 0.7)

    verify_result, web_sources = web_verify_answer(query, doc_answer)
    supplement = ""
    if verify_result:
        if verify_result.get("corrections"):
            supplement += "**Web corrections:** " + "; ".join(verify_result["corrections"]) + "\n\n"
        if verify_result.get("supplement"):
            supplement += verify_result["supplement"]

    conf_score, conf_break = compute_confidence(
        retrieval_score, len(citations_data),
        len(set(c["file"] for c in citations_data)),
        completeness_sc, web_verified=True, conflict_detected=False,
    )

    _msg_id = str(uuid.uuid4())[:6]
    assistant_msg = {
        "role": "assistant", "content": doc_answer,
        "citations": citations_data, "web_sources": web_sources,
        "confidence_score": conf_score, "confidence_breakdown": conf_break,
        "supplement": supplement, "show_email": True, "_msg_id": _msg_id,
    }
    sess["messages"].append(assistant_msg)
    if not incognito_chat:
        supabase.table("chat_messages").insert({
            "session_id": sid,
            "role": "assistant",
            "content": assistant_msg["content"]
        }).execute()

    return jsonify({
        "ok": True, "type": "verified_answer",
        "answer": doc_answer,
        "citations": citations_data,
        "web_sources": web_sources,
        "confidence_score": conf_score,
        "confidence_breakdown": conf_break,
        "confidence_color": conf_color(conf_score),
        "confidence_label": conf_label(conf_score),
        "supplement": supplement,
    })


# ================================================================
# CONFLICT — USE DOC ANSWER AS-IS
# ================================================================
@app.route("/api/conflict_use_doc", methods=["POST"])
def api_conflict_use_doc():
    store = _get_user_store()
    sess, sid = _active_sess(store)
    data = request.get_json(force=True)
    incognito_chat = data.get("incognito_chat", False)

    doc_answer      = data.get("doc_answer", "")
    citations_data  = data.get("citations", [])
    retrieval_score = data.get("retrieval_score", 0.5)
    completeness_sc = data.get("completeness_score", 0.7)

    conf_score, conf_break = compute_confidence(
        retrieval_score, len(citations_data),
        len(set(c["file"] for c in citations_data)),
        completeness_sc, conflict_detected=True,
    )

    _msg_id = str(uuid.uuid4())[:6]
    assistant_msg = {
        "role": "assistant", "content": doc_answer,
        "citations": citations_data, "web_sources": [],
        "confidence_score": conf_score, "confidence_breakdown": conf_break,
        "show_email": True, "_msg_id": _msg_id,
    }
    sess["messages"].append(assistant_msg)
    if not incognito_chat:
        supabase.table("chat_messages").insert({
            "session_id": sid,
            "role": "assistant",
            "content": assistant_msg["content"]
        }).execute()

    return jsonify({
        "ok": True, "type": "doc_answer",
        "answer": doc_answer,
        "citations": citations_data,
        "web_sources": [],
        "confidence_score": conf_score,
        "confidence_breakdown": conf_break,
        "confidence_color": conf_color(conf_score),
        "confidence_label": conf_label(conf_score),
    })


# ================================================================
# EMAIL
# ================================================================
@app.route("/api/send_email", methods=["POST"])
def api_send_email():
    data = request.get_json(force=True)
    to_addr   = data.get("to", "").strip()
    subject   = data.get("subject", "Answer from Infera")
    body      = data.get("body", "")
    smtp_host = data.get("smtp_host", "smtp.gmail.com")
    smtp_port = int(data.get("smtp_port", 465))
    smtp_user = data.get("smtp_user", "")
    smtp_pass = data.get("smtp_pass", "")

    if not all([to_addr, smtp_user, smtp_pass]):
        return jsonify({"ok": False, "error": "Missing email configuration"}), 400

    try:
        send_email_smtp(to_addr, subject, body, smtp_host, smtp_port, smtp_user, smtp_pass)
        return jsonify({"ok": True, "message": f"Sent to {to_addr}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ================================================================
# STATUS / INFO
# ================================================================
@app.route("/api/status", methods=["GET"])
def api_status():
    store = _get_user_store()
    sess, sid = _active_sess(store)
    return jsonify({
        "ok": True,
        "active_session": sid,
        "rag_ready": sess.get("rag_ready", False),
        "file_names": sess.get("file_names", []),
        "language": store["settings"].get("response_language", "English"),
        "languages_available": LANGUAGE_OPTIONS,
        "detected_languages": sess.get("detected_languages", []),
    })

@app.route("/api/session/<sid>/rename", methods=["POST"])
def rename_session(sid):
    data = request.get_json(force=True)
    new_title = data.get("title", "").strip()

    if not new_title:
        return jsonify({"ok": False, "error": "Empty title"}), 400

    supabase.table("chat_sessions").update({
        "title": new_title
    }).eq("id", sid).execute()

    return jsonify({"ok": True})

@app.route("/api/session/<sid>/delete", methods=["POST"])
def delete_session(sid):
    supabase.table("chat_messages").delete().eq("session_id", sid).execute()
    supabase.table("chat_sessions").delete().eq("id", sid).execute()

    return jsonify({"ok": True})

@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})

# ================================================================
# RUN
# ================================================================
if __name__ == "__main__":
    print("\n[*] Infera Flask Server starting...")
    print("    Loading embedding model (first time may take a minute)...")
    get_model()  # Pre-load
    print("    [OK] Model loaded!")
    print("    Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
