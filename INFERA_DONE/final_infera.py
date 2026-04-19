import uuid
import time
import base64

# ================================================================
# IMPORTS
# ================================================================
import os, re, json, tempfile, requests, smtplib, shutil
from urllib.parse import urlparse, parse_qs
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from requests.auth import HTTPBasicAuth
import numpy as np
import faiss
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from duckduckgo_search import DDGS
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ── Branch 3: Optional multimedia dependencies ──────────────────
try:
    import ffmpeg
except ImportError:
    ffmpeg = None

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from docx import Document
from bs4 import BeautifulSoup
from supabase import create_client, Client

try:
    from langdetect import detect as _detect_lang
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

# ── Thinking Mode: BM25 ─────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

# ================================================================
# GROQ CLIENT
# ================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
SUPABASE_URL = "https://uvkdrgbgjbugucouikrp.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Rate-limit-safe LLM wrapper ─────────────────────────────────
def _llm_call(messages, max_tokens=500, temperature=0.0, retries=3, model="llama-3.3-70b-versatile"):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait = min(2 ** attempt * 5, 30)
                print(f"  [LLM] Rate limited, waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Groq API rate limit exceeded after retries. Please wait a few minutes.")

# ================================================================
# WHISPER CONFIG
# ================================================================
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    return _whisper_model

# ================================================================
# FFMPEG BINARY DISCOVERY
# ================================================================
_ffmpeg_path_cache = None

def _find_ffmpeg():
    global _ffmpeg_path_cache
    if _ffmpeg_path_cache is not None:
        return _ffmpeg_path_cache
    found = shutil.which("ffmpeg")
    if found:
        _ffmpeg_path_cache = found
        return _ffmpeg_path_cache
    import platform
    candidates = []
    system = platform.system()
    if system == "Windows":
        user_home = os.environ.get("USERPROFILE", "")
        candidates = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            os.path.join(user_home, "ffmpeg", "bin", "ffmpeg.exe") if user_home else "",
        ]
    elif system == "Darwin":
        candidates = ["/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/bin/ffmpeg"]
    else:
        candidates = ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
    for path in candidates:
        if path and os.path.isfile(path):
            _ffmpeg_path_cache = path
            return _ffmpeg_path_cache
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.isfile(bundled):
            _ffmpeg_path_cache = bundled
            return _ffmpeg_path_cache
    except (ImportError, Exception):
        pass
    import platform
    system = platform.system()
    if system == "Windows":
        install_hint = "Download from https://ffmpeg.org/download.html and add to PATH"
    elif system == "Darwin":
        install_hint = "Run: brew install ffmpeg"
    else:
        install_hint = "Run: sudo apt install ffmpeg"
    raise RuntimeError(f"ffmpeg executable not found. Please install: {install_hint}")

# ================================================================
# LANGUAGE SUPPORT
# ================================================================
LANGUAGE_NAMES = {
    "en":"English","hi":"Hindi","de":"German","fr":"French","es":"Spanish",
    "zh-cn":"Chinese","zh-tw":"Chinese","ja":"Japanese","ko":"Korean",
    "ar":"Arabic","pt":"Portuguese","ru":"Russian","it":"Italian","nl":"Dutch",
    "tr":"Turkish","pl":"Polish","sv":"Swedish","bn":"Bengali","ta":"Tamil",
    "te":"Telugu","mr":"Marathi","gu":"Gujarati","kn":"Kannada","ml":"Malayalam",
    "pa":"Punjabi","ur":"Urdu","th":"Thai","vi":"Vietnamese","id":"Indonesian",
    "uk":"Ukrainian","cs":"Czech","ro":"Romanian","el":"Greek","hu":"Hungarian",
    "fi":"Finnish","da":"Danish","no":"Norwegian","he":"Hebrew","fa":"Persian",
    "sw":"Swahili","af":"Afrikaans",
}

LANGUAGE_OPTIONS = [
    "English","Hindi","German","French","Spanish","Chinese","Japanese","Korean",
    "Arabic","Portuguese","Russian","Italian","Dutch","Turkish","Polish","Swedish",
    "Bengali","Tamil","Telugu","Marathi","Gujarati","Kannada","Malayalam","Punjabi",
    "Urdu","Thai","Vietnamese","Indonesian","Ukrainian","Czech","Romanian","Greek",
    "Hungarian","Finnish","Danish","Norwegian","Hebrew","Persian","Swahili","Afrikaans",
]

def detect_language(text: str) -> str:
    if not _LANGDETECT_AVAILABLE:
        return "en"
    try:
        return _detect_lang(text[:1500])
    except Exception:
        return "en"

# ================================================================
# EMBEDDING MODEL  (singleton)
# ================================================================
_embedding_model = None

def get_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

# ================================================================
# ★ THINKING MODE — CROSS-ENCODER SINGLETON
# Loaded lazily — only when thinking mode is first used.
# Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, accurate, ~85MB)
# ================================================================
_cross_encoder = None

def get_cross_encoder():
    """Return the cross-encoder singleton, loading it on first call."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print("[Thinking Mode] Loading cross-encoder (one-time)…")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("[Thinking Mode] Cross-encoder ready.")
    return _cross_encoder

# ================================================================
# MATH DETECTION
# ================================================================
_GARBLED_MATH_RE = re.compile(
    r'(?:\b\d+\s+\d+\s+\d+\b|[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]'
    r'|(?:dt|dx|d2|dv)\s*[a-zA-Z]|[A-Za-z]{1,2}\s+\d+\s+[A-Za-z]{1,2})'
)

def _is_garbled_math(text):
    tokens = text.split()
    if not tokens: return False
    if sum(1 for t in tokens if len(t) <= 2) / len(tokens) > 0.45: return True
    if len(_GARBLED_MATH_RE.findall(text)) >= 2: return True
    return False

def _is_math_query(query):
    kws = ["derive","derivation","equation","formula","proof","expression",
           "mathematically","calculate","solve","amplitude","frequency",
           "oscillat","resonan","integral","differential","differentiat"]
    return any(kw in query.lower() for kw in kws)

# ================================================================
# FILE EXTRACTION
# ================================================================

def preprocess_image_for_ocr(pil_image):
    # ── BUG I-1 FIX: Handle RGBA → RGB before grayscale conversion ──
    # PIL images are RGB (not BGR), so use COLOR_RGB2GRAY.
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    elif pil_image.mode not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")
    img = np.array(pil_image)
    # If already grayscale (mode "L"), skip conversion
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return Image.fromarray(thresh)

# ── BUG I-2 FIX: Dynamically detect available Tesseract languages ──
_tesseract_lang_cache = None

def _get_tesseract_lang_string():
    """Detect available Tesseract language packs and return a safe lang string."""
    global _tesseract_lang_cache
    if _tesseract_lang_cache is not None:
        return _tesseract_lang_cache
    try:
        available = pytesseract.get_languages(config='')
        # Filter out 'osd' (orientation/script detection) — not a real language
        available = [l for l in available if l != 'osd']
    except Exception as e:
        print(f"[WARNING] Could not query Tesseract languages: {e}. Falling back to 'eng'.")
        _tesseract_lang_cache = "eng"
        return _tesseract_lang_cache
    # Preferred languages in priority order
    preferred = ["eng", "hin"]
    selected = [l for l in preferred if l in available]
    if not selected:
        # Use whatever is available, or 'eng' as absolute fallback
        selected = available[:3] if available else ["eng"]
    _tesseract_lang_cache = "+".join(selected)
    print(f"[INFO] Tesseract languages available: {available}")
    print(f"[INFO] Tesseract using: {_tesseract_lang_cache}")
    return _tesseract_lang_cache

def extract_text_from_scanned_pdf(file_path):
    images = convert_from_path(file_path)
    all_text = []
    page_map = []
    lang = _get_tesseract_lang_string()
    for page_num, img in enumerate(images, 1):
        img = preprocess_image_for_ocr(img)
        try:
            text = pytesseract.image_to_string(img, lang=lang)
        except Exception as e:
            print(f"[WARNING] OCR failed on page {page_num} with lang='{lang}': {e}")
            # Retry with just English as fallback
            try:
                text = pytesseract.image_to_string(img, lang="eng")
            except Exception as e2:
                print(f"[WARNING] OCR fallback also failed on page {page_num}: {e2}")
                text = ""
        if text.strip():
            cleaned = clean_text(text)
            sentences = split_sentences(cleaned)
            all_text.extend(sentences)
            page_map.extend([page_num] * len(sentences))
    return all_text, page_map

def clean_text(text):
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if s.strip()]

def extract_pdf_text(file_path):
    pages_text = []
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            if not reader.pages:
                raise ValueError("PDF contains no pages.")
            for page_num, page in enumerate(reader.pages, 1):
                raw = page.extract_text() or ""
                if raw:
                    pages_text.append((page_num, clean_text(raw)))
    except (OSError, IOError) as e:
        raise RuntimeError(f"Failed to read PDF: {e}")

    all_sentences, sentence_page_map = [], []
    garbled_count = 0

    for page_num, page_text in pages_text:
        sents = split_sentences(page_text)
        for s in sents:
            if _is_garbled_math(s):
                garbled_count += 1
        all_sentences.extend(sents)
        sentence_page_map.extend([page_num] * len(sents))

    MIN_SENTENCE_THRESHOLD = 20
    MIN_CHAR_THRESHOLD = 500
    total_chars = sum(len(s) for s in all_sentences)

    if (
        len(all_sentences) < MIN_SENTENCE_THRESHOLD
        or total_chars < MIN_CHAR_THRESHOLD
        or len(pages_text) == 0
    ):
        print("[INFO] Low text detected → using OCR fallback")
        sentences, sentence_page_map = extract_text_from_scanned_pdf(file_path)
        if sentences:
            return sentences, sentence_page_map, False
        print("[WARNING] OCR failed → returning original extraction")
        return all_sentences, sentence_page_map, False

    has_garbled = (
        len(all_sentences) > 0
        and garbled_count / len(all_sentences) > 0.15
    )
    return all_sentences, sentence_page_map, has_garbled

def extract_docx_text(file_path):
    try:
        doc = Document(file_path)
        raw = " ".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        raise RuntimeError(f"Failed to read DOCX: {e}")
    return split_sentences(clean_text(raw))

def _read_text_file(file_path):
    raw = None
    for enc in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                raw = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if raw is None:
        with open(file_path, "r", encoding="latin-1") as f:
            raw = f.read()
    return raw

def extract_html_text(file_path):
    try:
        raw  = _read_text_file(file_path)
        soup = BeautifulSoup(raw, "html.parser")
        return split_sentences(clean_text(soup.get_text(separator=" ")))
    except (OSError, IOError) as e:
        raise RuntimeError(f"Failed to read HTML: {e}")

def extract_txt_text(file_path):
    try:
        return split_sentences(clean_text(_read_text_file(file_path)))
    except (OSError, IOError) as e:
        raise RuntimeError(f"Failed to read TXT: {e}")

def process_local_file(file_path):
    ext = Path(file_path).suffix.lower()
    has_garbled_math = False
    if ext == ".pdf":
        sentences, sentence_page_map, has_garbled_math = extract_pdf_text(file_path)
    elif ext == ".docx":
        sentences = extract_docx_text(file_path); sentence_page_map = None
    elif ext == ".html":
        sentences = extract_html_text(file_path); sentence_page_map = None
    elif ext == ".txt":
        sentences = extract_txt_text(file_path); sentence_page_map = None
    elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
        return process_image_with_llm(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    return {"sentences": sentences, "source_file": os.path.basename(file_path),
            "sentence_page_map": sentence_page_map, "has_garbled_math": has_garbled_math}

# ================================================================
# IMAGE → LLM → TEXT
# ================================================================
LLM_IMAGE_LIMIT = 5

def _encode_image_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def process_image_with_llm(file_path):
    base64_img = _encode_image_base64(file_path)
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Describe this image in detail so that the description can be "
                    "used for semantic search and question answering. Include objects, "
                    "relationships, labels, text in image, diagrams, and any relevant structure."
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_img}"
                }},
            ],
        }],
    )
    description = response.choices[0].message.content
    cleaned = clean_text(description)
    sentences = split_sentences(cleaned)
    return {
        "sentences": sentences,
        "source_file": os.path.basename(file_path),
        "sentence_page_map": None,
        "has_garbled_math": False,
    }

# ================================================================
# MULTIMEDIA & URL INGESTION
# ================================================================
def transcribe_audio(file_path):
    ext = Path(file_path).suffix.lower()
    tmp_wav = None
    try:
        if ext not in (".wav", ".mp3", ".m4a") and ffmpeg is not None:
            ffmpeg_bin = _find_ffmpeg()
            tmp_wav_fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_wav = tmp_wav_fd.name
            tmp_wav_fd.close()
            ffmpeg.input(file_path).output(tmp_wav, ac=1, ar=16000).overwrite_output().run(
                cmd=ffmpeg_bin, quiet=True
            )
            file_path = tmp_wav
        model = get_whisper_model()
        segments, _ = model.transcribe(file_path)
        raw_text = " ".join([seg.text for seg in segments])
        return _clean_transcript_text(raw_text)
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)

def transcribe_video(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise ValueError("Video file is empty (0 bytes).")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg Python bindings not installed. Run: pip install ffmpeg-python")
    ffmpeg_bin = _find_ffmpeg()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        try:
            ffmpeg.input(file_path).output(tmp_path, ac=1, ar=16000).overwrite_output().run(
                cmd=ffmpeg_bin, quiet=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio from video: {e}")
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise RuntimeError("ffmpeg produced no audio output.")
        text = transcribe_audio(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    if not text or not text.strip():
        raise ValueError("Video transcription returned empty text.")
    return text

_MAX_TRANSCRIPT_CHARS = 120_000

_YT_ID_RE = re.compile(
    r'(?:youtube\.com/(?:watch\?.*?v=|embed/|v/|shorts/|live/|clip/)'
    r'|youtu\.be/)'
    r'([a-zA-Z0-9_-]{11})',
    re.IGNORECASE,
)

def _extract_youtube_video_id(url):
    url = url.strip().replace('\\n', '').replace('\\r', '').replace(' ', '')
    m = _YT_ID_RE.search(url)
    if m:
        return m.group(1)
    parsed = urlparse(url)
    vid = parse_qs(parsed.query).get("v", [None])[0]
    if vid and len(vid) >= 11:
        return vid[:11]
    if "youtu.be" in url:
        path = parsed.path.strip("/")
        if path and len(path) >= 11:
            return path[:11]
    for prefix in ("/embed/", "/v/", "/shorts/", "/live/", "/clip/"):
        if prefix in parsed.path:
            segment = parsed.path.split(prefix)[1].split("/")[0].split("?")[0]
            if segment and len(segment) >= 11:
                return segment[:11]
    return None

def _clean_transcript_text(text):
    if not text:
        return text
    text = re.sub(r'\[(?:Music|Applause|Laughter|Cheering|Cheers|Silence|Inaudible'
                  r'|Background noise|Background music|Foreign|foreign)\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', '', text)
    text = re.sub(r'\b(?:um|uh|er|erm|ah|hmm|hm|mhm|uhh|umm)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:you know,?|i mean,?|sort of|kind of)\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def ingest_youtube(url):
    url = url.strip()
    video_id = _extract_youtube_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract YouTube video ID from URL: {url}")
    errors = []

    def _truncate_at_sentence(txt, max_chars):
        if len(txt) <= max_chars:
            return txt
        cut = txt[:max_chars].rfind('. ')
        if cut > max_chars * 0.7:
            return txt[:cut + 1]
        return txt[:max_chars]

    if YouTubeTranscriptApi is not None:
        api = YouTubeTranscriptApi()
        lang_sets = [
            ('en',),
            ('en', 'en-US', 'en-GB'),
            ('hi', 'de', 'fr', 'es', 'ja', 'ko', 'pt', 'ru', 'zh-Hans', 'zh-Hant'),
        ]
        for langs in lang_sets:
            for _retry in range(2):
                try:
                    result = api.fetch(video_id, languages=langs)
                    raw_data = result.to_raw_data()
                    if raw_data:
                        text = " ".join(seg.get('text', '') for seg in raw_data)
                        text = _clean_transcript_text(text)
                        if text and len(text) > 20:
                            text = _truncate_at_sentence(text, _MAX_TRANSCRIPT_CHARS)
                            return text
                    break
                except Exception as e:
                    err_msg = str(e).lower()
                    if _retry == 0 and ("timeout" in err_msg or "connect" in err_msg):
                        time.sleep(2)
                        continue
                    errors.append(f"transcript API ({langs}): {e}")
                    break
        try:
            transcript_list = api.list(video_id)
            for t in transcript_list:
                try:
                    result = t.fetch()
                    raw_data = result.to_raw_data()
                    if raw_data:
                        text = " ".join(seg.get('text', '') for seg in raw_data)
                        text = _clean_transcript_text(text)
                        if text and len(text) > 20:
                            return _truncate_at_sentence(text, _MAX_TRANSCRIPT_CHARS)
                except Exception:
                    continue
        except Exception as e:
            errors.append(f"transcript list: {e}")

    if yt_dlp is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': tmp_path.replace('.wav', ''),
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
            'quiet': True, 'no_warnings': True, 'socket_timeout': 30,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            actual_path = tmp_path
            base = tmp_path.replace('.wav', '')
            for ext in ('.wav', '.m4a', '.webm', '.mp3', ''):
                candidate = base + ext
                if os.path.exists(candidate):
                    actual_path = candidate
                    break
            text = _clean_transcript_text(transcribe_audio(actual_path))
            return _truncate_at_sentence(text, _MAX_TRANSCRIPT_CHARS)
        except Exception as e:
            errors.append(f"yt-dlp: {e}")
        finally:
            base = tmp_path.replace('.wav', '')
            for ext_c in ('.wav', '.m4a', '.webm', '.mp3', ''):
                fpath = base + ext_c
                if os.path.exists(fpath):
                    try: os.remove(fpath)
                    except: pass
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass

    detail = "; ".join(errors[:3]) if errors else "No transcript method available"
    raise RuntimeError(f"Could not fetch transcript for YouTube video '{video_id}'. Details: {detail}")

def _try_wikipedia_api(url):
    try:
        parsed = urlparse(url)
        lang = parsed.netloc.split('.')[0]
        if "/wiki/" not in parsed.path:
            return None
        title = parsed.path.split("/wiki/")[1].split("#")[0]
        api_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {"action": "query", "format": "json", "prop": "extracts",
                  "explaintext": True, "titles": title, "redirects": 1}
        resp = requests.get(api_url, params=params, timeout=20)
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract and len(extract) > 200:
                return extract
    except Exception:
        pass
    return None

def _smart_truncate_web_text(text, max_chars=80000):
    if len(text) <= max_chars:
        return text
    paragraphs = [p.strip() for p in re.split(r'\n{2,}|\r\n{2,}', text) if p.strip()]
    paragraphs = [p for p in paragraphs if len(p.split()) >= 3]
    kept, total = [], 0
    for para in paragraphs:
        if total + len(para) > max_chars and kept:
            break
        kept.append(para)
        total += len(para)
    return "\n\n".join(kept)

def ingest_website(url):
    text = None
    errors = []
    if 'wikipedia.org/wiki/' in url:
        text = _try_wikipedia_api(url)
        if text and len(text.strip()) >= 200:
            return _smart_truncate_web_text(text, max_chars=80000)
    if (not text or len(text.strip()) < 200) and trafilatura is not None:
        for _attempt in range(2):
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    extracted = trafilatura.extract(
                        downloaded, favor_recall=True, include_comments=False,
                        include_tables=True, include_links=False,
                        deduplicate=True, no_fallback=False,
                    )
                    if extracted and len(extracted.strip()) >= 200:
                        text = extracted
                        break
                    extracted2 = trafilatura.extract(
                        downloaded, favor_recall=True, include_tables=True,
                        output_format='txt', no_fallback=False,
                    )
                    if extracted2 and len(extracted2.strip()) > len((text or '').strip()):
                        text = extracted2
                    break
                elif _attempt == 0:
                    time.sleep(1)
            except Exception as e:
                errors.append(f"trafilatura: {e}")
                if _attempt == 0:
                    time.sleep(1)
    if not text or len(text.strip()) < 200:
        _user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36',
        ]
        for _attempt, ua in enumerate(_user_agents):
            try:
                headers = {'User-Agent': ua, 'Accept': 'text/html,application/xhtml+xml', 'Accept-Language': 'en-US,en;q=0.5'}
                resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
                resp.raise_for_status()
                content_type = resp.headers.get('content-type', '')
                if 'text/html' in content_type or 'application/xhtml' in content_type:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'iframe', 'svg']):
                        tag.decompose()
                    main = (soup.find('main') or soup.find('article') or
                            soup.find(attrs={'role': 'main'}) or
                            soup.find(attrs={'id': re.compile(r'content|article|main|body', re.I)}) or
                            soup.find(attrs={'class': re.compile(r'content|article|main|post|body|entry', re.I)}))
                    bs_text = clean_text(main.get_text(separator=' ') if main else (soup.body.get_text(separator=' ') if soup.body else soup.get_text(separator=' ')))
                    bs_lines = [line.strip() for line in bs_text.split('.') if len(line.strip()) > 30]
                    bs_text = '. '.join(bs_lines)
                    if bs_text and (not text or len(bs_text) > len(text)):
                        text = bs_text
                    if text and len(text.strip()) >= 200:
                        break
            except Exception as e:
                errors.append(f"requests+bs4: {e}")
    if (not text or len(text.strip()) < 100) and sync_playwright is not None and trafilatura is not None:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                try:
                    page.goto(url, timeout=25000, wait_until='networkidle')
                    page.wait_for_timeout(2000)
                    html = page.content()
                    pw_text = trafilatura.extract(html, favor_recall=True, include_tables=True, deduplicate=True)
                    if pw_text and (not text or len(pw_text) > len(text)):
                        text = pw_text
                finally:
                    browser.close()
        except Exception as e:
            errors.append(f"playwright: {e}")
    if not text or len(text.strip()) < 30:
        detail = "; ".join(errors) if errors else "No extractable text content found"
        raise ValueError(f"Could not extract text from '{url}'. {detail}")
    text = _smart_truncate_web_text(text, max_chars=80000)
    return text

def _split_text_recursive(text, chunk_size=1000, chunk_overlap=200):
    if not text or not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    separators = ['\n\n', '\n', '. ', '! ', '? ', '; ', ', ', ' ']
    chunks = []
    def _do_split(text_block, sep_idx):
        if len(text_block) <= chunk_size:
            if text_block.strip():
                chunks.append(text_block.strip())
            return
        if sep_idx >= len(separators):
            for i in range(0, len(text_block), chunk_size - chunk_overlap):
                piece = text_block[i:i + chunk_size].strip()
                if piece:
                    chunks.append(piece)
            return
        sep = separators[sep_idx]
        parts = text_block.split(sep)
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) > chunk_size and current:
                if current.strip():
                    chunks.append(current.strip())
                overlap_text = current[-chunk_overlap:] if len(current) > chunk_overlap else current
                current = overlap_text + sep + part
            else:
                current = candidate
        if current.strip():
            if len(current) > chunk_size:
                _do_split(current, sep_idx + 1)
            else:
                chunks.append(current.strip())
    _do_split(text, 0)
    deduped = []
    for c in chunks:
        if not deduped or c != deduped[-1]:
            deduped.append(c)
    return deduped

def ingest_new_source(source, source_name):
    text = None
    is_url = source.startswith("http://") or source.startswith("https://")
    if is_url:
        source = source.strip().replace(' ', '%20')
        if any(yt_pattern in source.lower() for yt_pattern in [
            "youtube.com/watch", "youtu.be/", "youtube.com/embed/",
            "youtube.com/v/", "youtube.com/shorts/", "youtube.com/live/",
        ]):
            text = ingest_youtube(source)
        else:
            text = ingest_website(source)
    else:
        ext = Path(source).suffix.lower()
        if ext in [".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac", ".wma", ".opus", ".amr",
                   ".mpeg", ".mpga", ".mp2", ".mp2a", ".m2a", ".m3a", ".mpa", ".mp4a"]:
            text = transcribe_audio(source)
        elif ext in [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"]:
            text = transcribe_video(source)
        else:
            raise ValueError(f"Unsupported media format: {ext}")
    if not text:
        raise ValueError("Extraction failed: No text extracted.")
    text = clean_text(text)
    chunks_raw = _split_text_recursive(text, chunk_size=1000, chunk_overlap=200)
    sentences = chunks_raw if chunks_raw else split_sentences(text)
    if not sentences:
        raise ValueError("Text was extracted but couldn't be split into usable chunks.")
    return {"sentences": sentences, "source_file": source_name,
            "sentence_page_map": None, "has_garbled_math": False,
            "target_words": 300}

# ================================================================
# CONFLUENCE
# ================================================================
def _confluence_headers():
    return {"Accept": "application/json", "Content-Type": "application/json"}

def _build_auth(email, token):
    return HTTPBasicAuth(email, token)

def fetch_confluence_by_id(base_url, page_id, email, token):
    base_url = base_url.rstrip("/")
    url      = f"{base_url}/rest/api/content/{page_id}"
    params   = {"expand": "body.storage,title,space,ancestors"}
    try:
        resp = requests.get(url, params=params, auth=_build_auth(email, token),
                            headers=_confluence_headers(), timeout=15)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Could not connect to Confluence at '{base_url}'.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Confluence request timed out.")
    if resp.status_code == 401: raise RuntimeError("Authentication failed.")
    if resp.status_code == 403: raise RuntimeError("Permission denied.")
    if resp.status_code == 404: raise RuntimeError(f"Page ID '{page_id}' not found.")
    if not resp.ok: raise RuntimeError(f"Confluence API error {resp.status_code}: {resp.text[:200]}")
    data       = resp.json()
    page_title = data.get("title", f"confluence_page_{page_id}")
    html_body  = data.get("body", {}).get("storage", {}).get("value", "")
    if not html_body.strip(): raise ValueError(f"Page '{page_title}' has no body content.")
    return _parse_confluence_html(html_body, f"[Confluence] {page_title}")

def fetch_confluence_by_title(base_url, space_key, title, email, token):
    base_url = base_url.rstrip("/")
    url      = f"{base_url}/rest/api/content"
    params   = {"spaceKey": space_key, "title": title, "expand": "body.storage", "limit": 1}
    try:
        resp = requests.get(url, params=params, auth=_build_auth(email, token),
                            headers=_confluence_headers(), timeout=15)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Could not connect to Confluence at '{base_url}'.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Confluence request timed out.")
    if resp.status_code == 401: raise RuntimeError("Authentication failed.")
    if not resp.ok: raise RuntimeError(f"Confluence API error {resp.status_code}: {resp.text[:200]}")
    results = resp.json().get("results", [])
    if not results: raise ValueError(f"No page found with title '{title}' in space '{space_key}'.")
    page       = results[0]
    page_title = page.get("title", title)
    html_body  = page.get("body", {}).get("storage", {}).get("value", "")
    if not html_body.strip(): raise ValueError(f"Page '{page_title}' has no body content.")
    return _parse_confluence_html(html_body, f"[Confluence] {page_title}")

def fetch_confluence_space(base_url, space_key, email, token, max_pages=20):
    base_url = base_url.rstrip("/")
    url      = f"{base_url}/rest/api/content"
    params   = {"spaceKey": space_key, "expand": "body.storage",
                "limit": max_pages, "start": 0, "type": "page"}
    try:
        resp = requests.get(url, params=params, auth=_build_auth(email, token),
                            headers=_confluence_headers(), timeout=20)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Could not connect to Confluence at '{base_url}'.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Confluence request timed out.")
    if resp.status_code == 401: raise RuntimeError("Authentication failed.")
    if not resp.ok: raise RuntimeError(f"Confluence API error {resp.status_code}: {resp.text[:200]}")
    results = resp.json().get("results", [])
    if not results: raise ValueError(f"No pages found in space '{space_key}'.")
    docs = []
    for page in results:
        page_title = page.get("title", "untitled")
        html_body  = page.get("body", {}).get("storage", {}).get("value", "")
        if not html_body.strip(): continue
        try:
            docs.append(_parse_confluence_html(html_body,
                        f"[Confluence:{space_key}] {page_title}"))
        except Exception:
            continue
    if not docs: raise ValueError(f"No readable pages found in space '{space_key}'.")
    return docs

def _parse_confluence_html(html_body, source_name):
    soup = BeautifulSoup(html_body, "html.parser")
    for tag in soup.find_all(["script","style","ac:image","ri:attachment","ac:parameter"]):
        tag.decompose()
    for macro in soup.find_all("ac:structured-macro"):
        macro.replace_with(macro.get_text(separator=" "))
    text = clean_text(soup.get_text(separator=" "))
    if not text.strip(): raise ValueError(f"Could not extract any text from '{source_name}'.")
    return {"sentences": split_sentences(text), "source_file": source_name,
            "sentence_page_map": None, "has_garbled_math": False}

def ingest_doc_data_list(doc_data_list, sess):
    """
    Ingest a list of document_data dicts into the session.
    Builds FAISS index + embeddings as before.
    ★ Also builds BM25 index stored in sess["bm25"] for thinking mode.
    """
    all_chunks  = list(sess.get("chunks", []))
    any_garbled = sess.get("has_garbled_math", False)
    for doc_data in doc_data_list:
        if doc_data.get("has_garbled_math"): any_garbled = True
        doc_chunks = chunk_text(doc_data)
        offset     = len(all_chunks)
        for c in doc_chunks: c["chunk_id"] += offset
        all_chunks.extend(doc_chunks)
    if all_chunks:
        emb = create_embeddings(all_chunks)
        idx = build_faiss_index(emb)

        # ── ★ BM25 index for thinking mode ──────────────────────
        bm25_obj = None
        if _BM25_AVAILABLE:
            tokenized = [c["text"].lower().split() for c in all_chunks]
            bm25_obj  = BM25Okapi(tokenized)

        sess.update({
            "index":            idx,
            "chunks":           all_chunks,
            "embeddings":       emb,
            "rag_ready":        True,
            "has_garbled_math": any_garbled,
            # Thinking mode artefacts
            "bm25":             bm25_obj,
        })

# ================================================================
# CHUNKING
# ================================================================
def chunk_text(document_data, target_words=None, overlap_sentences=3):
    if target_words is None:
        target_words = document_data.get("target_words", 150)
    sentences         = document_data["sentences"]
    source_file       = document_data["source_file"]
    sentence_page_map = document_data.get("sentence_page_map")
    has_garbled_math  = document_data.get("has_garbled_math", False)
    chunks, i = [], 0
    while i < len(sentences):
        chunk_sentences, word_count, j = [], 0, i
        while j < len(sentences):
            sent = sentences[j]
            wc   = len(sent.split())
            if word_count + wc > target_words and chunk_sentences: break
            chunk_sentences.append(sent); word_count += wc; j += 1
        if not chunk_sentences: break
        if word_count < 20 and chunks: break
        pages = []
        if sentence_page_map:
            for si in range(i, min(j, len(sentence_page_map))):
                p = sentence_page_map[si]
                if p is not None and p not in pages: pages.append(p)
            pages.sort()
        chunks.append({"chunk_id": len(chunks), "text": " ".join(chunk_sentences),
                       "source_file": source_file, "pages": pages,
                       "sentence_start": i, "sentence_end": j,
                       "has_garbled_math": has_garbled_math})
        step = max(1, (j - i) - overlap_sentences)
        i += step
    return chunks

# ================================================================
# EMBEDDINGS & FAISS
# ================================================================
def create_embeddings(chunks):
    texts = [c["text"] for c in chunks]
    return get_model().encode(texts, convert_to_numpy=True).astype("float32")

def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# ================================================================
# STOP WORDS & KEYWORDS
# ================================================================
_STOP_WORDS = frozenset({
    "i","me","my","we","our","you","your","he","she","it","its","they","them","their",
    "this","that","these","those","is","am","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","shall","should","can","could",
    "may","might","must","a","an","the","and","but","or","nor","not","so","yet",
    "in","on","at","to","for","of","with","by","from","as","into","about","between",
    "through","during","before","after","above","below","up","down","out","off","over",
    "under","what","which","who","whom","how","when","where","why","if","then","than",
    "because","while","although","all","each","every","both","few","more","most","some",
    "such","no","only","own","same","too","very","give","tell","explain","describe",
    "define","find","just","also","here","there",
})

def _extract_keywords(text):
    words = re.findall(r'\b\w{2,}\b', text.lower(), re.UNICODE)
    return {w for w in words if w not in _STOP_WORDS}

# ================================================================
# PLANNING AGENT
# ================================================================
_PLAN_SYSTEM = """You are an expert query planning agent for a RAG system with full multi-turn conversation awareness.
You MUST handle queries in ANY language. Always keep the rewritten query in its ORIGINAL language.

Given a user query and full conversation history, produce a JSON object with EXACTLY these keys:

{
  "query_type": "<direct|comparison|followup|summary|task>",
  "rewritten": "<fully self-contained query — ALL pronouns/references resolved, in the ORIGINAL language>",
  "sub_queries": ["<search string 1>", "<search string 2>"],
  "references_previous": true|false
}

TYPE RULES:
- "direct"    : brand-new single-concept question → sub_queries=[rewritten]
- "comparison": comparing ≥2 concepts → one sub_query per concept
- "followup"  : set references_previous=true AND query_type="followup" for ANY of:
    * query contains pronouns (it, they, that, this, those, he, she, them, its, their)
    * query is ≤6 words AND history is non-empty
    * query uses "more", "explain more", "tell me more", "elaborate", "continue", "what else"
    * query re-asks or rephrases the same topic as a previous turn
    * query asks about a sub-aspect of something previously discussed
- "summary"   : user wants broad overview of documents
- "task"      : user wants to act (send email, export, save, share)

REWRITING RULES:
- Resolve ALL pronouns using history
- "explain more" → "Explain [last topic] in more detail"
- Repeated question → produce clearest, most specific version
- ALWAYS produce a standalone rewritten query

Output ONLY valid JSON. No markdown. No explanation."""

def planning_agent(query, chat_history, available_files):
    recent = []
    for msg in chat_history[-8:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        recent.append(f"{role}: {msg['content'][:400]}")
    history_text = "\n".join(recent) if recent else "None"
    files_text   = ", ".join(available_files) if available_files else "None"
    user_prompt  = f"""Conversation history:\n{history_text}\n\nAvailable documents: {files_text}\n\nCurrent query: "{query}"\n\nProduce the JSON plan."""
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"system","content":_PLAN_SYSTEM},
                      {"role":"user","content":user_prompt}],
            max_tokens=400, temperature=0.0,
        )
        raw  = resp.choices[0].message.content.strip()
        raw  = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
        plan = json.loads(raw)
        for k in ("query_type","rewritten","sub_queries"):
            if k not in plan: raise ValueError(f"Missing key: {k}")
        if not isinstance(plan["sub_queries"], list) or not plan["sub_queries"]:
            plan["sub_queries"] = [plan["rewritten"]]
        return plan
    except Exception:
        return {"query_type":"direct","rewritten":query,"sub_queries":[query],"references_previous":False}

# ================================================================
# ★ THINKING MODE PIPELINE
# BM25 + FAISS → merge → cross-encoder rerank → adaptive MMR
# Only called when thinking_mode=True in query_system.
# ================================================================

# ── Comparison-query keywords for adaptive MMR ───────────────────
_COMPARISON_KEYWORDS = frozenset({
    "compare", "comparison", "difference", "differences", "differ",
    "vs", "versus", "contrast", "distinguish", "similarities",
    "similarity", "pros and cons", "advantages", "disadvantages",
})

def _is_comparison_query(query: str) -> bool:
    """Lightweight rule-based check: is this a comparison / multi-topic query?"""
    q = query.lower()
    # Direct keyword match
    if any(kw in q for kw in _COMPARISON_KEYWORDS):
        return True
    # Pattern: "X and Y", "X or Y" with question words
    if re.search(r'\b(?:what|how|which|explain)\b.*\b(?:and|or)\b', q):
        return True
    return False


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid for normalising cross-encoder logits to 0-1."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)


def thinking_pipeline(sub_queries: list, session: dict, top_k: int = 5) -> tuple:
    """
    Deep retrieval pipeline for Thinking Mode.

    Architecture (v2 — upgraded):
      1. Per sub-query independent FAISS + BM25 retrieval with scores
      2. Weighted hybrid merge (0.6 semantic + 0.4 keyword), dedup by chunk_id
      3. Adaptive MMR on merged pool → diverse candidate shortlist (~20)
      4. Cross-encoder rerank on shortlist → top_k final chunks
      5. Dynamic retrieval score from reranker confidence

    Args:
        sub_queries: list of query strings (from planning agent)
        session:     dict with 'chunks', 'index', 'bm25', 'embeddings'
        top_k:       number of final chunks to return

    Returns (selected_chunks, retrieval_score).
    Returns ([], 0.0) if no index or no chunks.
    """
    chunks     = session.get("chunks", [])
    index      = session.get("index")
    bm25       = session.get("bm25")

    if not chunks or index is None or not sub_queries:
        return [], 0.0

    embed_model   = get_model()
    primary_query = " ".join(sub_queries)          # combined for reranking
    per_query_k   = min(10, len(chunks))            # per sub-query fetch limit

    # ── Step 1+2: Per sub-query retrieval → weighted hybrid merge ─
    # Accumulate best hybrid score per chunk_id across all sub-queries
    chunk_best_score = {}                           # chunk_id → best hybrid score

    for sq in sub_queries:
        # — FAISS semantic retrieval with L2 distances —
        sq_vec = np.array(embed_model.encode([sq])).astype("float32")
        D, I   = index.search(sq_vec, per_query_k)

        faiss_raw = {}   # chunk_id → inverse-L2 similarity
        for dist, idx in zip(D[0], I[0]):
            if idx == -1 or idx >= len(chunks):
                continue
            faiss_raw[int(idx)] = 1.0 / (1.0 + float(dist))

        # — BM25 keyword retrieval with scores —
        bm25_raw = {}    # chunk_id → raw BM25 score
        if bm25 is not None:
            bm25_all  = bm25.get_scores(sq.lower().split())
            top_bm25  = sorted(range(len(bm25_all)),
                               key=lambda i: bm25_all[i], reverse=True)[:per_query_k]
            for i in top_bm25:
                if i < len(chunks) and float(bm25_all[i]) > 0:
                    bm25_raw[i] = float(bm25_all[i])

        # — Min-max normalisation within this sub-query —
        def _minmax(d):
            if not d:
                return {}
            vals = list(d.values())
            lo, hi = min(vals), max(vals)
            rng = hi - lo if hi > lo else 1.0
            return {k: (v - lo) / rng for k, v in d.items()}

        faiss_norm = _minmax(faiss_raw)
        bm25_norm  = _minmax(bm25_raw)

        # — Weighted hybrid: 0.6 semantic + 0.4 keyword —
        all_ids = set(faiss_norm.keys()) | set(bm25_norm.keys())
        for cid in all_ids:
            hybrid = 0.6 * faiss_norm.get(cid, 0.0) + 0.4 * bm25_norm.get(cid, 0.0)
            if cid not in chunk_best_score or hybrid > chunk_best_score[cid]:
                chunk_best_score[cid] = hybrid

    if not chunk_best_score:
        return [], 0.0

    # Sort by hybrid score → candidate pool
    sorted_candidates = sorted(chunk_best_score.items(),
                               key=lambda x: x[1], reverse=True)
    max_pool = min(30, len(sorted_candidates))
    pool_chunks = [chunks[cid] for cid, _ in sorted_candidates[:max_pool]]

    print(f"  [Thinking] {len(sub_queries)} sub-queries → "
          f"{len(chunk_best_score)} unique chunks → top {len(pool_chunks)} candidates")

    # ── Step 3: Adaptive MMR on candidate pool (BEFORE reranking) ─
    # Reduces pool to a diverse shortlist so cross-encoder runs on
    # fewer but broader candidates (avoids discarding reranked gems).
    is_comparison = _is_comparison_query(primary_query)
    if is_comparison:
        mmr_rel, mmr_div = 0.6, 0.4
        print("  [Thinking] Adaptive MMR: comparison mode (0.6/0.4)")
    else:
        mmr_rel, mmr_div = 0.8, 0.2
        print("  [Thinking] Adaptive MMR: precision mode (0.8/0.2)")

    mmr_target = min(20, len(pool_chunks))   # shortlist size for cross-encoder

    pool_embs   = embed_model.encode([c["text"] for c in pool_chunks]).astype("float32")
    query_emb   = embed_model.encode([primary_query]).astype("float32")[0]

    def _norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    query_emb_n = _norm(query_emb)
    pool_embs_n = np.array([_norm(e) for e in pool_embs])

    mmr_selected = []
    while len(mmr_selected) < mmr_target:
        best_i, best_score = -1, -float("inf")
        for i in range(len(pool_chunks)):
            if i in mmr_selected:
                continue
            relevance = float(np.dot(query_emb_n, pool_embs_n[i]))
            if mmr_selected:
                diversity = max(
                    float(np.dot(pool_embs_n[i], pool_embs_n[j]))
                    for j in mmr_selected
                )
            else:
                diversity = 0.0
            score = mmr_rel * relevance - mmr_div * diversity
            if score > best_score:
                best_score = score
                best_i     = i
        if best_i == -1:
            break
        mmr_selected.append(best_i)

    shortlist = [pool_chunks[i] for i in mmr_selected]

    if not shortlist:
        return [], 0.0

    print(f"  [Thinking] MMR shortlist: {len(shortlist)} chunks for cross-encoder")

    # ── Step 4: Cross-encoder rerank on shortlist ────────────────
    cross_encoder = get_cross_encoder()
    pairs     = [[primary_query, c["text"]] for c in shortlist]
    ce_scores = cross_encoder.predict(pairs)

    scored = sorted(zip(shortlist, ce_scores), key=lambda x: x[1], reverse=True)
    final_chunks = [chunk for chunk, _ in scored[:top_k]]
    final_ce     = [float(s) for _, s in scored[:top_k]]

    # ── Step 5: Dynamic retrieval score ──────────────────────────
    if final_ce:
        norm_scores     = [_sigmoid(s) for s in final_ce]
        retrieval_score = float(np.mean(norm_scores))
        retrieval_score = max(0.1, min(0.98, retrieval_score))
        print(f"  [Thinking] Dynamic retrieval score: {retrieval_score:.3f}  "
              f"(raw CE top-3: {[f'{s:.2f}' for s in final_ce[:3]]})")
    else:
        retrieval_score = 0.0

    return final_chunks, retrieval_score

# ================================================================
# QUERY SYSTEM
# ★ Modified to accept thinking_mode + session parameters.
# When thinking_mode=True the deep pipeline runs instead.
# When thinking_mode=False (default) — original logic unchanged.
# ================================================================
_ABS_COS_FLOOR = 0.25
_REL_CUTOFF    = 0.92

def query_system(sub_queries, index, chunks, embeddings,
                 k_per_query=4,
                 thinking_mode: bool = False,
                 session: dict = None):
    """
    Main retrieval entry point.

    thinking_mode=False (default): fast FAISS + hybrid keyword scoring.
    thinking_mode=True           : BM25 + FAISS + cross-encoder + MMR via
                                   thinking_pipeline(). Requires session dict.
    """

    # ── ★ THINKING MODE BRANCH ───────────────────────────────────
    if thinking_mode and session is not None:
        print(f"  [Thinking] Entering pipeline with {len(sub_queries)} sub-queries")
        chunks_selected, retrieval_score = thinking_pipeline(
            sub_queries if sub_queries else [""], session
        )

        if not chunks_selected:
            return [], [], False, 0.0

        chunk_texts = [c["text"] for c in chunks_selected]
        citations   = [
            {"file": c["source_file"], "pages": list(c.get("pages", []))}
            for c in chunks_selected
        ]
        any_garbled = any(c.get("has_garbled_math", False) for c in chunks_selected)

        return chunk_texts, citations, any_garbled, retrieval_score

    # ── NORMAL MODE (original logic unchanged) ───────────────────
    m = get_model()
    seen_chunk_ids = set()
    all_results    = []
    for sq in sub_queries:
        qe      = np.array(m.encode([sq])).astype("float32")
        fetch_k = min(k_per_query * 10, len(chunks))
        D, I    = index.search(qe, fetch_k)
        query_kws  = _extract_keywords(sq)
        candidates = []
        for dist, ci in zip(D[0], I[0]):
            if ci == -1: continue
            cos_sim  = cosine_similarity([qe[0]], [embeddings[ci]])[0][0]
            if cos_sim < _ABS_COS_FLOOR: continue
            chunk_kws = _extract_keywords(chunks[ci]["text"])
            kw_hits   = len(query_kws & chunk_kws)
            kw_ratio  = kw_hits / max(len(query_kws), 1)
            hybrid    = 0.6 * cos_sim + 0.4 * kw_ratio
            candidates.append((hybrid, cos_sim, ci))
        if not candidates: continue
        candidates.sort(reverse=True)
        best_hybrid   = candidates[0][0]
        hybrid_cutoff = best_hybrid * _REL_CUTOFF
        taken = 0
        for hybrid, cos_sim, ci in candidates:
            if taken >= k_per_query: break
            if hybrid < hybrid_cutoff and taken > 0: continue
            if ci in seen_chunk_ids: continue
            seen_chunk_ids.add(ci)
            taken += 1
            all_results.append((hybrid, ci))
    if not all_results: return [], [], False, 0.0
    all_results.sort(key=lambda x: x[0], reverse=True)
    all_results = all_results[:5]
    chunk_texts, citations, any_garbled = [], [], False
    top_score = all_results[0][0]
    for _, ci in all_results:
        chunk = chunks[ci]
        chunk_texts.append(chunk["text"])
        citations.append({"file": chunk["source_file"], "pages": list(chunk.get("pages", []))})
        if chunk.get("has_garbled_math"): any_garbled = True
    return chunk_texts, citations, any_garbled, float(top_score)

# ================================================================
# CONFLICT DETECTION
# ================================================================
_CONFLICT_SYSTEM = """You are a fact-consistency checker for a RAG system.
Given multiple text chunks from possibly different documents, check if they contain CONTRADICTORY factual claims.
Respond with ONLY a JSON object:
{
  "has_conflict": true|false,
  "conflict_summary": "<one-line description or empty string>",
  "severity": "low|medium|high"
}
HIGH = directly opposite facts. MEDIUM = disagree on key attributes. LOW = minor inconsistency.
Only flag REAL contradictions — different levels of detail is NOT a conflict.
Output ONLY valid JSON."""

def detect_conflicts(query, chunks, citations):
    if len(chunks) < 2: return None
    chunk_context = ""
    for i, (chunk, cit) in enumerate(zip(chunks[:5], citations[:5])):
        chunk_context += f"\n[Source {i+1}: {cit['file']}]\n{chunk[:500]}\n"
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"system","content":_CONFLICT_SYSTEM},
                      {"role":"user","content":f"Query: {query}\n\nChunks:\n{chunk_context}"}],
            max_tokens=200, temperature=0.0,
        )
        raw    = re.sub(r"^```(?:json)?","",resp.choices[0].message.content.strip()).rstrip("`").strip()
        result = json.loads(raw)
        return result if result.get("has_conflict") else None
    except Exception:
        return None

# ================================================================
# COMPLETENESS CHECK
# ================================================================
_COMPLETE_SYSTEM = """You are an answer completeness evaluator.
Given a query and a draft answer, determine if the answer is complete or seems truncated/missing key steps.
Respond with ONLY a JSON object:
{
  "is_complete": true|false,
  "completeness_score": <0.0 to 1.0>,
  "missing_aspects": ["aspect 1"],
  "needs_web_supplement": true|false,
  "reason": "<one-line explanation>"
}
Output ONLY valid JSON."""

def check_completeness(query, answer):
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"system","content":_COMPLETE_SYSTEM},
                      {"role":"user","content":f"Query: {query}\n\nAnswer:\n{answer[:1500]}"}],
            max_tokens=200, temperature=0.0,
        )
        raw = re.sub(r"^```(?:json)?","",resp.choices[0].message.content.strip()).rstrip("`").strip()
        return json.loads(raw)
    except Exception:
        return {"is_complete":True,"completeness_score":0.7,"missing_aspects":[],
                "needs_web_supplement":False,"reason":""}

# ================================================================
# CONFIDENCE SCORING
# ================================================================
def compute_confidence(retrieval_score, n_chunks, n_unique_sources,
                       completeness_score, web_verified=False,
                       conflict_detected=False, query_type="direct",
                       from_trusted_web=False):
    if retrieval_score <= 0.0:
        retrieval_pts = 0
    else:
        norm = max(0.0, (retrieval_score - 0.25) / (0.65 - 0.25))
        retrieval_pts = int(20 + norm * 25)
    retrieval_pts = min(45, retrieval_pts)
    source_pts    = min(20, max(0, n_unique_sources) * 8 + min(n_chunks, 3) * 2)
    complete_pts  = min(25, int(completeness_score * 25))
    web_pts       = 10 if from_trusted_web else (7 if web_verified else 0)
    conflict_pen  = 12 if conflict_detected else 0
    raw   = retrieval_pts + source_pts + complete_pts + web_pts - conflict_pen
    score = max(10, min(97, raw))
    breakdown = {"retrieval": retrieval_pts, "coverage": source_pts,
                 "completeness": complete_pts, "web_verify": web_pts,
                 "conflict_pen": conflict_pen}
    return score, breakdown

def conf_color(score):
    if score >= 82: return "#22c55e"
    if score >= 65: return "#fbbf24"
    return "#ef4444"

def conf_label(score):
    if score >= 82: return "High Confidence"
    if score >= 65: return "Moderate Confidence"
    if score >= 45: return "Low Confidence"
    return "Very Low — Verify Manually"

# ================================================================
# TRUSTED WEB SEARCH
# ================================================================
_TRUSTED_DOMAINS = {
    "wikipedia.org","britannica.com","scholarpedia.org","wikimedia.org",
    "nature.com","sciencedirect.com","pubmed.ncbi.nlm.nih.gov","arxiv.org",
    "researchgate.net","springer.com","ieee.org","mdpi.com","frontiersin.org",
    "scholar.google.com","jstor.org","wiley.com","plos.org","acm.org",
    "semanticscholar.org","ssrn.com","cambridge.org","oxfordacademic.com",
    "nationalgeographic.com","nasa.gov","nih.gov","who.int","un.org",
    "cdc.gov","fda.gov","nsf.gov","energy.gov","epa.gov","nist.gov",
    "europa.eu","worldbank.org","imf.org",
    "bbc.com","bbc.co.uk","reuters.com","apnews.com","theguardian.com",
    "nytimes.com","economist.com","washingtonpost.com","ft.com",
    "scientificamerican.com","wired.com","arstechnica.com","theverge.com",
    "investopedia.com","bloomberg.com","forbes.com",
    "coursera.org","khanacademy.org","edx.org","mit.edu","stanford.edu",
    "harvard.edu","ox.ac.uk",
    "docs.python.org","developer.mozilla.org","microsoft.com","cloud.google.com",
    "aws.amazon.com","docs.oracle.com","pytorch.org","tensorflow.org",
    "stackoverflow.com","geeksforgeeks.org","towardsdatascience.com",
    "medium.com","analyticsvidhya.com","machinelearningmastery.com","huggingface.co",
}

_BLOCKED_DOMAINS = {
    "baidu.com","qq.com","weibo.com","youku.com","bilibili.com",
    "zhihu.com","csdn.net","163.com","sohu.com","sina.com.cn",
    "naver.com","daum.net","kakao.com",
    "aliexpress.com","taobao.com","jd.com","pinduoduo.com",
    "amazon.com","ebay.com","wish.com","shopify.com","etsy.com",
    "pinterest.com","tumblr.com","tiktok.com","instagram.com",
    "facebook.com","twitter.com","x.com","reddit.com",
    "quora.com","answers.com","ask.com","ehow.com","wikihow.com",
    "about.com","livestrong.com","buzzfeed.com","boredpanda.com",
    "glassdoor.com","indeed.com","linkedin.com","crunchbase.com",
    "yelp.com","tripadvisor.com","trustpilot.com",
    "scribd.com","slideshare.net","issuu.com",
}

def _is_trusted(url):
    u = url.lower()
    if any(bd in u for bd in _BLOCKED_DOMAINS): return False
    if any(td in u for td in _TRUSTED_DOMAINS): return True
    if any(u.endswith(ext) or f".{ext}/" in u for ext in ("gov","edu","ac.uk","ac.in","ac.jp","edu.au")): return True
    return False

def _is_english_result(r):
    body = r.get("body","") or r.get("snippet","")
    if not body: return True
    return sum(1 for c in body if ord(c) < 128) / max(len(body),1) > 0.75

def _rewrite_query_for_web(query):
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role":"system","content":
                    "You are a search query optimizer. Given a user question, rewrite it "
                    "into a precise, unambiguous web search query that will return factual, "
                    "educational results from authoritative sources. "
                    "Rules:\n"
                    "- Add context words to disambiguate\n"
                    "- 'differentiate X and Y' means 'X vs Y comparison key differences'\n"
                    "- Remove vague pronouns\n"
                    "- Keep it concise (max 15 words)\n"
                    "- Output ONLY the rewritten query, nothing else."},
                {"role":"user","content":f"Rewrite this query for web search: \"{query}\""}
            ],
            max_tokens=60, temperature=0.0,
        )
        rewritten = resp.choices[0].message.content.strip().strip('"').strip("'")
        if len(rewritten) < 5: return query
        return rewritten
    except Exception:
        return query

def _compute_result_relevance(query, result):
    query_kws = _extract_keywords(query)
    if not query_kws: return 0.5
    title    = (result.get("title","") or "").lower()
    body     = (result.get("body","") or result.get("snippet","") or "").lower()
    combined = title + " " + body
    hits = sum(1 for kw in query_kws if kw in combined)
    keyword_ratio = hits / max(len(query_kws), 1)
    if len(set(title.split())) <= 2 and keyword_ratio < 0.3:
        return 0.1
    return min(1.0, keyword_ratio * 1.2 + 0.1)

_RESULT_RELEVANCE_THRESHOLD = 0.25

def _ddg_search_robust(query, n_results=8):
    search_query = _rewrite_query_for_web(query)
    raw = []
    for attempt in [
        lambda: list(DDGS().text(search_query, max_results=n_results * 3)),
        lambda: list(DDGS().text(query, max_results=n_results * 3)),
        lambda: list(DDGS().text(re.sub(r'[^\w\s]',' ',query)[:100], max_results=n_results*2)),
        lambda: list(DDGS().text(" ".join([w for w in query.split() if len(w)>3][:6]), max_results=n_results*2)),
    ]:
        if raw: break
        try: raw = attempt() or []
        except Exception: raw = []
    if not raw: return []
    filtered = [r for r in raw if _is_english_result(r)
                and not any(bd in r.get("href","").lower() for bd in _BLOCKED_DOMAINS)]
    scored = []
    for r in filtered:
        relevance = _compute_result_relevance(query, r)
        if relevance >= _RESULT_RELEVANCE_THRESHOLD:
            scored.append((relevance, r))
    trusted = [(rel, r) for rel, r in scored if _is_trusted(r.get("href",""))]
    neutral = [(rel, r) for rel, r in scored if not _is_trusted(r.get("href",""))]
    trusted.sort(key=lambda x: x[0], reverse=True)
    neutral.sort(key=lambda x: x[0], reverse=True)
    ordered = [r for _, r in trusted][:n_results]
    remaining = n_results - len(ordered)
    if remaining > 0:
        ordered.extend([r for _, r in neutral][:remaining])
    return ordered

def web_search_answer(query, n_results=8):
    results = _ddg_search_robust(query, n_results)
    if not results:
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":
                    f"Answer accurately and factually using your training knowledge. "
                    f"For ALL mathematical equations use LaTeX: $$...$$ for display, \\(...\\) for inline.\n"
                    f"Cite well-known sources where possible. Begin directly.\n\nQuestion: {query}"}],
                max_tokens=700,
            )
            answer = ("*(Answering from AI knowledge — no trusted web sources found.)*\n\n"
                      + resp.choices[0].message.content)
        except Exception as e:
            answer = f"Unable to retrieve answer: {e}"
        return answer, [], False

    has_trusted = any(_is_trusted(r.get("href","")) for r in results)
    web_context, web_sources = "", []
    for i, r in enumerate(results):
        title = r.get("title","") or r.get("body","")[:60]
        body  = r.get("body", r.get("snippet",""))[:600]
        url   = r.get("href", r.get("url",""))
        trusted_flag = _is_trusted(url)
        trust = "TRUSTED SOURCE" if trusted_flag else ""
        web_context += f"\n[Source {i+1}{(' -- '+trust) if trust else ''}: {title}]\n{body}\n"
        web_sources.append({"title":title,"url":url,"snippet":body[:200],"trusted":trusted_flag})

    is_comparison = any(w in query.lower() for w in [
        "difference","differentiate","compare","comparison","vs","versus","distinguish","contrast","differ",
    ])
    format_instruction = (
        "FORMAT: Present the answer as a **markdown table**. After the table, write a 1-2 sentence summary.\n"
        if is_comparison else ""
    )
    prompt = f"""You are Infera -- an intelligent, fact-driven research assistant.
Answer the question using ONLY the web search results below.
{format_instruction}CRITICAL RULES:
1. ONLY use information from TRUSTED sources. Non-trusted sources can supplement only.
2. Be accurate, well-structured, cite sources.
3. Do NOT invent or fabricate facts.
4. Be CONCISE — answer ONLY what is asked.
5. For ALL math: use LaTeX $$...$$ or \\(...\\).

Question: {query}

Web Search Results:
{web_context}

Provide a clear, concise, factual answer:"""
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            max_tokens=900,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"Error: {e}"
    return answer, web_sources, has_trusted

def web_verify_answer(query, doc_answer, n_results=4):
    results = _ddg_search_robust(query, n_results)
    if not results: return None, []
    web_context = "\n".join(f"\n[Web {i+1}]: {r.get('body','')[:350]}" for i,r in enumerate(results))
    web_sources = [{"title":r.get("title",""),"url":r.get("href","")} for r in results]
    prompt = f"""You are a fact-checker. Compare the document answer with web sources.
Query: {query}
Document answer: {doc_answer[:800]}
Web sources: {web_context}
Respond with ONLY JSON:
{{"consistent":true,"verified_claims":[],"corrections":[],"supplement":"","web_confidence_boost":0}}"""
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            max_tokens=400, temperature=0.0,
        )
        raw = re.sub(r"^```(?:json)?","",resp.choices[0].message.content.strip()).rstrip("`").strip()
        return json.loads(raw), web_sources
    except Exception:
        return None, []

def web_supplement_incomplete(query, doc_answer, missing_aspects):
    search_query = query + " " + " ".join(missing_aspects[:2])
    results = _ddg_search_robust(search_query, n_results=4)
    if not results: results = _ddg_search_robust(query, n_results=4)
    if not results: return "", []
    web_context = "\n".join(r.get("body","")[:300] for r in results[:3])
    web_sources = [{"title":r.get("title",""),"url":r.get("href","")} for r in results]
    prompt = f"""The following answer is incomplete.
Use web context to fill ONLY the missing parts. Be brief and accurate.
For ALL math use LaTeX $$...$$ or \\(...\\).
Original answer: {doc_answer[:600]}
Missing aspects: {', '.join(missing_aspects)}
Web context: {web_context}
Provide ONLY the supplementary information, labeled 'Web Supplement:'"""
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            max_tokens=400,
        )
        return resp.choices[0].message.content, web_sources
    except Exception:
        return "", []

# ================================================================
# EMAIL SENDER
# ================================================================
def send_email_smtp(to_address, subject, body, smtp_host, smtp_port, smtp_user, smtp_pass):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = to_address
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, to_address, msg.as_string())

# ================================================================
# LLM SYNTHESIS  — multilingual
# ================================================================
def _is_comparison_query(query):
    q = query.lower()
    return any(w in q for w in [
        "difference","differentiate","compare","comparison","vs","versus",
        "distinguish","contrast","differ","similarities and differences",
    ])

def get_length_instruction(query, query_type):
    q = query.lower()
    if query_type == "comparison" or _is_comparison_query(query):
        return ("Present the comparison in a **markdown table** with columns for each concept "
                "and rows for key attributes/differences. Add a brief summary after the table. "
                "Be concise -- do NOT add derivations or formulas unless explicitly asked.")
    if any(w in q for w in ["brief","short","summarize","quickly","what is","define"]):
        return "Answer in 2-3 sentences only. Be concise."
    if any(w in q for w in ["derive","derivation","proof","prove"]):
        return "Provide the derivation or proof step by step."
    if any(w in q for w in ["explain","detail","elaborate","how does","why"]):
        return "Answer in moderate detail with clear structure."
    return "Answer concisely and accurately. Only answer what is asked."

def generate_summary_groq(query, chunks, query_type="direct",
                           doc_has_garbled_math=False, response_language="English",
                           thinking_mode=False):
    context            = "\n\n".join(chunks)
    query_is_math      = _is_math_query(query)

    lang_clause = (
        f"IMPORTANT: You MUST respond entirely in {response_language}.\n"
        if response_language != "English" else ""
    )

    if doc_has_garbled_math and query_is_math:
        math_clause = ("IMPORTANT -- PDF math extraction is corrupted. "
                       "Reconstruct well-known equations from your knowledge. Use clean notation only.\n")
    elif doc_has_garbled_math:
        math_clause = "Note: Source contains garbled math. Rely on readable text and your knowledge.\n"
    else:
        math_clause = ""

    # ── ★ THINKING MODE: enhanced synthesis prompt ────────────────
    if thinking_mode:
        comparison_note = ""
        if query_type == "comparison" or _is_comparison_query(query):
            comparison_note = (
                "\nThis is a COMPARISON query. Present a clear side-by-side analysis. "
                "Use a markdown table for key differences, then expand on each point below the table.\n"
            )

        prompt = f"""You are Infera — an advanced document analysis engine operating in Deep Thinking Mode.
{lang_clause}{math_clause}{comparison_note}
You have been given multiple retrieved passages from the user's documents.
Your task is to SYNTHESIZE them into a comprehensive, well-structured answer.

INSTRUCTIONS:
1. **Structure your answer clearly** — use markdown headings (##, ###) to organize distinct topics or sections.
2. **Explain, don't just list** — for each key point, provide a brief explanation, not just a bullet point.
3. **Group related information** — combine similar ideas from different passages. Avoid repeating the same fact.
4. **Preserve important details** — do not over-summarize. Keep useful specifics, data, and examples.
5. **Maintain readability** — balance paragraphs with bullet points. Avoid dense walls of text.
6. **Do NOT copy chunks verbatim** — rephrase and integrate the information naturally.
7. **Stay grounded in context** — use ONLY the provided context. If information is insufficient, say so.
8. For ALL math: use LaTeX $$...$$ for display, \\(...\\) for inline.

Question: {query}

Context (retrieved passages):
{context}
"""
        max_tok = 1400
    else:
        # ── NORMAL MODE: original concise prompt (unchanged) ──────
        length_instruction = get_length_instruction(query, query_type)

        comparison_instruction = ""
        if query_type == "comparison" or _is_comparison_query(query):
            comparison_instruction = (
                "\nFORMAT: You MUST present the comparison as a **markdown table**. "
                "Use | Column1 | Column2 | format. Include rows for all key differences. "
                "After the table, write a 1-2 sentence summary. Do NOT add derivations.\n"
            )

        prompt = f"""You are an intelligent, precise assistant. You answer ONLY what is asked -- nothing more.
{lang_clause}{length_instruction}
{math_clause}{comparison_instruction}
CRITICAL RULES:
1. Use ONLY the provided context to answer.
2. If the context does not contain enough information, say so honestly.
3. Do NOT add derivations, formulas, proofs unless explicitly asked.
4. Keep your answer focused and relevant.
5. For ALL math: use LaTeX $$...$$ for display, \\(...\\) for inline.

Question: {query}

Context:
{context}
"""
        max_tok = 900

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tok,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error contacting Groq API: {e}"

def generate_answer(query, chunks, citations, query_type="direct",
                    doc_has_garbled_math=False, response_language="English",
                    thinking_mode=False):
    if not chunks:
        return None, []
    seen_texts, unique_chunks, unique_citations = set(), [], []
    for chunk_text_val, citation in zip(chunks, citations):
        key = " ".join(chunk_text_val.split())
        if key not in seen_texts:
            seen_texts.add(key)
            unique_chunks.append(chunk_text_val)
            unique_citations.append({"file": citation.get("file","Unknown"),
                                      "pages": citation.get("pages",[])})
    answer = generate_summary_groq(query, unique_chunks, query_type,
                                    doc_has_garbled_math, response_language,
                                    thinking_mode=thinking_mode)
    return answer, unique_citations

# ================================================================
# PERSONAL / SMALLTALK DETECTION
# ================================================================
_PERSONAL_PATTERNS = re.compile(
    r'\b(how are you|how r u|how do you feel|are you okay|are you fine|'
    r'what are you doing|what do you do|do you have feelings|do you feel|'
    r'are you human|are you a robot|are you alive|are you conscious|'
    r'do you have emotions|are you real|who are you|what are you|'
    r'are you happy|are you sad|are you tired|do you sleep|do you eat|'
    r'do you breathe|can you feel|what is your name|your name|'
    r'tell me about yourself|introduce yourself|'
    r'hello|hi there|hey infera|good morning|good evening|good afternoon|'
    r'thank you|thanks|thank u|appreciate it|great job|well done)\b',
    re.IGNORECASE
)

_PERSONAL_SYSTEM = """You are Infera — an Inference-Based Intelligence engine, not a human.
When users ask personal, emotional, or social questions, respond with warmth and clarity.
Always clarify you are an AI agent named Infera without human emotions or physical existence.
Be respectful, brief, and slightly witty. Never pretend to have feelings you don't have.
Do not use phrases like "As an AI language model". Just speak naturally as Infera."""

def is_personal_query(query):
    return bool(_PERSONAL_PATTERNS.search(query.strip()))

def handle_personal_query(query, response_language="English"):
    lang_note = f" Respond in {response_language}." if response_language != "English" else ""
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"system","content":_PERSONAL_SYSTEM + lang_note},
                      {"role":"user","content":query}],
            max_tokens=200, temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception:
        return ("I'm Infera — an AI inference engine, not a human. "
                "I don't have emotions or experiences, but I'm fully here to help you "
                "find answers from your documents or the web. What would you like to know?")

# ================================================================
# CONVERSATION CONTEXT BUILDER
# ================================================================
def build_conversation_context(messages):
    parts, web_sources = [], []
    for msg in reversed(messages):
        if msg.get("role") != "assistant": continue
        content = msg.get("content","")
        if not content: continue
        label = "Web-sourced answer" if msg.get("web_sources") else "Document-sourced answer"
        parts.append(f"[{label}]\n{content[:700]}")
        if msg.get("web_sources") and not web_sources:
            web_sources = msg["web_sources"]
        if len(parts) >= 4: break
    return "\n\n---\n\n".join(parts), web_sources

# ================================================================
# SESSION HELPERS
# ================================================================
MAX_FILE_SIZE_MB = 20

def new_session():
    """
    Create a fresh session dict.
    ★ Includes thinking_mode flag (user-controlled, default OFF)
    and bm25 slot for thinking pipeline.
    """
    return {
        "title":            "New Chat",
        "messages":         [],
        "file_names":       [],
        "rag_ready":        False,
        "index":            None,
        "chunks":           [],
        "embeddings":       None,
        "has_garbled_math": False,
        "detected_languages": [],
        # ── ★ Thinking Mode state ──
        "thinking_mode":    False,   # user toggles this via UI/API
        "bm25":             None,    # populated by ingest_doc_data_list
    }

def silent_ingest(file_storage_list, sess):
    """
    Ingest uploaded files into the session.
    Returns dict with 'ingested' (successful filenames) and 'failures' (list of {file, error}).
    ★ BUG I-3 FIX: Failures are captured and returned, never silently suppressed.
    ★ BUG I-4 FIX: Files producing zero usable chunks are marked as failed, not indexed.
    """
    all_chunks  = list(sess.get("chunks", []))
    any_garbled = sess.get("has_garbled_math", False)
    already     = set(sess.get("file_names", []))
    ingested    = []
    failures    = []  # BUG I-3: Track all failures with details
    det_langs   = list(sess.get("detected_languages", []))
    for filename, file_bytes, ext in file_storage_list:
        if filename in already:
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            doc_data = process_local_file(tmp_path)
            doc_data["source_file"] = filename
            if doc_data.get("has_garbled_math"):
                any_garbled = True
            sample = " ".join(doc_data["sentences"][:20])
            if sample.strip():
                lang_code = detect_language(sample)
                lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
                det_langs.append({"file": filename, "lang": lang_name})
            doc_chunks = chunk_text(doc_data)

            # ── BUG I-4 FIX: Validate chunks before marking as indexed ──
            if not doc_chunks or all(not c.get("text", "").strip() for c in doc_chunks):
                error_msg = f"No usable text extracted from '{filename}' — zero valid chunks produced."
                print(f"[INGEST FAIL] {error_msg}")
                failures.append({"file": filename, "error": error_msg})
                continue

            offset = len(all_chunks)
            for c in doc_chunks:
                c["chunk_id"] += offset
            all_chunks.extend(doc_chunks)
            ingested.append(filename)
        except Exception as e:
            # ── BUG I-3 FIX: Capture and report every failure ──
            import traceback
            error_msg = f"{type(e).__name__}: {e}"
            print(f"[INGEST FAIL] {filename}: {error_msg}")
            traceback.print_exc()
            failures.append({"file": filename, "error": error_msg})
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    if ingested and all_chunks:
        emb = create_embeddings(all_chunks)
        idx = build_faiss_index(emb)

        # ── ★ Rebuild BM25 after file ingestion too ──────────────
        bm25_obj = None
        if _BM25_AVAILABLE:
            tokenized = [c["text"].lower().split() for c in all_chunks]
            bm25_obj  = BM25Okapi(tokenized)

        sess.update({
            "index": idx, "chunks": all_chunks, "embeddings": emb,
            "rag_ready": True, "file_names": list(already) + ingested,
            "has_garbled_math": any_garbled, "detected_languages": det_langs,
            "bm25": bm25_obj,
        })

    # Store failures in session for API access
    sess["ingestion_failures"] = sess.get("ingestion_failures", []) + failures
    return {"ingested": ingested, "failures": failures}

# ================================================================
# MATH NORMALIZATION MODULE
# ================================================================
_UNICODE_TO_LATEX = {
    '∑': r'\sum',        '∫': r'\int',        '∬': r'\iint',
    '∭': r'\iiint',      '∮': r'\oint',        '√': r'\sqrt',
    '∞': r'\infty',      '∂': r'\partial',     '∇': r'\nabla',
    '∆': r'\Delta',      'α': r'\alpha',        'β': r'\beta',
    'γ': r'\gamma',      'δ': r'\delta',        'ε': r'\epsilon',
    'ζ': r'\zeta',       'η': r'\eta',          'θ': r'\theta',
    'λ': r'\lambda',     'μ': r'\mu',           'ν': r'\nu',
    'ξ': r'\xi',         'π': r'\pi',           'ρ': r'\rho',
    'σ': r'\sigma',      'τ': r'\tau',          'φ': r'\phi',
    'χ': r'\chi',        'ψ': r'\psi',          'ω': r'\omega',
    'Γ': r'\Gamma',      'Δ': r'\Delta',        'Θ': r'\Theta',
    'Λ': r'\Lambda',     'Ξ': r'\Xi',           'Π': r'\Pi',
    'Σ': r'\Sigma',      'Φ': r'\Phi',          'Ψ': r'\Psi',
    'Ω': r'\Omega',      '≠': r'\neq',          '≤': r'\leq',
    '≥': r'\geq',        '≈': r'\approx',       '≡': r'\equiv',
    '∈': r'\in',         '∉': r'\notin',        '⊂': r'\subset',
    '⊃': r'\supset',     '⊆': r'\subseteq',     '⊇': r'\supseteq',
    '∩': r'\cap',        '∪': r'\cup',          '∅': r'\emptyset',
    '∀': r'\forall',     '∃': r'\exists',       '¬': r'\neg',
    '∧': r'\wedge',      '∨': r'\vee',          '→': r'\to',
    '←': r'\leftarrow',  '↔': r'\leftrightarrow','⟹': r'\Rightarrow',
    '⟺': r'\Leftrightarrow','×': r'\times',    '÷': r'\div',
    '±': r'\pm',         '∓': r'\mp',           '·': r'\cdot',
    '°': r'^{\circ}',    '′': r"'",             '″': r"''",
    '½': r'\frac{1}{2}', '⅓': r'\frac{1}{3}',  '¼': r'\frac{1}{4}',
    '⁻': r'^{-}',        '²': r'^{2}',          '³': r'^{3}',
    '∝': r'\propto',     '∼': r'\sim',          '≅': r'\cong',
    '⊕': r'\oplus',      '⊗': r'\otimes',       '‖': r'\|',
    '⌊': r'\lfloor',     '⌋': r'\rfloor',       '⌈': r'\lceil',
    '⌉': r'\rceil',
}

_MATH_DELIM_RE = re.compile(
    r'\$\$[\s\S]*?\$\$'
    r'|\\\[[\s\S]*?\\\]'
    r'|\\\([\s\S]*?\\\)'
    r'|\$[^$\n]+?\$'
    r'|\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}',
    re.MULTILINE
)

_ASCII_MATH_RE = re.compile(
    r'\bsqrt\(([^)]+)\)'
    r'|\bpow\(([^,]+),\s*([^)]+)\)'
    r'|\bsum\(([^)]+)\)'
    r'|\bint\(([^)]+)\)',
    re.IGNORECASE
)

def normalize_math_in_text(text: str) -> str:
    if not text:
        return text
    for uni_char, latex_cmd in _UNICODE_TO_LATEX.items():
        if uni_char in text:
            text = _replace_outside_delimiters(text, uni_char, latex_cmd)
    def _ascii_to_latex(m):
        if m.group(0).lower().startswith('sqrt'):
            return r'\sqrt{' + m.group(1) + r'}'
        if m.group(0).lower().startswith('pow'):
            return m.group(2) + r'^{' + m.group(3) + r'}'
        if m.group(0).lower().startswith('sum'):
            return r'\sum ' + m.group(4)
        if m.group(0).lower().startswith('int'):
            return r'\int ' + m.group(5)
        return m.group(0)
    text = _ASCII_MATH_RE.sub(_ascii_to_latex, text)
    return text

def _replace_outside_delimiters(text: str, target: str, replacement: str) -> str:
    result = []
    last = 0
    for m in _MATH_DELIM_RE.finditer(text):
        segment = text[last:m.start()]
        result.append(segment.replace(target, replacement))
        result.append(m.group(0))
        last = m.end()
    result.append(text[last:].replace(target, replacement))
    return ''.join(result)

def extract_math_blocks(text: str) -> list:
    blocks = []
    for i, m in enumerate(_MATH_DELIM_RE.finditer(text)):
        raw = m.group(0)
        is_display = (
            raw.startswith('$$') or
            raw.startswith('\\[') or
            raw.startswith('\\begin')
        )
        blocks.append({
            'id':       f'math_{i:03d}',
            'type':     'block' if is_display else 'inline',
            'raw':      raw,
            'latex':    raw,
            'position': [m.start(), m.end()],
        })
    return blocks


# ================================================================
# THINKING MODE — SESSION TOGGLE HELPER
# Called by Flask route /api/set_thinking_mode
# ================================================================
def set_thinking_mode(sess: dict, enabled: bool) -> dict:
    """
    Toggle thinking mode for a session.
    Returns a status dict for the API response.
    """
    if enabled and not _BM25_AVAILABLE:
        return {
            "ok": False,
            "error": "rank_bm25 is not installed. Run: pip install rank-bm25",
            "thinking_mode": False,
        }
    sess["thinking_mode"] = bool(enabled)
    return {
        "ok":           True,
        "thinking_mode": sess["thinking_mode"],
        "message":      f"Thinking mode {'enabled' if enabled else 'disabled'}.",
    }