import base64
import html
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler

st.set_page_config(
    page_title="AI Scout",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "welcomed" not in st.session_state:
    st.toast("Visca el Barça! 🔵🔴", icon="🌟")
    st.session_state.welcomed = True


def get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default


# =========================================================
# DATA
# =========================================================
@st.cache_data(show_spinner=False)
def load_players_data():
    candidates = [
        Path("role_scored_players.csv"),
        Path("cleaned_players_outfield.csv"),
        Path("cleaned_players_full.csv"),
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        return None, None

    df = pd.read_csv(found)
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")].copy()

    numeric_cols = [
        "Age", "Min", "starter_probability", "market_value_m", "estimated_market_value_m",
        "Gls_per90", "Ast_per90", "G+A_per90", "Sh_per90", "SoT_per90", "Int_per90",
        "TklW_per90", "Crs_per90", "Fls_per90", "Fld_per90", "Off_per90",
        "CrdY_per90", "CrdR_per90", "Saves_per90", "CS%", "Save%"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "role_group" not in df.columns:
        if "Pos_clean" in df.columns:
            df["role_group"] = df["Pos_clean"].astype(str)
        elif "Pos" in df.columns:
            df["role_group"] = df["Pos"].astype(str)
        else:
            df["role_group"] = "Unknown"

    if "league_name" not in df.columns and "Comp" in df.columns:
        df["league_name"] = df["Comp"].astype(str).str.split(n=1).str[-1]

    if "starter_probability" not in df.columns:
        df["starter_probability"] = 0.5

    if "market_value_m" in df.columns:
        df["display_value_m"] = df["market_value_m"]
        df["value_source"] = "Live market value"
    elif "estimated_market_value_m" in df.columns:
        df["display_value_m"] = df["estimated_market_value_m"]
        df["value_source"] = "Estimated value"
    else:
        minutes = df["Min"] if "Min" in df.columns else pd.Series(np.zeros(len(df)))
        prob = df["starter_probability"].fillna(0.5)
        norm_min = (minutes - minutes.min()) / (minutes.max() - minutes.min() + 1e-9)
        df["display_value_m"] = (2 + 38 * (0.62 * prob + 0.38 * norm_min)).round(1)
        df["value_source"] = "Estimated value"

    img_cols = [c for c in ["player_image_url", "image_url", "photo_url", "player_photo"] if c in df.columns]
    df["player_image_final"] = df[img_cols[0]] if img_cols else np.nan

    for col in ["Player", "Squad", "league_name", "Pos_clean", "role_group"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df, found.name


@st.cache_data(show_spinner=False)
def load_branding_data():
    branding_path = Path("club_branding.csv")
    if branding_path.exists():
        brand_df = pd.read_csv(branding_path)
        brand_df.columns = [c.strip().lower() for c in brand_df.columns]
        return brand_df
    return None


def default_brand_profiles():
    return {
        "Barcelona": {
            "primary": "#0A1F44",
            "secondary": "#A50044",
            "accent": "#EDBB00",
            "hero_image": "",
            "logo": "",
            "manager_name": "Hansi Flick",
            "manager_photo": "",
        },
        "Real Madrid": {
            "primary": "#111827",
            "secondary": "#5B21B6",
            "accent": "#F8FAFC",
            "hero_image": "",
            "logo": "",
            "manager_name": "Carlo Ancelotti",
            "manager_photo": "",
        },
        "Arsenal": {
            "primary": "#111827",
            "secondary": "#C1121F",
            "accent": "#F8FAFC",
            "hero_image": "",
            "logo": "",
            "manager_name": "Mikel Arteta",
            "manager_photo": "",
        },
        "Manchester City": {
            "primary": "#0F172A",
            "secondary": "#38BDF8",
            "accent": "#F8FAFC",
            "hero_image": "",
            "logo": "",
            "manager_name": "Pep Guardiola",
            "manager_photo": "",
        },
        "Liverpool": {
            "primary": "#111827",
            "secondary": "#C1121F",
            "accent": "#F9FAFB",
            "hero_image": "",
            "logo": "",
            "manager_name": "Arne Slot",
            "manager_photo": "",
        },
    }

import unicodedata

def strip_accents(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('utf-8')


def normalize_lookup_text(value: str) -> str:
    value = str(value or "").strip().lower()
    replacements = {
        "fc ": "",
        " fc": "",
        "cf ": "",
        " cf": "",
        " afc": "",
        " sc": "",
        " ac": "",
        " athletic club": " athletic",
        "manchester city": "man city",
        "manchester united": "man united",
        "paris saint germain": "psg",
        "paris saint-germain": "psg",
        "internazionale": "inter",
        "football club": "",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return " ".join(value.replace("-", " ").split())


def normalize_team_name(value: str) -> str:
    return normalize_lookup_text(value)

def team_match_score(target_team: str, candidate_team: str) -> int:
    target = normalize_team_name(target_team)
    candidate = normalize_team_name(candidate_team)
    if not target or not candidate:
        return 0
    if target == candidate:
        return 100
    if target in candidate or candidate in target:
        return 70
    target_tokens = set(target.split())
    candidate_tokens = set(candidate.split())
    overlap = len(target_tokens & candidate_tokens)
    if overlap:
        return overlap * 10
    return 0


@st.cache_data(show_spinner=False)
def fetch_url_as_data_uri(url: str):
    import requests
    import base64
    if not url or not str(url).startswith(("http://", "https://")):
        return None
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200 or not response.content:
            return None
        content_type = response.headers.get("Content-Type", "image/png").split(";")[0].strip()
        encoded = base64.b64encode(response.content).decode()
        return f"data:{content_type};base64,{encoded}"
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_page_og_image(page_url: str):
    if not page_url or not str(page_url).startswith(("http://", "https://")):
        return ""
    try:
        response = requests.get(page_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return ""
        html_text = response.text
        patterns = [
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
            r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
        ]
        for pattern in patterns:
            match = re.search(pattern, html_text, flags=re.IGNORECASE)
            if match:
                return html.unescape(match.group(1))
    except Exception:
        return ""
    return ""


@st.cache_data(show_spinner=False, ttl=3600)
def pick_store_kit_image(local_candidates, product_pages, remote_fallbacks=None):
    local_path = first_existing_path(local_candidates)
    if local_path:
        return local_path
    for page_url in product_pages:
        image_url = fetch_page_og_image(page_url)
        if image_url:
            return image_url
    for item in remote_fallbacks or []:
        if item:
            return item
    return ""


BARCA_BG_CANDIDATES = [
    r"C:\Users\moham\OneDrive\Desktop\ML-Projects\football\Barcelona.jpg",
    "Barcelona.jpg",
    "assets/Barcelona.jpg",
    "assets/barcelona.jpg",
]


def first_existing_path(candidates):
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return ""


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_wikipedia_thumbnail(search_query: str, size: int = 700):
    query = str(search_query or "").strip()
    if not query:
        return ""
    headers = {"User-Agent": "BarcaAIScout/1.0"}
    try:
        search_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={quote(query)}&utf8=&format=json&srlimit=1"
        )
        res = requests.get(search_url, headers=headers, timeout=15)
        if res.status_code != 200:
            return ""
        data = res.json()
        items = data.get("query", {}).get("search", [])
        if not items:
            return ""
        title = items[0].get("title", "")
        if not title:
            return ""
        img_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&titles={quote(title)}&prop=pageimages&format=json&pithumbsize={size}"
        )
        img_res = requests.get(img_url, headers=headers, timeout=15)
        if img_res.status_code != 200:
            return ""
        pages = img_res.json().get("query", {}).get("pages", {})
        for page in pages.values():
            thumb = page.get("thumbnail", {}) if isinstance(page, dict) else {}
            if thumb.get("source"):
                return thumb["source"]
    except Exception:
        return ""
    return ""


@st.cache_data(show_spinner=False, ttl=3600)
def pick_media_source(local_candidates, wiki_queries, remote_fallbacks=None, size: int = 700):
    local_path = first_existing_path(local_candidates)
    if local_path:
        return local_path
    for query in wiki_queries:
        result = fetch_wikipedia_thumbnail(query, size=size)
        if result:
            return result
    for item in remote_fallbacks or []:
        if item:
            return item
    return ""


def get_brand_for_team():
    # Hardcoded Premium FC Barcelona Hub configuration
    return {
        "primary": "#004D98",
        "secondary": "#A50044",
        "accent": "#EDBB00",
        "logo": pick_media_source(
            ["assets/barca_crest.png", "assets/barcelona_logo.png", "barca_crest.png"],
            [],
            ["https://upload.wikimedia.org/wikipedia/en/thumb/4/47/FC_Barcelona_%28crest%29.svg/500px-FC_Barcelona_%28crest%29.svg.png"],
        ),
        "hero_image": first_existing_path(BARCA_BG_CANDIDATES),
        "manager_name": "Hansi Flick",
        "manager_photo": pick_media_source(
            ["assets/hansi_flick.png", "assets/hansi_flick.jpg", "assets/managers/hansi_flick.png", "assets/managers/hansi_flick.jpg"],
            ["Hansi Flick football manager", "Hansi Flick"],
            [],
        ),
        "stadium": "Spotify Camp Nou",
    }


def asset_to_b64(path_str: str):
    if not path_str:
        return None
    path = Path(path_str)
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None


# =========================================================
# LOAD DATA EARLY
# =========================================================
df, source_name = load_players_data()
if df is None:
    st.error(
        "No data file found. Put one of these files next to app.py: "
        "role_scored_players.csv, cleaned_players_outfield.csv, or cleaned_players_full.csv"
    )
    st.stop()

brand_df = load_branding_data()
roles = sorted(df["role_group"].dropna().unique().tolist())
teams = sorted(df["Squad"].dropna().unique().tolist())
leagues = sorted(df["league_name"].dropna().unique().tolist()) if "league_name" in df.columns else []

if "selected_team_theme" not in st.session_state:
    st.session_state.selected_team_theme = "Barcelona"
if "page" not in st.session_state:
    st.session_state.page = "shortlist"
if "selected_player_name" not in st.session_state:
    st.session_state.selected_player_name = None


def resolve_image_source(path_or_url: str, uploaded_bytes: bytes | None = None):
    if uploaded_bytes is not None:
        encoded = base64.b64encode(uploaded_bytes).decode()
        return f"data:image/png;base64,{encoded}"
    value = str(path_or_url or "").strip()
    if not value:
        return ""
    if value.startswith(("http://", "https://")):
        return fetch_url_as_data_uri(value) or value
    local_b64 = asset_to_b64(value)
    if local_b64:
        suffix = Path(value).suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        return f"data:{mime};base64,{local_b64}"
    return ""

def get_hero_bg_style(src):
    if src:
        safe_src = src.replace("'", "%27")
        return f"url('{safe_src}')"
    return "none"

# =========================================================
# DYNAMIC THEME PREPARATION
# =========================================================
brand = get_brand_for_team()
hero_url = brand.get("hero_image", "")
logo_url = brand.get("logo", "")
logo_upload = None

hero_src = resolve_image_source(hero_url)
logo_src = resolve_image_source(logo_url, logo_upload.getvalue() if logo_upload is not None else None)
hero_bg_value = get_hero_bg_style(hero_src)

hero_bg_value = get_hero_bg_style(hero_src)

hero_bg_value = get_hero_bg_style(hero_src)

app_bg = (
    f"background-image: linear-gradient(rgba(0, 20, 50, .65), rgba(4, 14, 34, .85)), {hero_bg_value};"
    if hero_bg_value != "none"
    else f"background: radial-gradient(circle at 0% 0%, color-mix(in srgb, {brand['secondary']} 60%, transparent) 0%, transparent 40%), radial-gradient(circle at 100% 100%, color-mix(in srgb, {brand['primary']} 80%, transparent) 0%, transparent 40%), linear-gradient(135deg, #020813 0%, #061126 100%);"
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800;900&display=swap');
    :root {{
        --club-primary: {brand['primary']};
        --club-secondary: {brand['secondary']};
        --club-accent: {brand['accent']};
        --text: #F8FAFC;
        --muted: #94A3B8;
        --border: rgba(255,255,255,.12);
        --glass-bg: rgba(20, 30, 50, 0.4);
        --glass-border: rgba(255, 255, 255, 0.08);
    }}
    html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif !important; letter-spacing: -0.01em; }}
    .stApp {{ {app_bg} background-size: cover; background-position: center center; background-attachment: fixed; color: var(--text); }}
    .block-container {{ max-width: 1450px; padding-top: 2rem; padding-bottom: 3rem; }}
    h1, h2, h3, h4, h5, h6 {{ color: var(--text) !important; font-weight: 800; letter-spacing: -0.02em; }}
    div[data-testid='stSidebar'] {{ background: linear-gradient(180deg, rgba(0, 30, 80, 0.4) 0%, rgba(60, 0, 30, 0.4) 100%); backdrop-filter: blur(24px); border-right: 1px solid var(--glass-border); }}
    .hero {{ 
        display: grid; 
        grid-template-columns: 1fr 280px; 
        gap: 2rem; 
        align-items: center; 
        background: linear-gradient(135deg, rgba(0, 77, 152, 0.6), rgba(165, 0, 68, 0.5)); 
        border: 1px solid var(--border); 
        border-radius: 32px; 
        padding: 2.5rem 3rem; 
        box-shadow: 0 24px 50px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.2); 
        margin-bottom: 2rem; 
        backdrop-filter: blur(20px); 
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); 
    }}
    .hero:hover {{ transform: translateY(-6px); box-shadow: 0 32px 60px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.3); }}
    .hero-badge {{ 
        display: inline-block; 
        padding: 0.35rem 1rem; 
        border-radius: 999px; 
        background: linear-gradient(90deg, rgba(237, 187, 0, 0.2), rgba(237, 187, 0, 0.05)); 
        color: var(--club-accent); 
        border: 1px solid rgba(237, 187, 0, 0.3); 
        font-size: 0.85rem; 
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem; 
    }}
    [data-testid="stSidebar"] {{ display: none !important; }}
    [data-testid="collapsedControl"] {{ display: none !important; }}
    .hero-title {{ font-size: 3rem; font-weight: 900; line-height: 1.1; margin-bottom: 0.75rem; text-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
    .hero-sub {{ color: #E2E8F0; font-size: 1.15rem; line-height: 1.6; font-weight: 300; max-width: 800px; }}
    
    .metric-card {{ 
        background: var(--glass-bg); 
        backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border); 
        border-radius: 24px; 
        padding: 1.5rem; 
        box-shadow: 0 8px 32px rgba(0,0,0,0.3); 
        transition: all 0.3s ease; 
        position: relative;
        overflow: hidden;
    }}
    .metric-card::before {{
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, var(--club-accent), transparent);
        opacity: 0; transition: opacity 0.3s ease;
    }}
    .metric-card:hover {{ transform: translateY(-4px); box-shadow: 0 16px 40px rgba(0,0,0,0.4); border-color: rgba(255,255,255,0.2); }}
    .metric-card:hover::before {{ opacity: 1; }}
    .metric-label {{ color: var(--muted); font-size: 0.95rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }}
    .metric-value {{ font-size: 2rem; font-weight: 900; color: white; text-shadow: 0 2px 10px rgba(0,0,0,0.2); }}
    .metric-sub {{ color: var(--club-accent); font-size: 0.85rem; font-weight: 600; margin-top: 0.5rem; display: flex; align-items: center; gap: 0.25rem; }}
    
    .glass-card {{ 
        background: var(--glass-bg); 
        backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border); 
        border-radius: 28px; 
        padding: 1.5rem; 
        box-shadow: 0 16px 40px rgba(0,0,0,0.3); 
    }}
    
    .player-card {{ 
        background: var(--glass-bg); 
        backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border); 
        border-radius: 28px; 
        overflow: hidden; 
        box-shadow: 0 12px 32px rgba(0,0,0,0.3); 
        margin-bottom: 1.5rem; 
        transition: all 0.3s ease; 
    }}
    .player-card:hover {{ transform: translateY(-4px) scale(1.01); box-shadow: 0 20px 48px rgba(0,0,0,0.5); border-color: rgba(255,255,255,0.2); }}
    .player-top {{ 
        display: grid; 
        grid-template-columns: 130px 1fr; 
        gap: 1.5rem; 
        align-items: center; 
        padding: 1.5rem; 
        background: linear-gradient(135deg, rgba(255,255,255,0.05), transparent); 
    }}
    .player-avatar {{ 
        width: 130px; 
        height: 130px; 
        border-radius: 24px; 
        overflow: hidden; 
        background: linear-gradient(135deg, var(--club-primary), var(--club-secondary)); 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        font-size: 2.5rem; 
        font-weight: 900; 
        color: white; 
        border: 2px solid rgba(255,255,255,0.15);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }}
    .player-name {{ font-size: 1.75rem; font-weight: 900; margin-bottom: 0.35rem; text-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
    .player-meta {{ color: #CBD5E1; font-size: 1rem; line-height: 1.6; }}
    .player-bottom {{ padding: 0 1.5rem 1.5rem; }}
    .benefit-box, .profile-box {{ 
        background: rgba(0,0,0,0.2); 
        border: 1px solid rgba(255,255,255,0.08); 
        border-radius: 16px; 
        padding: 1rem 1.2rem; 
        margin-top: 1rem; 
        font-size: 0.95rem;
        line-height: 1.5;
    }}
    
    /* Modern Tables */
    .modern-table-container {{ overflow-x: auto; border-radius: 18px; border: 1px solid var(--border); box-shadow: 0 14px 30px rgba(0,0,0,0.3); background: rgba(0,0,0,0.2); backdrop-filter: blur(8px); margin-bottom: 1.5rem; }}
    .modern-table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.95rem; }}
    .modern-table th {{ background: linear-gradient(135deg, var(--club-primary), color-mix(in srgb, var(--club-secondary) 80%, black)); color: white; padding: 1rem; font-weight: 700; white-space: nowrap; position: sticky; top: 0; z-index: 2; }}
    .modern-table td {{ padding: 0.85rem 1rem; border-bottom: 1px solid rgba(255,255,255,0.06); color: var(--text); white-space: nowrap; }}
    .modern-table tbody tr {{ transition: background 0.2s ease, transform 0.2s ease; }}
    .modern-table tbody tr:hover {{ background: rgba(255,255,255,0.12); cursor: pointer; transform: scale(1.005) translateX(2px); z-index: 1; position: relative; }}
    .modern-table tbody tr:nth-child(even) {{ background: rgba(255,255,255,0.02); }}

    /* Broadcast Animations */
    @keyframes broadcast-pan {{
        0% {{ transform: scale(1.02) translate(0, 0); }}
        50% {{ transform: scale(1.06) translate(1px, -2px); }}
        100% {{ transform: scale(1.02) translate(-1px, 1px); }}
    }}
    .broadcast-image {{
        width: 100%;
        height: 280px;
        object-fit: cover;
        border-radius: 24px;
        box-shadow: 0 16px 32px rgba(0,0,0,0.4);
        border: 2px solid rgba(255,255,255,0.1);
        animation: broadcast-pan 12s infinite alternate ease-in-out;
        transform-origin: center center;
    }}
    .broadcast-container {{
        overflow: hidden;
        border-radius: 24px;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, color-mix(in srgb, var(--club-primary) 70%, transparent), color-mix(in srgb, var(--club-secondary) 30%, transparent));
        padding: 0;
    }}

    .stTabs [data-baseweb='tab-list'] {{ gap: 0.5rem; background: rgba(0,0,0,0.2); padding: 0.5rem; border-radius: 20px; margin-bottom: 2rem; border: 1px solid var(--border); box-shadow: inset 0 2px 10px rgba(0,0,0,0.2); }}
    .stTabs [data-baseweb='tab'] {{ background: transparent; border-radius: 14px; padding: 0.75rem 1.5rem; border: none; color: var(--muted); font-weight: 700; transition: all 0.3s ease; }}
    .stTabs [aria-selected='true'] {{ background: linear-gradient(135deg, var(--club-primary), var(--club-secondary)); border: 1px solid rgba(255,255,255,0.2); color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
    .stButton > button, .stDownloadButton > button {{ 
        background: linear-gradient(135deg, var(--club-primary), color-mix(in srgb, var(--club-primary) 60%, black)); 
        color: white; 
        border: 1px solid rgba(255,255,255,0.2); 
        border-radius: 16px; 
        font-weight: 800; 
        padding: 0.75rem 1.5rem; 
        transition: all 0.3s ease; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{ 
        background: linear-gradient(135deg, var(--club-secondary), color-mix(in srgb, var(--club-secondary) 60%, black)); 
        border-color: rgba(255,255,255,0.4); 
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }}
    
    .legend-card {{
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 1.25rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        display: flex;
        flex-direction: column;
        height: 100%;
        position: relative;
        overflow: hidden;
    }}
    .legend-card:hover {{
        transform: translateY(-12px) rotate(1deg);
        border-color: var(--club-accent);
        box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        background: rgba(30, 45, 75, 0.6);
    }}
    .legend-card:hover::after {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(135deg, transparent, rgba(237, 187, 0, 0.1));
        pointer-events: none;
    }}
    .legend-img-container {{
        width: 100%;
        aspect-ratio: 1;
        border-radius: 18px;
        overflow: hidden;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }}
    .legend-img-container img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}
    .legend-stats {{
        font-size: 0.8rem;
        color: var(--muted);
        margin-top: 0.5rem;
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }}
    .legend-stat-item {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .legend-trophy-icon {{ color: var(--club-accent); font-weight: 800; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_modern_html_table(df: pd.DataFrame, title: str = "") -> str:
    if df.empty:
        return ""
    headers = "".join([f"<th>{col}</th>" for col in df.columns])
    rows = ""
    for _, row in df.iterrows():
        cells = "".join([
            f"<td>{val:.2f}</td>" if isinstance(val, float) else f"<td>{val}</td>"
            for val in row
        ])
        rows += f"<tr>{cells}</tr>"
    title_html = f"<h4>{title}</h4>" if title else ""
    return f"{title_html}<div class='modern-table-container'><table class='modern-table'><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></div>"

# =========================================================
# LIVE ENRICHMENT
# =========================================================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_thesportsdb_players(player_name: str):
    """Returns a list of potential player matches from TheSportsDB."""
    # TheSportsDB works best with last name only
    name_clean = str(player_name or "").strip()
    if not name_clean:
        return []
    name_clean = strip_accents(name_clean)
    try:
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={quote(name_clean)}"
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("player") or []
    except Exception:
        return []




def pick_best_thesportsdb_match(items, team_name: str):
    if not items:
        return None
    best_item = None
    best_score = -1
    for item in items:
        item_score = team_match_score(team_name, item.get("strTeam", ""))
        if item_score > best_score:
            best_score = item_score
            best_item = item
    return best_item or items[0]


def enrich_player_live(row: pd.Series):
    name = str(row.get("Player", "")).strip()
    team = str(row.get("Squad", "")).strip()
    # Extract last name for best search results (TheSportsDB works best with last name)
    name_parts = name.split()
    last_name = name_parts[-1] if name_parts else name
    full_name = name

    # Strategy: Try full name first, then last name only (most reliable)
    items = fetch_thesportsdb_players(full_name)
    if not items:
        items = fetch_thesportsdb_players(last_name)
    match = pick_best_thesportsdb_match(items, team)

    if match:
        image_url = match.get("strCutout") or match.get("strThumb") or ""
        return {
            "image": image_url or row.get("player_image_final", np.nan),
            "nationality": match.get("strNationality"),
            "birth": match.get("dateBorn"),
            "height": match.get("strHeight"),
            "weight": match.get("strWeight"),
            "position_live": match.get("strPosition"),
            "description": match.get("strDescriptionEN"),
        }

    return {
        "image": row.get("player_image_final", np.nan),
        "nationality": None,
        "birth": None,
        "height": None,
        "weight": None,
        "position_live": None,
        "description": None,
    }


# =========================================================
# HELPERS
# =========================================================
def metric_card(label, value, sub=""):
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def initials(name: str):
    parts = str(name).split()
    return "".join([p[:1].upper() for p in parts[:2]]) if parts else "PL"


def role_metrics(role):
    mapping = {
        "FW": ["Gls_per90", "Ast_per90", "Sh_per90", "SoT_per90"],
        "MF": ["Ast_per90", "G+A_per90", "Crs_per90", "Int_per90", "TklW_per90"],
        "DF": ["Int_per90", "TklW_per90", "Crs_per90", "CrdY_per90"],
        "FW/MF": ["Gls_per90", "Ast_per90", "G+A_per90", "Crs_per90"],
        "DF/MF": ["Int_per90", "TklW_per90", "Ast_per90", "Crs_per90"],
        "DF/FW": ["Gls_per90", "Int_per90", "TklW_per90"],
        "GK": ["Saves_per90", "CS%", "Save%"],
    }
    return mapping.get(role, [])


def user_friendly_metric_labels():
    return {
        "Gls_per90": "Goals per 90",
        "Ast_per90": "Assists per 90",
        "G+A_per90": "Goal contributions per 90",
        "Sh_per90": "Shots per 90",
        "SoT_per90": "Shots on target per 90",
        "Int_per90": "Interceptions per 90",
        "TklW_per90": "Tackles won per 90",
        "Crs_per90": "Crosses per 90",
        "Fls_per90": "Fouls committed per 90",
        "Fld_per90": "Fouls won per 90",
        "Off_per90": "Offsides per 90",
        "CrdY_per90": "Yellow cards per 90",
        "CrdR_per90": "Red cards per 90",
        "Saves_per90": "Saves per 90",
        "CS%": "Clean-sheet %",
        "Save%": "Save %",
        "starter_probability": "Starter readiness",
        "display_value_m": "2025/26 scouting value (€m)",
        "Min": "Minutes played",
        "Age": "Age",
    }


def build_player_benefit(row, role):
    benefits = []
    if role in ["FW", "FW/MF"]:
        if row.get("Gls_per90", 0) >= 0.35:
            benefits.append("adds reliable goal output")
        if row.get("Sh_per90", 0) >= 2.5:
            benefits.append("creates consistent attacking threat")
        if row.get("Ast_per90", 0) >= 0.18:
            benefits.append("can both create and finish chances")
    if role in ["MF", "FW/MF", "DF/MF"]:
        if row.get("Ast_per90", 0) >= 0.18:
            benefits.append("improves chance creation")
        if row.get("Crs_per90", 0) >= 1.2:
            benefits.append("adds useful delivery from wider zones")
        if row.get("Int_per90", 0) >= 1.2 or row.get("TklW_per90", 0) >= 1.2:
            benefits.append("helps the team win the ball back quicker")
    if role in ["DF", "DF/MF", "DF/FW"]:
        if row.get("Int_per90", 0) >= 1.5:
            benefits.append("reads danger early and cuts passing lanes")
        if row.get("TklW_per90", 0) >= 1.5:
            benefits.append("strengthens defending in duels")
    if role == "GK":
        if row.get("Saves_per90", 0) >= 2.5:
            benefits.append("adds shot-stopping security")
        if row.get("Save%", 0) >= 70:
            benefits.append("brings efficient goal prevention")
    if not benefits:
        benefits = ["fits the role profile well", "offers a balanced performance profile"]
    text = ", ".join(benefits[:3])
    return text[0].upper() + text[1:] + "."


def compute_recommendations(df, target_team, role, target_league, budget_m, top_n, max_age, min_minutes, alpha):
    role_df = df[df["role_group"] == role].copy()
    if role_df.empty:
        return pd.DataFrame()

    if "Age" in role_df.columns:
        role_df = role_df[role_df["Age"] <= max_age]
    if "Min" in role_df.columns:
        role_df = role_df[role_df["Min"] >= min_minutes]
    role_df = role_df[role_df["display_value_m"] <= budget_m]

    if target_league != "Any league" and "league_name" in role_df.columns:
        league_mask = role_df["league_name"].astype(str).str.contains(str(target_league), case=False, na=False)
        if league_mask.any():
            role_df = role_df[league_mask]

    team_players = df[(df["Squad"] == target_team) & (df["role_group"] == role)].copy()
    candidates = role_df[role_df["Squad"] != target_team].copy()
    if candidates.empty:
        return pd.DataFrame()

    feature_candidates = [
        c for c in [
            "Age", "Gls_per90", "Ast_per90", "G+A_per90", "Sh_per90", "SoT_per90",
            "Fls_per90", "Fld_per90", "Off_per90", "Crs_per90", "Int_per90",
            "TklW_per90", "CrdY_per90", "CrdR_per90", "Saves_per90"
        ] if c in candidates.columns
    ]

    if len(feature_candidates) >= 3 and not team_players.empty:
        scaler = StandardScaler()
        scaler.fit(df[df["role_group"] == role][feature_candidates].fillna(0))
        team_scaled = scaler.transform(team_players[feature_candidates].fillna(0))
        cand_scaled = scaler.transform(candidates[feature_candidates].fillna(0))
        profile = team_scaled.mean(axis=0).reshape(1, -1)
        candidates["team_fit"] = cosine_similarity(cand_scaled, profile).flatten()
        low, high = candidates["team_fit"].min(), candidates["team_fit"].max()
        candidates["team_fit_norm"] = (candidates["team_fit"] - low) / (high - low + 1e-9)
    else:
        candidates["team_fit"] = 0.5
        candidates["team_fit_norm"] = 0.5

    prob = candidates["starter_probability"].fillna(0.5)
    low, high = prob.min(), prob.max()
    candidates["readiness"] = (prob - low) / (high - low + 1e-9)
    candidates["value_score"] = 1 - (candidates["display_value_m"] / budget_m).clip(0, 1)
    candidates["final_score"] = (
        alpha * candidates["readiness"]
        + (1 - alpha) * candidates["team_fit_norm"]
        + 0.15 * candidates["value_score"]
    )
    candidates["benefit_summary"] = candidates.apply(lambda r: build_player_benefit(r, role), axis=1)
    return candidates.sort_values("final_score", ascending=False).head(top_n)


def fifa_style_scores(row, role):
    def clamp(v):
        return int(max(30, min(99, round(v))))

    finishing = 55 + 42 * row.get("Gls_per90", 0) + 8 * row.get("SoT_per90", 0) + 5 * row.get("Sh_per90", 0)
    playmaking = 55 + 55 * row.get("Ast_per90", 0) + 20 * row.get("G+A_per90", 0) + 8 * row.get("Crs_per90", 0)
    defending = 55 + 18 * row.get("Int_per90", 0) + 18 * row.get("TklW_per90", 0)
    control = 55 + 20 * row.get("Fld_per90", 0) + 8 * row.get("Ast_per90", 0) + 6 * row.get("Crs_per90", 0)
    intensity = 55 + 14 * row.get("Fls_per90", 0) + 12 * row.get("Int_per90", 0) + 12 * row.get("TklW_per90", 0)
    discipline = 82 - 45 * row.get("CrdR_per90", 0) - 12 * row.get("CrdY_per90", 0)

    if role == "GK":
        finishing = 35
        playmaking = 45
        defending = 55 + 10 * row.get("Save%", 0) / 10
        control = 55 + 10 * row.get("CS%", 0) / 10
        intensity = 55 + 15 * row.get("Saves_per90", 0)
        discipline = 80 - 10 * row.get("CrdY_per90", 0)

    scores = {
        "Finishing": clamp(finishing),
        "Playmaking": clamp(playmaking),
        "Defending": clamp(defending),
        "Control": clamp(control),
        "Intensity": clamp(intensity),
        "Discipline": clamp(discipline),
    }
    overall = int(round(np.mean(list(scores.values()))))
    return overall, scores


def plot_radar(selected_player_row, role_df, role, key_prefix="shortlist"):
    metrics = [m for m in role_metrics(role) if m in role_df.columns]
    if len(metrics) < 3:
        st.info("Not enough role-specific metrics available for radar view.")
        return
    team_median = role_df[metrics].median(numeric_only=True)
    player_vals = selected_player_row[metrics].fillna(0)
    plot_df = pd.DataFrame({
        "Metric": metrics,
        "Player": player_vals.values,
        "Role median": team_median.values,
    })
    scaler = MinMaxScaler()
    plot_df[["Player", "Role median"]] = scaler.fit_transform(plot_df[["Player", "Role median"]])
    rename = user_friendly_metric_labels()
    plot_df["Metric"] = plot_df["Metric"].map(lambda x: rename.get(x, x))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=plot_df["Player"], theta=plot_df["Metric"], fill="toself", name="Selected player"))
    fig.add_trace(go.Scatterpolar(r=plot_df["Role median"], theta=plot_df["Metric"], fill="toself", name="Role median"))
    fig.update_layout(
        template="plotly_dark",
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 1])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_radar_{selected_player_row['Player']}")


def show_player_profile(row, all_df):
    role = row.get("role_group", "")
    role_df = all_df[all_df["role_group"] == role].copy()
    live = enrich_player_live(row)
    overall, scores = fifa_style_scores(row, role)
    labels = user_friendly_metric_labels()

    c1, c2 = st.columns([0.8, 1.2])
    with c1:
        player_image_src = resolve_image_source(live.get("image", ""))
        if player_image_src:
            st.markdown(
                f"<img src='{player_image_src}' style='width:100%;height:300px;object-fit:cover;border-radius:24px;border:1px solid rgba(255,255,255,.10);' />",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='player-avatar' style='width:100%;height:300px;border-radius:24px'>{initials(row['Player'])}</div>",
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown(f"## {row['Player']}")
        meta = [
            f"Club: {row.get('Squad', '—')}",
            f"League: {row.get('league_name', '—')}",
            f"Position: {row.get('role_group', '—')}",
            f"Age: {int(row['Age']) if pd.notna(row.get('Age')) else '—'}",
            f"2025/26 scouting value: €{row.get('display_value_m', np.nan):.1f}m" if pd.notna(row.get('display_value_m', np.nan)) else "Estimated value: —",
        ]
        if live.get("nationality"):
            meta.append(f"Nationality: {live['nationality']}")
        st.write(" • ".join(meta))

        m1, m2, m3 = st.columns(3)
        with m1:
            metric_card("Overall", str(overall), "App scouting rating")
        with m2:
            metric_card("Starter readiness", f"{row.get('starter_probability', 0.5):.2f}", "Higher is better")
        with m3:
            metric_card("Minutes played", f"{int(row.get('Min', 0))}", "Current sample")

        st.markdown(
            f"<div class='benefit-box'><b>How this player helps the team:</b> {row.get('benefit_summary', build_player_benefit(row, role))}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### FIFA-style scouting ratings")
    rating_df = pd.DataFrame({"Attribute": list(scores.keys()), "Rating": list(scores.values())})
    fig = px.bar(rating_df, x="Rating", y="Attribute", orientation="h", text="Rating", range_x=[0, 100])
    fig.update_traces(textposition="outside")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
    st.plotly_chart(fig, use_container_width=True, key=f"bars_{row['Player']}")

    t1, t2, t3 = st.tabs(["Simple summary", "Full statistics", "Radar comparison"])
    with t1:
        metrics = [m for m in role_metrics(role) if m in row.index]
        if metrics:
            for m in metrics:
                val = row.get(m, np.nan)
                if pd.notna(val):
                    st.markdown(
                        f"<div class='profile-box'><b>{labels.get(m, m)}</b><br>{val:.2f}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No role metrics available for this player.")
    with t2:
        stat_cols = [
            "Age", "Min", "starter_probability", "display_value_m",
            "Gls_per90", "Ast_per90", "G+A_per90", "Sh_per90", "SoT_per90",
            "Int_per90", "TklW_per90", "Crs_per90", "Fls_per90", "Fld_per90",
            "Off_per90", "CrdY_per90", "CrdR_per90", "Saves_per90", "CS%", "Save%",
        ]
        stat_cols = [c for c in stat_cols if c in row.index]
        stats_df = pd.DataFrame({
            "Statistic": [labels.get(c, c) for c in stat_cols],
            "Value": [row[c] for c in stat_cols],
        })
        st.markdown(render_modern_html_table(stats_df, "Full statistics"), unsafe_allow_html=True)
    with t3:
        plot_radar(row, role_df, role, key_prefix="profile")


# =========================================================
# HERO + MANAGER CARD
# =========================================================
HANSI_PROFILE = {
    "age": 61,
    "teams": ["Victoria Bammental", "TSG Hoffenheim", "Bayern Munich", "Germany", "FC Barcelona"],
    "trophies": [
        "2× Bundesliga",
        "1× DFB-Pokal",
        "1× UEFA Champions League",
        "1× DFL-Supercup",
        "1× UEFA Super Cup",
        "1× FIFA Club World Cup",
        "1× UEFA Men's Coach of the Year",
    ],
    "bio": "German coach known for high pressing, fast vertical attacks, and the 2020 Bayern sextuple.",
}


def render_hansi_profile_card():
    st.markdown("### Hansi Flick profile")
    hc1, hc2 = st.columns([0.75, 1.25])
    with hc1:
        if manager_src:
            st.image(manager_src, use_container_width=True)
    with hc2:
        st.markdown("**Hans-Dieter 'Hansi' Flick**")
        st.caption("FC Barcelona head coach")
        st.markdown(f"**Age:** {HANSI_PROFILE['age']}")
        st.markdown("**Teams coached:** " + ", ".join(HANSI_PROFILE["teams"]))
        st.markdown("**Brief:** " + HANSI_PROFILE["bio"])
        st.markdown("**Trophies / honours:**")
        for trophy in HANSI_PROFILE["trophies"]:
            st.markdown(f"- {trophy}")


manager_photo_url = brand.get("manager_photo", "")
manager_src = resolve_image_source(manager_photo_url)

manager_html = ""
if manager_src:
    manager_html = f"<div style='position: relative; overflow: hidden; border-radius: 20px; border: 2px solid rgba(255,255,255,0.15); box-shadow: 0 12px 30px rgba(0,0,0,0.4); aspect-ratio: 1/1; width: 100%'><a href='?show_hansi=1' target='_self'><img src='{manager_src}' title='Click for Hansi profile' style='width: 100%; height: 100%; object-fit: cover; transition: transform 0.3s ease;' onmouseover='this.style.transform=\"scale(1.05)\"' onmouseout='this.style.transform=\"scale(1)\"'/></a></div>"
else:
    manager_html = f"<div style='width:100px;height:100px;border-radius:20px;background:rgba(255,255,255,0.08);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:2rem;border: 1px solid rgba(255,255,255,0.1)'>{brand.get('manager_name','M')[0]}</div>"

import streamlit.components.v1 as components
import os
import base64

@st.cache_data(show_spinner=False)
def get_local_b64(filename, is_audio=False):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            if is_audio:
                return f"data:audio/webm;base64,{b64}"
            ext = filename.split('.')[-1].lower()
            return f"data:image/{ext};base64,{b64}"
    return "https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg"

cryff_img = get_local_b64("cryff.jpg")
pep_img = get_local_b64("pep.png")
buscq_img = get_local_b64("buscq.jpg")
neymar_img = get_local_b64("neymar.jpg")
messi_img = get_local_b64("messi.jpg")

anthem_b64 = get_local_b64("anthem.webm", is_audio=True)
if anthem_b64.startswith("http"):
    anthem_b64 = "https://upload.wikimedia.org/wikipedia/commons/3/30/Cant_del_Bar%C3%A7a.ogg"

col_a, col_b = st.columns([1, 15])
with col_a:
    components.html(f"""
    <style>
    body {{ margin: 0; padding: 0; overflow: hidden; background: transparent; display:flex; align-items:center; justify-content:center; }}
    .speaker-btn {{ width: 44px; height: 44px; border-radius: 22px; background: rgba(0,0,0,0.6); border: 2px solid #EDBB00; display: flex; align-items: center; justify-content: center; cursor: pointer; color: #fff; transition: all 0.3s; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }}
    .speaker-btn:hover {{ background: #A50044; transform: scale(1.05); }}
    .speaker-btn svg {{ width: 20px; height: 20px; fill: currentColor; }}
    </style>
    <div class="speaker-btn" id="sbtn" onclick="toggleAudio()" title="Play Barça Anthem">
      <svg viewBox="0 0 24 24"><path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/></svg>
    </div>
    <audio id="anthem" src="{anthem_b64}" loop></audio>
    <script>
        const a = document.getElementById('anthem');
        const b = document.getElementById('sbtn');
        let p = false;
        function toggleAudio() {{
            if(p) {{ a.pause(); p=false; b.style.background="rgba(0,0,0,0.6)"; }}
            else {{ a.play(); p=true; b.style.background="#A50044"; }}
        }}
    </script>
    """, height=50)

st.markdown(
    f"""<div style="all:initial;display:block;">
<style>
@keyframes legend-cycle {{
0%, 15% {{ opacity: 0; transform: translateY(10px); z-index: 0; }}
17%, 32% {{ opacity: 1; transform: translateY(0); z-index: 10; }}
34%, 100% {{ opacity: 0; transform: translateY(-10px); z-index: 0; }}
}}
.legends-carousel {{ position: relative; height: 55px; margin-top: 1.5rem; }}
.legend-msg {{ position: absolute; opacity: 0; animation: legend-cycle 25s infinite; display: flex; gap: 12px; align-items: center; background: rgba(0,0,0,0.3); padding: 6px 16px 6px 6px; border-radius: 50px; border: 1px solid rgba(255,255,255,0.08); max-width: 100%; }}
.legend-msg:nth-child(1) {{ animation-delay: 0s; }}
.legend-msg:nth-child(2) {{ animation-delay: 5s; }}
.legend-msg:nth-child(3) {{ animation-delay: 10s; }}
.legend-msg:nth-child(4) {{ animation-delay: 15s; }}
.legend-msg:nth-child(5) {{ animation-delay: 20s; }}
.legend-photo {{ width: 44px; height: 44px; border-radius: 50%; object-fit: cover; border: 2px solid var(--club-accent); box-shadow: 0 4px 12px rgba(0,0,0,0.3); flex-shrink: 0; }}
.legend-quote {{ font-size: 0.95rem; font-style: italic; color: #E2E8F0; line-height: 1.3; }}
.legend-name {{ font-weight: 800; color: var(--club-accent); margin-right: 4px; }}
</style>
<div class='hero'>
<div>
<div style="display:flex; align-items:center; gap:15px; margin-bottom:12px;">
<img src="{logo_src}" style="width:70px; height:70px; object-fit:contain; filter:drop-shadow(0 4px 12px rgba(0,0,0,0.4));" />
<div class='hero-badge' style="margin-bottom:0;">Premium Recruitment Hub</div>
</div>
<div class='hero-title'>Barça Player<br/>Recruiter</div>
<div class='hero-sub'>More than a club. A philosophy born in La Masia and perfected on the grandest stages. This hub empowers scouts to identify talent that breathes the Blaugrana DNA—where technical brilliance, tactical intelligence, and unwavering passion unite to craft the future of world football.</div>
<div class="legends-carousel">
<div class="legend-msg">
<img src="{cryff_img}" class="legend-photo" />
<div class="legend-quote"><span class="legend-name">Cruyff:</span> "Playing football is very simple, but playing simple football is the hardest thing there is."</div>
</div>
<div class="legend-msg">
<img src="{pep_img}" class="legend-photo" />
<div class="legend-quote"><span class="legend-name">Pep:</span> "I will forgive if the players cannot get it right, but not if they do not try hard."</div>
</div>
<div class="legend-msg">
<img src="{neymar_img}" class="legend-photo" />
<div class="legend-quote"><span class="legend-name">Neymar:</span> "I'm a guy who likes to play with joy. I'm always happy, and football gives me that."</div>
</div>
<div class="legend-msg">
<img src="{buscq_img}" class="legend-photo" />
<div class="legend-quote"><span class="legend-name">Busquets:</span> "I prefer to intercept and steal the ball. It's a job that needs doing, and I like doing it."</div>
</div>
<div class="legend-msg">
<img src="{messi_img}" class="legend-photo" />
<div class="legend-quote"><span class="legend-name">Messi:</span> "You have to fight to reach your dream. You have to sacrifice and work hard for it."</div>
</div>
</div>
</div>
<div class='glass-card' style='padding: 1.5rem; display: flex; flex-direction: column; justify-content: center; height: 100%;'>
<div style='display: grid; grid-template-columns: 80px 1fr; gap: 1.25rem; align-items: center;'>
{manager_html}
<div>
<div style='font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: {brand.get("accent", "#EDBB00")}; font-weight: 800; margin-bottom: 0.25rem;'>Head Coach</div>
<div style='font-weight:900; font-size:1.4rem; line-height: 1.1; margin-bottom: 0.25rem;'>{brand.get('manager_name', 'Hansi Flick')}</div>
<div style='color:var(--muted); font-size:0.95rem;'>FC Barcelona</div>
<div style='color: white; font-size: 0.8rem; margin-top: 0.75rem; opacity: 0.7;'>Click photo for profile</div>
</div>
</div>
</div>
</div>
</div>""",
    unsafe_allow_html=True,
)

if st.query_params.get("show_hansi", "0") == "1":
    with st.container():
        st.markdown("<div class='glass-card' style='margin-bottom: 2rem;'>", unsafe_allow_html=True)
        render_hansi_profile_card()
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("Close Hansi profile"):
            st.query_params.clear()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

manager_tab, analyst_tab, lab_tab, legends_tab = st.tabs(["Manager View", "Analyst View", "Player Comparison", "🏆 Barça Legends"])


# =========================================================
# MANAGER VIEW
# =========================================================
with manager_tab:
    if st.session_state.page == "shortlist":
        st.markdown("### Command Center")
        
        st.markdown("<div class='glass-card' style='margin-bottom: 2rem;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom:1rem;color:var(--club-accent);text-transform:uppercase;font-size:0.9rem;letter-spacing:1px;margin-top:0;'>Primary Objectives</h4>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            selected_team = st.selectbox(
                "Your club",
                teams,
                index=teams.index(st.session_state.selected_team_theme) if st.session_state.selected_team_theme in teams else 0,
            )
        with c2:
            selected_role = st.selectbox("Position needed", roles, index=roles.index("MF") if "MF" in roles else 0)
        with c3:
            selected_league = st.selectbox("Target league", ["Any league"] + leagues)
        with c4:
            budget_cap = int(max(150, np.ceil(df["display_value_m"].max())))
            budget_m = st.slider("Budget (million €)", 1, budget_cap, min(80, budget_cap))

        st.markdown("<hr style='border-color:rgba(255,255,255,0.1); margin: 1.5rem 0;'/>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom:1rem;color:var(--club-accent);text-transform:uppercase;font-size:0.9rem;letter-spacing:1px;'>Advanced Parameters</h4>", unsafe_allow_html=True)
        
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            max_age = st.slider("Maximum age", 17, 38, 32)
        with c6:
            min_minutes = st.slider("Minimum minutes played", 0, int(df["Min"].max()) if "Min" in df.columns else 2500, 250)
        with c7:
            top_n = st.slider("Shortlist size", 3, 20, 12)
        with c8:
            alpha = st.slider("Priority: readiness vs team fit", 0.0, 1.0, 0.55, 0.05)
            
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Generate Tactical Shortlist", use_container_width=True):
            st.session_state.selected_team_theme = selected_team
            recs = compute_recommendations(df, selected_team, selected_role, selected_league, budget_m, top_n, max_age, min_minutes, alpha)
            st.session_state.latest_recs = recs
            st.session_state.latest_team = selected_team
            st.session_state.latest_role = selected_role
            st.rerun()

        recs = st.session_state.get("latest_recs", pd.DataFrame())
        if isinstance(recs, pd.DataFrame) and not recs.empty:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                metric_card("Players shortlisted", str(len(recs)), "Manager-ready list")
            with m2:
                metric_card("Avg 2025/26 value", f"€{recs['display_value_m'].mean():.1f}m", recs["value_source"].iloc[0])
            with m3:
                metric_card("Average age", f"{recs['Age'].mean():.1f}", "Recruitment profile")
            with m4:
                metric_card("Best overall fit", f"{recs['final_score'].max():.2f}", "Top 2025/26 target")

            st.caption("All player values in this app are displayed as a 2025/26 scouting-value view based on the current dataset.")
            st.markdown("### Recommended players")
            labels = user_friendly_metric_labels()

            for i, row in recs.iterrows():
                live = enrich_player_live(row)
                avatar_src = resolve_image_source(live.get("image", ""))
                avatar = (
                    f"<img src='{avatar_src}' style='width:100%;height:100%;object-fit:cover;'/>"
                    if avatar_src
                    else initials(row["Player"])
                )
                metrics = []
                for m in role_metrics(st.session_state.get("latest_role", selected_role)):
                    if m in row.index and pd.notna(row[m]):
                        metrics.append(f"{labels.get(m, m)}: {row[m]:.2f}")
                highlight = " • ".join(metrics[:3]) if metrics else "Role-fit profile available"

                st.markdown(
                    f"""
                    <div class='player-card'>
                        <div class='player-top'>
                            <div class='player-avatar'>{avatar}</div>
                            <div>
                                <div class='player-name'>{row['Player']}</div>
                                <div class='player-meta'>{row['Squad']} • {row.get('league_name', 'League not available')}<br>
                                {row.get('role_group', '')} • Age {int(row['Age']) if pd.notna(row.get('Age')) else '—'} • {int(row['Min']) if pd.notna(row.get('Min')) else '—'} minutes<br>
                                2025/26 scouting value: €{row['display_value_m']:.1f}m</div>
                            </div>
                        </div>
                        <div class='player-bottom'>
                            <div class='benefit-box'><b>How this player helps the team:</b> {row['benefit_summary']}</div>
                            <div class='benefit-box'><b>Simple profile highlights:</b> {highlight}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Open full player page: {row['Player']}", key=f"open_{i}"):
                    st.session_state.selected_player_name = row["Player"]
                    st.session_state.page = "player_profile"
                    st.rerun()

            st.download_button(
                "Download shortlist CSV",
                data=recs.to_csv(index=False).encode("utf-8"),
                file_name=f"{st.session_state.get('latest_team','club')}_{st.session_state.get('latest_role','role')}_shortlist.csv",
                mime="text/csv",
            )

        elif "latest_recs" in st.session_state:
            st.warning("No players matched these filters. Try a bigger budget or a lower minutes threshold.")

    else:
        selected_name = st.session_state.get("selected_player_name")
        selected_df = df[df["Player"] == selected_name]
        if selected_df.empty:
            st.warning("Selected player was not found.")
        else:
            nav = st.columns([0.2, 0.8])
            with nav[0]:
                if st.button("← Back to shortlist"):
                    st.session_state.page = "shortlist"
                    st.rerun()
            with nav[1]:
                st.markdown("### Player page")
            row = selected_df.iloc[0]
            show_player_profile(row, df)


# =========================================================
# ANALYST VIEW
# =========================================================
with analyst_tab:
    st.markdown("### Technical scouting and analytics")
    st.caption("Interactive charts and deeper statistical review for analysts, scouts, and recruitment staff.")

    a1, a2, a3 = st.columns(3)
    with a1:
        role_view = st.selectbox("Role for analysis", roles, key="analyst_role")
    with a2:
        team_view = st.selectbox("Club for analysis", teams, key="analyst_team")

    role_df = df[df["role_group"] == role_view].copy()
    team_role_df = role_df[role_df["Squad"] == team_view].copy()

    with a3:
        pool_source = team_role_df if not team_role_df.empty else role_df
        pool = sorted(pool_source["Player"].dropna().unique().tolist())
        player_view = st.selectbox("Player", pool, key="analyst_player") if pool else None

    player_row = team_role_df[team_role_df["Player"] == player_view].head(1) if player_view and not team_role_df.empty else pd.DataFrame()
    benchmark_df = role_df.copy()

    if team_role_df.empty:
        st.info(f"No {role_view} players were found for {team_view}. Showing the full role pool instead.")
        display_df = role_df.copy()
        if player_view:
            player_row = role_df[role_df["Player"] == player_view].head(1)
    else:
        display_df = team_role_df.copy()

    c1, c2 = st.columns([1.05, 0.95])
    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Age vs estimated value")
        fig = px.scatter(
            display_df,
            x="Age",
            y="display_value_m",
            color="starter_probability",
            hover_data=[c for c in ["Player", "Squad", "league_name", "Min"] if c in display_df.columns],
            title=f"{team_view if not team_role_df.empty else 'Role pool'} players by age and estimated value",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key="analyst_scatter")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Radar comparison")
        if not player_row.empty:
            plot_radar(player_row.iloc[0], benchmark_df, role_view, key_prefix="analyst")
        else:
            st.info("Choose a player to display the radar comparison.")
        st.markdown("</div>", unsafe_allow_html=True)

    show_cols = [
        c for c in
        ["Player", "Squad", "league_name", "Age", "Min", "starter_probability", "display_value_m"]
        + [m for m in role_metrics(role_view) if m in display_df.columns]
        if c in display_df.columns
    ]
    final_display_df = display_df.sort_values("starter_probability", ascending=False)[show_cols].head(25).rename(columns=user_friendly_metric_labels())
    st.markdown(render_modern_html_table(final_display_df, "Role Candidates Analysis"), unsafe_allow_html=True)


# =========================================================
# STRATEGY LAB
# =========================================================
with lab_tab:
    st.markdown("### Player Comparison and Strategy lab")
    st.caption("Compare shortlist candidates and review overall squad fit strategy.")

    recs = st.session_state.get("latest_recs", pd.DataFrame())
    if recs.empty:
        st.info("Build a shortlist in the Manager View first to enable comparison tools.")
    else:
        c1, c2 = st.columns(2)
        
        # Get Barça squad for comparison
        barca_players = df[df["Squad"].str.contains("Barcelona", case=False, na=False)]["Player"].unique().tolist()
        if not barca_players:
            barca_players = recs["Player"].tolist() # Fallback if no barca players in data
            
        with c1:
            p1_name = st.selectbox("Current Barça Player (A)", barca_players, key="compare_a")
        with c2:
            p2_name = st.selectbox("Scouted Target (B)", recs["Player"].tolist(), key="compare_b")

        if p1_name and p2_name:
            p1_row = df[df["Player"] == p1_name].iloc[0]
            p2_row = recs[recs["Player"] == p2_name].iloc[0]

            col1, col2 = st.columns(2)
            
            # Helper to generate player comparison block
            def render_comparison_pane(p_row, recs_df, key_pref):
                live_info = enrich_player_live(p_row)
                img_src = resolve_image_source(live_info.get("image", ""))
                
                if img_src:
                    st.markdown(
                        f"<div class='broadcast-container'><img src='{img_src}' class='broadcast-image'/></div>", 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='broadcast-container' style='display:flex;align-items:center;justify-content:center;height:280px;font-size:4rem;font-weight:900;color:var(--text);'>{initials(p_row['Player'])}</div>",
                        unsafe_allow_html=True
                    )
                    
                st.markdown(f"**{p_row['Player']}** ({p_row['Squad']})")
                plot_radar(p_row, recs_df, p_row.get("role_group", ""), key_prefix=key_pref)
                
                # Efficiency Table
                eff_data = {
                    "Metric": ["Estimated Cost", "Starter Readiness", "Output Efficiency (G+A per 90)", "Age", "Minutes Analyzed"],
                    "Value": [
                        f"€{p_row.get('display_value_m', 0):.1f}m",
                        f"{p_row.get('starter_probability', 0)*100:.0f}%",
                        f"{p_row.get('G+A_per90', 0):.2f}",
                        f"{int(p_row.get('Age', 0))}",
                        f"{int(p_row.get('Min', 0))}"
                    ]
                }
                eff_df = pd.DataFrame(eff_data)
                st.markdown(render_modern_html_table(eff_df, "Investment stats"), unsafe_allow_html=True)
                
            with col1:
                render_comparison_pane(p1_row, recs, "lab_a")
            with col2:
                render_comparison_pane(p2_row, recs, "lab_b")

        st.markdown("---")
        st.markdown("#### Shortlist analytics")
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            avg_age = recs["Age"].mean() if "Age" in recs.columns else 0
            metric_card("Shortlist average age", f"{avg_age:.1f}", "Targeting youth" if avg_age < 24 else "Targeting experience")
        with lc2:
            total_val = recs["display_value_m"].sum() if "display_value_m" in recs.columns else 0
            metric_card("Total shortlist value", f"€{total_val:.1f}m", "2025/26 scouting view")
        with lc3:
            top_fit = recs["final_score"].max() if "final_score" in recs.columns else 0
            metric_card("Peak tactical fit", f"{top_fit:.2f}", "Maximum alignment found")

        st.markdown("#### Fit vs Readiness distribution")
        if "team_fit_norm" in recs.columns and "readiness" in recs.columns:
            fig = px.scatter(
                recs,
                x="team_fit_norm",
                y="readiness",
                size="display_value_m" if "display_value_m" in recs.columns else None,
                color="Age" if "Age" in recs.columns else None,
                hover_data=["Player", "Squad"],
                labels={"team_fit_norm": "Tactical Team Fit", "readiness": "Immediate Readiness"},
                title="Strategic positioning of shortlisted targets"
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key="lab_strategy_scatter")


# =========================================================
# BARÇA LEGENDS HALL OF FAME
# =========================================================
with legends_tab:
    st.markdown("## 🏆 FC Barcelona Hall of Fame")
    st.markdown("#### Celebrating the greatest names, the biggest nights, the shirts, and the Blaugrana identity")

    def legend_media(name: str, wiki_queries, local_candidates=None, remote_fallbacks=None, size: int = 900):
        src = pick_media_source(local_candidates or [], wiki_queries, remote_fallbacks=remote_fallbacks or [], size=size)
        return resolve_image_source(src)

    messi_media = legend_media(
        "Lionel Messi",
        ["Lionel Messi FC Barcelona", "Lionel Messi football player"],
        ["messi.jpg", "assets/legends/messi.jpg", "assets/messi.jpg"],
        ["https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Lionel_Messi_20180626.jpg/600px-Lionel_Messi_20180626.jpg"],
        1000,
    )
    cruyff_media = legend_media(
        "Johan Cruyff",
        ["Johan Cruyff FC Barcelona", "Johan Cruyff football player"],
        ["cryff.jpg", "assets/legends/cruyff.jpg", "assets/legends/cruyff.png", "assets/cruyff.jpg"],
        [],
        800,
    )
    busquets_media = legend_media(
        "Sergio Busquets",
        ["Sergio Busquets FC Barcelona", "Sergio Busquets football player"],
        ["buscq.jpg", "assets/legends/busquets.jpg", "assets/legends/busquets.png", "assets/busquets.jpg"],
        [],
        800,
    )
    neymar_media = legend_media(
        "Neymar Jr.",
        ["Neymar FC Barcelona", "Neymar football player Barcelona"],
        ["neymar.jpg", "assets/legends/neymar.jpg", "assets/legends/neymar.png", "assets/neymar.jpg"],
        [],
        800,
    )

    # Smaller icons section media
    xavi_media = legend_media(
        "Xavi Hernández",
        ["Xavi Hernández FC Barcelona", "Xavi Hernández"],
        ["assets/legends/xavi.jpg", "assets/legends/xavi.png", "assets/xavi.jpg"],
        ["https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Xavi_Hern%C3%A1ndez.jpg/600px-Xavi_Hern%C3%A1ndez.jpg"],
        600,
    )
    iniesta_media = legend_media(
        "Andrés Iniesta",
        ["Andrés Iniesta FC Barcelona"],
        ["assets/legends/iniesta.jpg", "assets/legends/iniesta.png", "assets/iniesta.jpg"],
        ["https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Andr%C3%A9s_Iniesta.jpg/600px-Andr%C3%A9s_Iniesta.jpg"],
        600,
    )
    ronaldinho_media = legend_media(
        "Ronaldinho",
        ["Ronaldinho Barcelona"],
        ["assets/legends/ronaldinho.jpg", "assets/legends/ronaldinho.png", "assets/ronaldinho.jpg"],
        ["https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Ronaldinho_11feb2007.jpg/600px-Ronaldinho_11feb2007.jpg"],
        600,
    )
    puyol_media = legend_media(
        "Carles Puyol",
        ["Carles Puyol FC Barcelona"],
        ["assets/legends/puyol.jpg", "assets/legends/puyol.png", "assets/puyol.jpg"],
        [],
        600,
    )
    suarez_media = legend_media(
        "Luis Suárez",
        ["Luis Suárez FC Barcelona"],
        ["assets/legends/suarez.jpg", "assets/legends/suarez.png", "assets/suarez.jpg"],
        [],
        600,
    )
    pique_media = legend_media(
        "Gerard Piqué",
        ["Gerard Piqué FC Barcelona"],
        ["assets/legends/pique.jpg", "assets/legends/pique.png", "assets/pique.jpg"],
        [],
        600,
    )

    st.markdown("---")
    st.markdown("### ✨ Featured Icons")
    col_feat1, col_feat2 = st.columns([1.0, 1.2])
    with col_feat1:
        st.markdown(
            f"""
            <div class="legend-card">
                <div class="legend-img-container" style="height: 480px;">
                    <img src="{messi_media}" />
                </div>
                <div style="font-size: 1.5rem; font-weight: 900; color: white; margin-top: 1rem;">Lionel Messi</div>
                <div style="font-size: 1.1rem; color: var(--club-accent); font-weight: 700;">The Greatest of All Time</div>
                <div class="legend-stats" style="font-size: 1rem; margin-top: 0.75rem;">
                    <div class="legend-stat-item"><span class="legend-trophy-icon">🏆</span> 35 Trophies with FC Barcelona</div>
                    <div class="legend-stat-item"><span class="legend-trophy-icon">⚡</span> Best Season: 2011/12 (73 Goals)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_feat2:
        st.markdown("#### The Definitive Blueprint")
        st.markdown(
            """
            The heart of the modern Barça era and the ultimate reference point for football excellence. 
            This Hall of Fame highlights his journey from La Masia to global immortality.
            
            - **8 Ballon d'Or awards**
            - **35 trophies with FC Barcelona**
            - **672 goals & 269 assists** for the first team
            """
        )
        messi_snapshot = pd.DataFrame(
            [
                {"Category": "Goals", "Value": 672},
                {"Category": "Assists", "Value": 269},
                {"Category": "Trophies", "Value": 35},
                {"Category": "La Liga", "Value": 10},
            ]
        )
        st.markdown(render_modern_html_table(messi_snapshot), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    def show_featured_card(column, title: str, subtitle: str, image_source: str, trophies: str, season: str):
        with column:
            st.markdown(
                f"""
                <div class="legend-card">
                    <div class="legend-img-container">
                        <img src="{image_source}" />
                    </div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: white;">{title}</div>
                    <div style="font-size: 0.9rem; color: #CBD5E1; margin: 0.25rem 0;">{subtitle}</div>
                    <div class="legend-stats">
                        <div class="legend-stat-item"><span class="legend-trophy-icon">🏆</span> {trophies} Trophies</div>
                        <div class="legend-stat-item"><span class="legend-trophy-icon">⚡</span> Best Season: {season}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    show_featured_card(f1, "Johan Cruyff", "The mastermind of Total Football.", cruyff_media, "13", "1973/74")
    show_featured_card(f2, "Sergio Busquets", "The ultimate Blaugrana pivot.", busquets_media, "32", "2010/11")
    show_featured_card(f3, "Neymar Jr.", "Magic, flair, and clinical finishing.", neymar_media, "8", "2014/15")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### 🏛️ Wall of Fame")
    st.caption("Celebrating the architects and warriors who built the Blaugrana legacy")
    
    def show_small_legend(column, title, image, trophies, season):
        with column:
            st.markdown(
                f"""
                <div class="legend-card" style="padding: 0.85rem;">
                    <div class="legend-img-container" style="margin-bottom: 0.5rem;">
                        <img src="{image}" />
                    </div>
                    <div style="font-size: 0.95rem; font-weight: 800; color: white; text-align: center;">{title}</div>
                    <div class="legend-stats" style="font-size: 0.72rem; align-items: center;">
                        <div>🏆 {trophies} Tr. | ⚡ {season}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    w1, w2, w3, w4, w5, w6 = st.columns(6)
    show_small_legend(w1, "Xavi", xavi_media, "25", "2008/09")
    show_small_legend(w2, "Iniesta", iniesta_media, "32", "2010/11")
    show_small_legend(w3, "Ronaldinho", ronaldinho_media, "5", "2005/06")
    show_small_legend(w4, "Puyol", puyol_media, "21", "2008/09")
    show_small_legend(w5, "Suárez", suarez_media, "13", "2015/16")
    show_small_legend(w6, "Piqué", pique_media, "31", "2014/15")

    st.markdown("---")
    camp_nou_media = legend_media(
        "Camp Nou",
        ["Camp Nou Barcelona stadium", "Spotify Camp Nou"],
        [
            r"C:\Users\moham\OneDrive\Desktop\ML-Projects\football\camp.jpg",
            "assets/camp.jpg",
            "assets/legends/camp.jpg",
            "assets/stadium/camp.jpg",
        ],
        ["https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Camp_Nou_-_Barcelona_%284%29.jpg/1280px-Camp_Nou_-_Barcelona_%284%29.jpg"],
        1200,
    )
    st.markdown("### Spotify Camp Nou")
    camp1, camp2 = st.columns([1.2, 1])
    with camp1:
        if camp_nou_media:
            st.image(camp_nou_media, use_container_width=True)
    with camp2:
        st.markdown(
            """
            **Stadium name:** Spotify Camp Nou  
            **City:** Barcelona, Catalonia, Spain  
            **Capacity identity:** One of football's most iconic stages and the symbolic home of the Blaugrana.

            Camp Nou is more than a venue. It is the theatre of Barça's positional play, academy culture, and greatest European nights. From Cruyff's ideas to Ronaldinho's revival and the Xavi–Iniesta–Busquets–Messi era, the stadium represents the emotional core of the club.
            """
        )

    st.markdown("---")
    st.markdown("### Club brief")
    club_cols = st.columns([1.15, 0.85])
    with club_cols[0]:
        st.markdown(
            f"""
            FC Barcelona was founded in **1899** and grew into one of the defining institutions of world football. The club identity is built around possession, technical quality, academy development, and the idea of **Més que un club**. In this hub, that identity translates into control, creativity, brave attacking play, and a strong midfield-first philosophy.

            **Home**: Barcelona, Catalonia  
            **Stadium**: **{brand.get('stadium', 'Spotify Camp Nou')}**  
            **Style DNA**: positional play, possession, width, and high technical security
            """
        )
        st.markdown(
            f"<div class='benefit-box'><b>Barça home:</b> {brand.get('stadium', 'Spotify Camp Nou')} remains the symbolic stage for the club's biggest European nights and the centre of the Blaugrana identity.</div>",
            unsafe_allow_html=True,
        )
    with club_cols[1]:
        metric_card("Home stadium", brand.get('stadium', 'Spotify Camp Nou'), "Blaugrana home")

    st.info("This hall of fame now focuses on Barça-only legends and a Camp Nou feature, with your local Messi, Neymar, Cruyff, Busquets, and stadium images wired in first.")

st.markdown(
    f"<div style='color:#DCE6FF;font-size:.9rem;margin-top:1.25rem;padding:.85rem 1rem;border-radius:16px;background:linear-gradient(135deg, rgba(165,0,68,.22), rgba(0,77,152,.26));border:1px solid rgba(255,255,255,.08);'>Barça AI Scouting Engine v2.1 • Inspired by Blaugrana excellence • Manager: {brand.get('manager_name')}</div>",
    unsafe_allow_html=True,
)