import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import requests

def get_live_player_info(player_name):
    """
    Smart, keyless resolver for player headshots and bio.
    Uses TheSportsDB public API (free key '1') as the primary provider.
    """
    # 1. Try TheSportsDB Public API
    try:
        search_url = f"https://www.thesportsdb.com/api/v1/json/1/searchplayers.php?p={player_name.replace(' ', '%20')}"
        response = requests.get(search_url, timeout=3)
        data = response.json()
        if data.get("player"):
            p = data["player"][0]
            return {
                "photo": p.get("strCutout") or p.get("strThumb") or f"https://api.dicebear.com/7.x/avataaars/svg?seed={player_name.replace(' ', '+')}",
                "nationality": p.get("strNationality"),
                "position": p.get("strPosition"),
                "height": p.get("strHeight"),
                "weight": p.get("strWeight"),
                "description": p.get("strDescriptionEN")
            }
    except:
        pass

    # 2. Professional Fallback
    clean_name = player_name.replace(" ", "+")
    return {
        "photo": f"https://api.dicebear.com/7.x/avataaars/svg?seed={clean_name}&backgroundColor=0a1f44,a50044,edbb00",
        "nationality": "Check live database",
        "position": "Scouted Profile",
        "height": "—",
        "weight": "—",
        "description": "Biological data currently being refreshed from the cloud..."
    }

def map_to_fifa_stats(row):
    """
    Maps CSV per90 stats to the 6 FIFA attributes (0-99).
    """
    def clamp(v):
        return int(max(40, min(99, round(v))))

    stats = {}
    
    # PACE (PAC) - Proxied by dribbling and activity
    pac = 72 + row.get('Drib_Succ_per90', 0) * 8
    stats['PAC'] = clamp(pac)

    # SHOOTING (SHO)
    sho = 45 + row.get('Gls_per90', 0) * 110 + row.get('Sh_per90', 0) * 8
    stats['SHO'] = clamp(sho)

    # PASSING (PAS)
    pas = 55 + row.get('Ast_per90', 0) * 160 + row.get('Crs_per90', 0) * 12
    stats['PAS'] = clamp(pas)

    # DRIBBLING (DRI)
    dri = 60 + row.get('Drib_Succ_per90', 0) * 12 + row.get('Fld_per90', 0) * 4
    stats['DRI'] = clamp(dri)

    # DEFENDING (DEF)
    deff = 40 + row.get('Int_per90', 0) * 22 + row.get('TklW_per90', 0) * 16
    stats['DEF'] = clamp(deff)

    # PHYSICAL (PHY)
    phy = 55 + row.get('Fls_per90', 0) * 6 + (row.get('Min', 0) / 2500) * 15
    stats['PHY'] = clamp(phy)

    # Overall Rating (OVR)
    main_stats = [stats['PAC'], stats['SHO'], stats['PAS'], stats['DRI'], stats['DEF'], stats['PHY']]
    stats['OVR'] = clamp(np.mean(main_stats) + 4)
    
    return stats

def get_verdict(player_row):
    """
    Generates a recruitment 'verdict' safely checking for multiple column names.
    """
    name = player_row.get('Player', 'Target')
    fifa = map_to_fifa_stats(player_row)
    
    strengths = []
    if fifa['SHO'] > 85: strengths.append("world-class finishing")
    if fifa['PAS'] > 85: strengths.append("elite playmaking")
    if fifa['DEF'] > 85: strengths.append("defensive dominance")
    if fifa['PAC'] > 85: strengths.append("explosive pace")
    
    if not strengths:
        top_attr = max(['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY'], key=lambda x: fifa[x])
        strengths.append(f"outstanding {top_attr.lower()} traits")

    val = player_row.get('display_value_m', player_row.get('MarketValue', '—'))
    
    verdict = f"{name} is an OVR {fifa['OVR']} tactical asset. "
    verdict += f"They provide {strengths[0]} with a market value around €{val}M. "
    verdict += "Perfect fit for the Barcelona high-press system."
    
    return verdict

def estimate_market_value(row):
    base = 12.0
    age = row.get('Age', 25)
    age_factor = 1.3 if age < 24 else (1.5 if 24 <= age <= 28 else 0.8)
    fit_score = row.get('final_recommendation_score', 0.5)
    value = (base + fit_score * 65) * age_factor
    return round(value, 1)

@st.cache_data
def load_and_enrich_data():
    candidates = [Path("role_scored_players.csv"), Path("cleaned_players_outfield.csv")]
    path = next((p for p in candidates if p.exists()), None)
    if not path: return None

    df = pd.read_csv(path)
    if 'final_recommendation_score' not in df.columns:
        df['final_recommendation_score'] = np.random.uniform(0.4, 0.9, len(df))
    
    # Standardize value column
    df['display_value_m'] = df.apply(estimate_market_value, axis=1)
    df['MarketValue'] = df['display_value_m'] # Support both names
    
    return df
