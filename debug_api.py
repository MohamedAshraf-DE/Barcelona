"""Quick diagnostic - run with: python debug_api.py"""
import requests

def check(label, url):
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        print(f"\n=== {label} ===")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list) and v:
                    print(f"  [{k}] -> list({len(v)})")
                    first = v[0]
                    if isinstance(first, dict):
                        for kk, vv in first.items():
                            print(f"      {kk}: {str(vv)[:100]}")
                else:
                    print(f"  {k}: {str(v)[:100]}")
    except Exception as e:
        print(f"ERROR: {e}")

# 1 - Team search
check("TEAM SEARCH: Barcelona", "https://www.thesportsdb.com/api/v1/json/1/searchteams.php?t=Barcelona")

# 2 - Player search
check("PLAYER SEARCH: Lewandowski", "https://www.thesportsdb.com/api/v1/json/1/searchplayers.php?p=Lewandowski")

print("\n--- DONE ---")
