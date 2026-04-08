import requests

urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Johan_Cruyff_1974c.jpg/400px-Johan_Cruyff_1974c.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Josep_Guardiola_in_1992.jpg/400px-Josep_Guardiola_in_1992.jpg"
]

for url in urls:
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        print(f"Status for {url}: {r.status_code}")
    except Exception as e:
        print(f"Error for {url}: {e}")
