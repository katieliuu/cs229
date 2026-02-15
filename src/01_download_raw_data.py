# %% 
from pathlib import Path
import requests

RAW_DIR = Path("data/raw_xpt")
RAW_DIR.mkdir(parents=True, exist_ok=True)

XPT_URLS = [
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt",
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.xpt"
]

def download(url: str) -> Path:
    out = RAW_DIR / url.split("/")[-1]
    if out.exists():
        print(f"✓ {out.name} (skip)")
        return out
    print(f"↓ {out.name}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out

for u in XPT_URLS:
    download(u)
# %%
