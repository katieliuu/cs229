"""
01_download_raw_data script allows you to specify the URLs 
of specific xpt data files to download from NHANES 2017-2018 website.
The requests module is then used to store the content of each of 
those xpt files into src/data/raw_xpt folder. Information on how to
use the requests module was obtained from the module's documentation,
https://requests.readthedocs.io/en/latest/user/quickstart/.

"""
import requests
from pathlib import Path

# might have to do os makedir
data_dir = Path('src/data/raw_xpt')

nhanes_urls = ["https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt",
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.xpt",
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GHB_J.xpt",
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.xpt",
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/INS_J.xpt",
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HDL_J.xpt",
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/TRIGLY_J.xpt",
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/TCHOL_J.xpt",
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BIOPRO_J.xpt",
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/ALB_CR_J.xpt",
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HSCRP_J.xpt",
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PAQ_J.xpt",
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SMQ_J.xpt",
"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.xpt"]

def download(url):
    out = data_dir / url.split("/")[-1] # keep only the filename, like BMX_J.xpt
    with requests.get(url, stream=True) as r:
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)
    return out

for url in nhanes_urls:
    download(url)

