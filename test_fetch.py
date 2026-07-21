import requests
from bs4 import BeautifulSoup

url = "http://hrlibrary.umn.edu/instree/auncharter.html"
headers = {"User-Agent": "Mozilla/5.0"}

resp = requests.get(url, headers=headers, timeout=20)
resp.encoding = resp.apparent_encoding 
soup = BeautifulSoup(resp.text, "html.parser")

for tag in soup(["script", "style"]):
    tag.decompose()
text = soup.get_text(separator="\n", strip=True)

print(len(text))
print(text[:500])