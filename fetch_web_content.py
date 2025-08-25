import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re

# ---------- CONFIG ----------
OUTPUT_DIR = "blog_posts"
TARGET_DIV_CLASS = "wp-site-blocks"  # <-- Adjust this to match your blog layout
URLS = [
    "https://fullstackdigital.io/blog/how-to-auto-register-custom-blocks-with-wordpress-6-8/",
    "https://fullstackdigital.io/blog/build-custom-post-types-with-structured-meta-fields-in-wordpress-without-3rd-party-plugins-like-acf/",
    "https://fullstackdigital.io/blog/how-to-build-multiple-acf-blocks-within-a-single-plugin-or-theme/",
    "https://fullstackdigital.io/blog/how-to-build-custom-blocks-using-advanced-custom-fields-acf-in-wordpress-block-creator-09/",
    # Add more URLs here
]
# ----------------------------


def slugify(url):
    """Create a safe filename from a URL."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    safe_path = re.sub(r'[^\w\-_.]', '_', path)
    return safe_path or "index"


def fetch_blog_post(url):
    """Fetch and extract the main content of a blog post."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0"
        }
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()

        soup = BeautifulSoup(res.text, 'html.parser')
        content_div = soup.find('div', class_=TARGET_DIV_CLASS)

        if content_div:
            text = content_div.get_text(separator="\n", strip=True)
        else:
            print(f"[!] No target div found for {url}. Using entire page text.")
            text = soup.get_text(separator="\n", strip=True)

        return text

    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None


def save_post(text, filename):
    """Save text to a .txt file in the output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[✓] Saved: {filepath}")


def main():
    for url in URLS:
        print(f"Fetching: {url}")
        content = fetch_blog_post(url)
        if content:
            filename = slugify(url)
            save_post(content, filename)


if __name__ == "__main__":
    main()
