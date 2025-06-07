import requests
from pathlib import Path
from tqdm import tqdm

try:
    from importlib.resources import files  # Python 3.9+
except ImportError:
    from importlib_resources import files  # For Python <3.9: pip install importlib-resources

def download_model(model_url, model_dir):
    filename = model_url.split("/")[-1]
    dest_path = model_dir / filename

    print(f"[I] Downloading model from: {model_url}")

    with requests.get(model_url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(dest_path, "wb") as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=f"Downloading {filename}"
        ) as progress:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))

    print(f"[I] Model saved to: {dest_path}")

def main():
    try:
        text_file = files("scripts") / "current_models.txt"
    except ModuleNotFoundError as e:
        raise RuntimeError("Cannot find scripts.current_models.txt") from e

    model_urls = [line.strip() for line in text_file.read_text().splitlines() if line.strip()]
    if not model_urls:
        raise ValueError("No model URLs found in current_models.txt")

    default_dir = Path.home() / ".lv_loanword_detection" / "pretrained_models"
    user_input = input(f"[?] Where should the models be saved? (default: {default_dir}): ").strip()
    model_dir = Path(user_input) if user_input else default_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    for model_url in model_urls:
        download_model(model_url, model_dir)

if __name__ == "__main__":
    main()
