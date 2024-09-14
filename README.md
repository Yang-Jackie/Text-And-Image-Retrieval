# Tui Challenge 2024 - Text and Image Retrieval

## Building locally
1. Clone this repo using Git.
2. Download the `.bin` Faiss index file and `id2frames_v2.json` to the directory, using this [link](https://drive.google.com/drive/folders/17wvwUT8ESPLz3zdAWSnDj1wGHYmQJnIX?usp=sharing).
3. Change the directory of your extracted frames through the variable `EXTRACTED_PATH` in `app.py`.
4. Install dependencies using `pip install -r requirements.txt`. Note that in case you have already installed before this commit, please use `pip install --upgrade --force-reinstall -r requirements.txt` to reinstall/upgrade your dependencies.
5. Running the app using `python app.py`. Note that this may take a while and may consume a huge amount of CPU during initialization. That said, the retrieval process is pretty fast and decent :orz:

