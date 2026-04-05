import subprocess
import os

GENRE_PLAYLISTS = {
    "tamang_selo": "https://youtube.com/playlist?list=PL5y2Z_36oIaoHjOK0lZG4YNLgoraHQE_k",
    "bhajan":      "https://youtube.com/playlist?list=PL1aXlMS46BTHbdDi7vbqUFz4jBc20rhjV",
    "deuda":       "https://youtube.com/playlist?list=PLfJBvoNea3pE--RriRnnHd0JtioprGKbZ",
    "newari":      "https://youtube.com/playlist?list=PLpd2lctDternG_eZCQ1fiVbgrSRhEcK6w",
    "tharu":       "https://youtube.com/playlist?list=PL3Zb3_fHlYMGO5TMflaY4wMR03CqqOwD1",
    "lok_dohori":  "https://youtube.com/playlist?list=PL4QNnZJr8sROwteWJMT3dXhfayzM67SGK&si=NDPajajzlR2Ml0Zm",   
}

def download_genre(genre, url, max_songs=50):
    if not url:
        print(f"Skipping {genre} — no URL yet")
        return
    
    output_dir = f"data/raw/{genre}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nDownloading {genre}...")
    subprocess.run([
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--playlist-end", str(max_songs),  # downloads all if playlist has less than 50
        "-o", f"{output_dir}/%(title)s.%(ext)s",
        url
    ])
    print(f"Done: {genre}")

if __name__ == "__main__":
    for genre, playlist_url in GENRE_PLAYLISTS.items():
        if playlist_url:  # Only download if the URL is not empty
            download_genre(genre, playlist_url)