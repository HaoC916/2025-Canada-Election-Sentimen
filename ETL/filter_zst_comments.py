"""
*
* Program: filter_comments_zst.py
*
* Modified Date: November 2025
*
* Purpose: Step 1a of Reddit ETL
*    Stream RC_YYYY-MM.zst with Zstandard,
*    keep only target subreddits and non-deleted comments,
*    write out filtered JSONL for later Spark processing.
*
"""

import argparse
import json
import zstandard as zstd

TARGET = {
    "canada", "canadapolitics", "canadianpolitics",
    "lpc", "cpc", "ndp", "canadianconservative", "canadanews"
}

def stream_filter(infile, outfile):
    kept = 0
    seen = 0
    bad = 0

    with open(infile, "rb") as fh, open(outfile, "w", encoding="utf-8") as out:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            buffer = b""
            while True:
                chunk = reader.read(16384)
                if not chunk:
                    break
                buffer += chunk
                while True:
                    pos = buffer.find(b"\n")
                    if pos < 0:
                        break
                    line, buffer = buffer[:pos], buffer[pos+1:]
                    seen += 1
                    try:
                        obj = json.loads(line)
                        sub = str(obj.get("subreddit", "")).lower()
                        if sub in TARGET:
                            body = str(obj.get("body", "")).strip()
                            if body and body not in ("[deleted]", "[removed]"):
                                out.write(line.decode("utf-8", errors="ignore"))
                                out.write("\n")
                                kept += 1
                    except Exception:
                        bad += 1
                        continue
            # flush last line if file didn't end with '\n'
            if buffer:
                try:
                    obj = json.loads(buffer)
                    sub = str(obj.get("subreddit", "")).lower()
                    if sub in TARGET:
                        body = str(obj.get("body", "")).strip()
                        if body and body not in ("[deleted]", "[removed]"):
                            out.write(buffer.decode("utf-8", errors="ignore"))
                            out.write("\n")
                            kept += 1
                except Exception:
                    bad += 1

    print(f"Finished! seen={seen:,} kept={kept:,} bad_json={bad:,}")
    print(f"Output: {outfile}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", help="path to RC_YYYY-MM.zst")
    ap.add_argument("outfile", help="path to write filtered .jsonl")
    args = ap.parse_args()
    stream_filter(args.infile, args.outfile)