"""
*
* Program: filter_submissions_zst.py
*
* Modified Date: November 2025
*
* Purpose: Step 1b of Reddit ETL
*    Stream RS_YYYY-MM.zst with Zstandard,
*    keep only target subreddits and non-deleted submissions,
*    write out filtered JSONL for later Spark processing.
*
"""

#!/usr/bin/env python3
# filter_submissions_zst.py
import argparse
import json
import zstandard as zstd

TARGET = {
    "canada", "canadapolitics", "canadianpolitics",
    "lpc", "cpc", "ndp", "canadianconservative", "canadanews"
}

# Used Copilot to help write codes
def stream_filter_submissions(infile, outfile):
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
                        out.write(buffer.decode("utf-8", errors="ignore"))
                        out.write("\n")
                        kept += 1
                except Exception:
                    bad += 1

    print(f"Finished submissions! seen={seen:,} kept={kept:,} bad_json={bad:,}")
    print(f"Output: {outfile}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", help="path to RS_YYYY-MM.zst")
    ap.add_argument("outfile", help="path to write filtered submissions .jsonl")
    args = ap.parse_args()
    stream_filter_submissions(args.infile, args.outfile)
