"""
*
* Program: comments_filter.py
*
* Modified Date: November 2025
*
* Purpose: Step 1a of Reddit ETL
*    Filter and clean monthly Reddit comments,
*    keep only needed fields for later joining with submissions.
*
"""

from pyspark import SparkConf, SparkContext
import sys
import json

assert sys.version_info >= (3, 5)  # Python 3.5+

TARGET_SUBREDDITS = set(["canada", "canadanews", "canadapolitics", "canadianpolitics",
                         "lpc", "cpc", "ndp", "canadianconservative"])

KEEP_FIELDS = ["id", "link_id", "parent_id", "subreddit", "author",
               "created_utc", "score", "body", "permalink"]

def parse_one(line):
    try:
        obj = json.loads(line)

        # check the target fields
        for k in KEEP_FIELDS:
            if k not in obj:
                return None
        
        # filter subreddit
        sub = str(obj["subreddit"]).strip()
        if not sub:
            return None
        sub_lower = sub.lower()
        if sub_lower not in TARGET_SUBREDDITS:
            return None
        
        # clean comments
        body = str(obj["body"]).strip()
        if(not body) or (body in ("[deleted]", "[removed]")):
            return None
        
        try:
            created_utc = int(float(obj["created_utc"]))
        except Exception:
            return None
        
        try:
            score = int(obj["score"])
        except Exception:
            score = 0

        link_id_raw = str(obj["link_id"]).strip()
        link_id_short = link_id_raw
        if link_id_raw.startswith("t3_"):
            link_id_short = link_id_raw[3:]
        
        cleaned_data = {
            "id": str(obj["id"]),
            "link_id_short": link_id_short,
            "parent_id": str(obj["parent_id"]),
            "subreddit": sub_lower,
            "author": str(obj["author"]).strip(),
            "created_utc": created_utc,
            "score": score,
            "body": body,
            "permalink": str(obj["permalink"]).strip()
        }
        
        return cleaned_data
    
    except Exception:
        return None
        
def output_format(obj):
    # dict -> JSON line
    return json.dumps(obj, ensure_ascii=False)

def main(inputs, output):
    text = sc.textFile(inputs, minPartitions=8)
    
    cleaned = (text
               .map(parse_one)
               .filter(lambda x: x is not None)
               ).cache()

    total = text.count()
    kept = cleaned.count()
    print(f"[INFO] total lines: {total}, kept after filter: {kept}")

    cleaned.map(output_format).saveAsTextFile(output)

    print(f"[OK] wrote cleaned comments to: {output}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: spark-submit comments_filter.py <inputs> <output>")
        sys.exit(1)

    conf = SparkConf().setAppName('comments_filter')
    sc = SparkContext(conf=conf)
    sc.setLogLevel('WARN')
    assert sc.version >= '3.0'  # Spark 3+

    inputs = sys.argv[1]   # e.g., data/comments/RC_2025-01.json
    output = sys.argv[2]   # e.g., out/comments_clean/2025-01
    main(inputs, output)