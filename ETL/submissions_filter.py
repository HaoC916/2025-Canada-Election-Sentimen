"""
* Program: submissions_filter.py
*
* Modified Date: November 2025
*
* Purpose: Step 2b of Reddit ETL
*    Filter and clean monthly Reddit submissions,
*    keep only needed fields for joining with comments.
"""

from pyspark import SparkConf, SparkContext
import sys
import json

assert sys.version_info >= (3, 5)  # Python 3.5+

TARGET_SUBREDDITS = set(["canada", "canadanews", "canadapolitics", "canadianpolitics",
                         "lpc", "cpc", "ndp", "canadianconservative"])

KEEP_FIELDS = ["id", "subreddit", "title"]

def parse_one(line):
    try:
        obj = json.loads(line)
        
        for k in KEEP_FIELDS:
            if k not in obj:
                return None
    
        sub = str(obj["subreddit"]).strip()
        if not sub:
            return None
        sub_lower = sub.lower()
        if sub_lower not in TARGET_SUBREDDITS:
            return None

        cleaned_data = {
            "id": str(obj["id"]),
            "subreddit": sub_lower,
            "title": str(obj["title"]).strip()
        }
        
        return cleaned_data
    
    except Exception:
        return None
    
def output_format(obj):
    return json.dumps(obj, ensure_ascii=False)

def main(inputs, output):
    text = sc.textFile(inputs, minPartitions=8)
    cleaned = (text
               .map(parse_one)
               .filter(lambda x: x is not None)
              ).cache()

    total = text.count()
    kept = cleaned.count()
    print(f"[INFO] total submissions lines: {total}, kept after filter: {kept}")

    cleaned.map(output_format).saveAsTextFile(output)
    
    print(f"[OK] wrote cleaned submissions to: {output}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: spark-submit submissions_filter.py <inputs> <output>")
        sys.exit(1)

    conf = SparkConf().setAppName("submissions_filter")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    assert sc.version >= "3.0"

    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)