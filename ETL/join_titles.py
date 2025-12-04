"""
* Program: join_titles.py
*
* Modified Date: November 2025
*
* Purpose: Step 2c of Reddit ETL
*    Join cleaned comments with cleaned submissions by link_id_short/id
*    to add the submission title to each comment;
*    if no title can be added, keep the title filed empty.
*
"""

from pyspark import SparkConf, SparkContext
import sys
import json

assert sys.version_info >= (3, 5)  # Python 3.5+

def parse_comment(line):
    try:
        obj = json.loads(line)
        link_id = obj.get("link_id_short", None)
        if not link_id:
            return None
        return (str(link_id), obj)
    except Exception:
        return None
    
def parse_submission(line):
    try:
        obj = json.loads(line)
        curr_id = obj.get("id", None)
        if not curr_id:
            return None
        return (str(curr_id), obj)
    except Exception:
        return None

def merge_comment_with_submission(record):
    link_id, (comment, sub_opt) = record
    title = ""

    if sub_opt is not None:
        title = str(sub_opt.get("title", "")).strip()
    
    comment["title"] = title

    return comment

def output_format(obj):
    # dict -> JSON line
    return json.dumps(obj, ensure_ascii=False)

def main(comments_input, submissions_input, output):
    # read cleaned comments
    comments_raw = sc.textFile(comments_input, minPartitions=8)
    comments_kv = (comments_raw
                   .map(parse_comment)
                   .filter(lambda x: x is not None)
                   )

    # read cleaned submissions
    subs_raw = sc.textFile(submissions_input, minPartitions=4)
    subs_kv = (subs_raw
               .map(parse_submission)
               .filter(lambda x: x is not None)
               )

    # some basic stats for testing
    num_comments = comments_kv.count()
    num_subs = subs_kv.count()
    print(f"[INFO] cleaned comments count      : {num_comments}")
    print(f"[INFO] cleaned submissions count   : {num_subs}")

    # leftOuterJoin: keep all comments, attach matching submission if any
    joined = comments_kv.leftOuterJoin(subs_kv)

    # sub_opt is None when no match
    matched = joined.filter(lambda kv: kv[1][1] is not None).count()
    print(f"[INFO] comments with matched post  : {matched}")
    print(f"[INFO] comments with NO matched post: {num_comments - matched}")

    # merge submission fields into comment object
    final_comments = joined.map(merge_comment_with_submission)

    # save as JSON lines
    final_comments.map(output_format).saveAsTextFile(output)
    print(f"[OK] wrote joined comments to: {output}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: spark-submit join_titles.py <comments_input> <submissions_input> <output>")
        sys.exit(1)

    conf = SparkConf().setAppName("join_titles")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    assert sc.version >= "3.0"

    comments_input = sys.argv[1]
    submissions_input = sys.argv[2]
    output = sys.argv[3]

    main(comments_input, submissions_input, output)