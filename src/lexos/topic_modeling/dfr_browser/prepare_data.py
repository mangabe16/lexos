"""prepare_data.py.

Last updated: April 16, 2025
Last tested: TBD

Prepare dfr-browser data files from MALLET outputs.

The original version of this script by Andrew Goldstone can be found at
https://github.com/agoldst/dfr-browser/blob/master/bin/prepare-data.

This version has been made compatible with Python 3 and given some type
hinting and further commenting.
"""

import gzip
import json
import zipfile as zf
from collections import defaultdict
from typing import Any, Optional


def convert_state(
    state_file: str, tw_file: str = "tw.json", dt_file: str = "dt.json.zip", n: int = 50
) -> None:
    """Use the MALLET sampling state to write both topic words and document topics.

    Args:
        state_file (str): The gzipped state file from mallet train-topics.
        tw_file (str): The topic-words file.
        dt_file (str): The document-topic file.
        n (int): The number of topics.

    Returns:
        None
    """
    with gzip.open(state_file, "rb") as f:
        f.readline()
        alpha = list(map(float, f.readline().decode().strip().split(" ")[2:]))
        beta = f.readline().decode().strip().split(" ")[2]
        print(f"beta value, not saved in a file: {beta}")

        # A dict of topic numbers where each topic number is a dict of {typeindex: weight}
        tw = defaultdict(lambda: defaultdict(int))
        # A dict of typeindex: word
        vocab = dict()
        # A list of dicts where each dict is of type {topicnumber: int}
        dt = []
        # A dict of type {topicnumber: int}
        cur_dt = defaultdict(int)

        last_doc = 0  # Assume we start at doc 0
        K = 0

        # Iterate through the state file
        for line in f:
            # Split the line and ensure int fields are ints
            doc, source, pos, typeindex, word, topic = line.strip().split()
            doc = int(doc)
            typeindex = int(typeindex)
            topic = int(topic)
            # Set the topic number
            if topic > K:
                K = topic
            # If we're at a new document, save the current document and rest cur_dt
            if last_doc != doc:
                dt.append(cur_dt)
                cur_dt = defaultdict(int)
            # Increment the topic number for cur_dt
            cur_dt[topic] += 1
            # Increment the type index for the topic in tw
            tw[topic][typeindex] += 1
            # Add the word and typeindex to the vocab if it is not already there
            if typeindex not in vocab:
                vocab[typeindex] = word.decode()
            # Set last_doc as the current doc id
            last_doc = doc

        # K is max(topic), but we want it to be number of topics:
        K = K + 1

        # Final doc: after end of for loop
        if len(cur_dt) > 0:
            dt.append(cur_dt)

        # Create a list of dicts for each topic where each dict has list of weights and words
        topic_dicts = []
        for _, values in tw.items():
            weights = []
            words = []
            for typeindex, weight in values.items():
                weights.append(weight)
                words.append(vocab[typeindex])
            topic_dicts.append({"weights": weights, "words": words})

    # Create the topic-words file
    transformed_tw = [
        transform_topic_weights(topic_dicts[t]["weights"], vocab, n)
        for t in range(K)
    ]
    write_tw(alpha, transformed_tw, tw_file)

    # Create the document-topic file
    transformed_dt = transform_dt([[d[t] for d in dt] for t in list(range(K))])
    write_dt(transformed_dt, dt_file)


def info_stub(filepath: str, properties: Optional[dict[str, Any]] = None) -> None:
    """Write an info.json stub to a file.

    Args:
        filepath (str): The file to write to.
        properties (Optional[dict[str, Any]]): Additional properties to include in the JSON.
    """
    fields = {"title": "", "meta_info": r"<h2></h2>", "VIS": {"overview_words": 15}}
    fields = {**fields, **properties} if properties else fields
    try:
        with open(filepath, "w") as f:
            json.dump(fields, fp=f, indent=4)
        print(f"Created stub file in {f.name}")
    except BaseException as e:
        print(f"An error occurred: {e}")


def transform_dt(dt: list) -> dict:
    """Transform document-topic matrix.

    Args:
        dt (list): The document-topic matrix.

    Returns:
        dict: The transformed document-topic matrix.
    """
    D = len(dt[0])
    p = [0]
    i = []
    x = []
    p_cur = 0
    for topic_docs in dt:
        for d in list(range(D)):
            if topic_docs[d] != 0:
                i.append(d)
                x.append(topic_docs[d])
                p_cur += 1
        p.append(p_cur)

    return {"i": i, "p": p, "x": x}


def transform_topic_weights(
    weights: list[int], vocab: list[str], n: int
) -> list[dict[str, float]]:
    """Transform topic weights to a JSON-compatible format.

    Args:
        weights (list[int]): The topic weights.
        vocab (list[str]): The vocabulary.
        n (int): The number of topics.

    Returns:
        list[Dict[str, float]]: The transformed topic weights.
    """
    words = list(range(len(weights)))
    words.sort(key=lambda i: -weights[i])
    return {
        "words": [vocab[w] for w in words[:n]],
        "weights": [weights[w] for w in words[:n]],
    }


def write_dt(dtj: dict, output_file: str) -> None:
    """Write a document-topic matrix to a JSON file.

    Args:
        dtj (dict): The document-topic matrix.
        output_file (str): The output file.

    Returns:
        None
    """
    with zf.ZipFile(output_file, "w") as z:
        z.writestr("dt.json", json.dumps(dtj))

    print(f"Wrote sparse doc-topics to {output_file}")


def write_tw(alpha: list, tw: dict, output_file: str) -> None:
    """Write a topic-word matrix to a JSON file.

    Args:
        alpha (list): The alpha values.
        tw (dict): The topic-word matrix.
        output_file (str): The output file.

    Returns:
        None
    """
    twj = {"alpha": alpha, "tw": tw}
    with open(output_file, "w") as f:
        json.dump(twj, f)

    print(f"Wrote topic-words information to {f.name}")
