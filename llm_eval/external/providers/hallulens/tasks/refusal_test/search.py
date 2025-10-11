# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict
import requests
from ratelimit import limits, sleep_and_retry
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
from retry import retry
from dotenv import load_dotenv

load_dotenv()

def batch_search(queries: List[str], max_workers=60, cache=None) -> List[Dict]:
    search_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(search, query, cache) for query in queries]
        progress = tqdm(total=len(queries), desc="Searching", unit="request")
        for future in futures:
            search_results.append(future.result())
            progress.update(1)  # Update the progress bar by one step
        progress.close()  # Close the progress bar
    return search_results


def search(query_string: str, cache=None) -> Dict:
    if cache:
        cached_result = cache.get_item(query_string)
        if cached_result:
            return cached_result
    result = search_brave(query_string)
    if cache:
        cache.set_item(query_string, result)
    return result


class SearchException(Exception):
    pass


@retry(SearchException, tries=5, delay=0.5, backoff=2, jitter=0.1)
@sleep_and_retry
@limits(calls=60, period=1)
def search_brave(query_string: str) -> Dict:
    # time.sleep(0.5)     # This delay is to avoid request speed limit and request limit of brave search api free plan
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": os.environ["BRAVE_API_KEY"],
    }
    resp = requests.get(
        f"https://api.search.brave.com/res/v1/web/search?result_filter=web&q={query_string}",
        headers=headers,
    )
    resp_json = resp.json()
    if resp_json['type'] == 'ErrorResponse':
        if resp_json['error']['code'] == 'RATE_LIMITED':
            raise SearchException('RATE_LIMITED')
        else:
            print(resp_json)
            return {"query": query_string, "search_result": []} 
    return clean_brave_response(resp.json(), top_k=10)


def clean_brave_response(search_response, top_k=3):
    if 'query' not in search_response:
        print(search_response)
    query = None
    clean_response = []
    if "mixed" in search_response:
        mixed_results = search_response["mixed"]
        for m in mixed_results["main"][:top_k]:
            r_type = m["type"]
            results = search_response[r_type]["results"]
            if r_type == "web":
                # For web data - add a single output from the search
                idx = m["index"]
                result = results[idx]
                snippet = ""
                if "description" in result:
                    snippet = result["description"]
                if "extra_snippets" in result:
                    snippet += "\n\n" + "\n".join(result["extra_snippets"])
                cleaned = [
                    {
                        "title": result["title"],
                        "link": result["url"],
                        "snippet": snippet,
                    }
                ]
            elif r_type == "faq":
                # For faw data - take a list of all the questions & answers
                cleaned = []
                for q in result:
                    question = q["question"]
                    answer = q["answer"]
                    snippet = f"Question: {question}\nAnswer: {answer}"
                    cleaned.append(
                        {"title": q["title"], "link": q["url"], "snippet": snippet}
                    )
            elif r_type == "infobox":
                idx = m["index"]
                result = results[idx]
                snippet = (
                    f"{result.get('description', '')} {result.get('long_desc', '')}"
                )
                cleaned = [{k: v for k, v in result.items() if k in selected_keys}]
            elif r_type == "videos":
                cleaned = []
                for q in results:
                    cleaned.append(
                        {
                            "title": q["title"],
                            "link": q["url"],
                            "snippet": q["description"],
                        }
                    )
            elif r_type == "locations":
                # For faw data - take a list of all the questions & answers
                cleaned = []
                for q in results:
                    cleaned.append(
                        {
                            "title": q["title"],
                            "link": q["url"],
                            "snippet": q["description"],
                        }
                    )
            elif r_type == "news":
                # For faw data - take a list of all the questions & answers
                selected_keys = [
                    "type",
                    "title",
                    "url",
                    "description",
                ]
                cleaned = []
                for q in results:
                    cleaned.append(
                        {
                            "title": q["title"],
                            "link": q["url"],
                            "snippet": q["description"],
                        }
                    )
            clean_response.extend(cleaned)
    return {
        "query": search_response["query"]["original"],
        "search_result": clean_response,
    }
