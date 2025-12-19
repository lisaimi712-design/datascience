import os
import csv
import time
import argparse
from urllib.parse import urlparse
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; LinkChecker/1.0; +https://example.org)'
}

def check_url(session, url, timeout=10):
    result = {
        'url': url,
        'status': None,
        'ok': False,
        'final_url': None,
        'error': None,
        'elapsed': None
    }
    if not isinstance(url, str) or not url.strip():
        result['error'] = 'empty_url'
        return result

    try:
        start = time.time()
        # Try HEAD first
        resp = session.head(url, allow_redirects=True, timeout=timeout, headers=HEADERS)
        result['status'] = resp.status_code
        result['final_url'] = resp.url
        result['elapsed'] = time.time() - start
        if resp.status_code < 400:
            result['ok'] = True
            return result

        # Some servers reject HEAD; try GET
        start = time.time()
        resp = session.get(url, allow_redirects=True, timeout=timeout, headers=HEADERS)
        result['status'] = resp.status_code
        result['final_url'] = resp.url
        result['elapsed'] = time.time() - start
        result['ok'] = resp.status_code < 400
        return result

    except requests.exceptions.Timeout:
        result['error'] = 'timeout'
        return result
    except requests.exceptions.ConnectionError:
        result['error'] = 'connection_error'
        return result
    except requests.exceptions.RequestException as e:
        result['error'] = str(e)
        return result

def main():
    parser = argparse.ArgumentParser(description='Check reachability of article URLs')
    parser.add_argument('--input', '-i', default=None, help='Input CSV path')
    parser.add_argument('--out', '-o', default=os.path.join('data', 'link_check_results.csv'), help='Output CSV path')
    parser.add_argument('--max', '-n', type=int, default=None, help='Max number of URLs to check')
    parser.add_argument('--workers', '-w', type=int, default=None, help='Number of threads to use (default: auto)')
    parser.add_argument('--skip-check', action='store_true', help='Skip network checks and use existing results CSV')
    args = parser.parse_args()

    # Find input
    candidates = [args.input, os.path.join('data', 'final_africa_with_languages.csv'), 'african_news_with_languages.csv', os.path.join('data', 'final_africa.csv')]
    path = None
    for p in candidates:
        if p and os.path.exists(p):
            path = p
            break
    if path is None:
        raise SystemExit('No input CSV found (looked for final_africa_with_languages.csv or african_news_with_languages.csv)')

    print(f"Using input: {path}", flush=True)
    df = pd.read_csv(path)
    if 'url' not in df.columns:
        raise SystemExit('Input CSV does not contain `url` column')

    urls = df['url'].fillna('').astype(str).unique().tolist()
    if args.max:
        urls = urls[:args.max]
    total_urls = len(urls)
    print(f"Found {total_urls} unique URLs to check", flush=True)

    out_dir = os.path.dirname(args.out) or 'data_world'
    os.makedirs(out_dir, exist_ok=True)
    keys = ['url', 'status', 'ok', 'final_url', 'error', 'elapsed']

    # If requested, skip actual network checks and use existing results
    if args.skip_check:
        print('Skipping network checks; will use existing results file if present', flush=True)
    else:
        # Prepare session
        session = requests.Session()

        # Threading parameters
        max_workers = args.workers or min(16, (os.cpu_count() or 4) * 2)
        print(f"Using {max_workers} worker threads", flush=True)

        # Write header for incremental output
        with open(args.out, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()

        # Submit tasks and write results as they come
        checked = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(check_url, session, u): u for u in urls}
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    res = {'url': futures.get(fut, ''), 'status': None, 'ok': False, 'final_url': None, 'error': str(e), 'elapsed': None}
                # Ensure 'ok' is boolean string
                res['ok'] = str(res.get('ok', False))
                # Append to CSV
                with open(args.out, 'a', newline='', encoding='utf-8') as fh:
                    writer = csv.DictWriter(fh, fieldnames=keys)
                    writer.writerow(res)
                checked += 1
                if checked % 100 == 0 or checked == total_urls:
                    print(f"  Checked {checked}/{total_urls}", flush=True)

    # Load results (either newly written or pre-existing) and normalize
    if not os.path.exists(args.out):
        raise SystemExit(f'Results file not found: {args.out}')
    dfres = pd.read_csv(args.out)
    # Normalize 'ok' column to boolean series
    if 'ok' in dfres.columns:
        ok_series = dfres['ok'].astype(str).str.lower().map(lambda s: s in ['true', '1', 'yes'])
    else:
        ok_series = pd.to_numeric(dfres.get('status'), errors='coerce').apply(lambda x: True if (not pd.isna(x) and 200 <= int(x) < 400) else False)

    total = len(dfres)
    ok = int(ok_series.sum())
    broken = total - ok
    print(f"Checked {total} URLs — OK: {ok}, Broken: {broken}", flush=True)
    bad = dfres[~ok_series]
    status_counts = bad['status'].fillna('error').value_counts().head(20)
    print('\nTop failure types:', flush=True)
    print(status_counts.to_string(), flush=True)
    print(f"Results file used: {args.out}", flush=True)

    # Filter the original dataset to only rows whose URL is OK
    original_df = pd.read_csv(path)
    good_urls = set(dfres.loc[ok_series, 'url'].dropna().astype(str).tolist())
    filtered_df = original_df[original_df['url'].astype(str).isin(good_urls)].copy()
    
    # Save to same directory as input file
    input_dir = os.path.dirname(path) or '.'
    out_filtered = os.path.join(input_dir, 'gdelt_valid.csv')
    os.makedirs(input_dir, exist_ok=True)
    filtered_df.to_csv(out_filtered, index=False)
    print(f'Wrote filtered dataset with {len(filtered_df)} rows to: {out_filtered}', flush=True)

if __name__ == '__main__':
    main()
