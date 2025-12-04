# run_rss_only.py
from data import AfricanNewsCollector
import data as data_module
import pandas as pd

# Optionally force the fallback parser (requests+ElementTree)
# Uncomment to test fallback even if feedparser is installed:
# data_module.HAVE_FEEDPARSER = False
# data_module.feedparser = None

collector = AfricanNewsCollector(newsapi_key='', mediacloud_key=None)

# Optionally restrict to a subset of RSS sources by their keys from data.py
# e.g.: sources = ['AllAfrica', 'News24', 'African Business']
sources = None  # None == use all in collector.african_news_sources

# Collect only RSS articles (limit per source for speed)
# optionally set days_back to limit to recent items (None = no limit)
days_back = 7
df_rss = collector.collect_african_rss_feeds(sources=sources, max_articles_per_source=1000, days_back=days_back)

print(f'Collected {len(df_rss)} RSS articles')

# Preprocess (normalizes dates, drops short titles)
df_clean = collector.preprocess_data(df_rss)

print(f'After preprocess: {len(df_clean)} articles')

# ===== Option C: infer country heuristics =====
def infer_country_from_row(row, country_list=None):
    """Heuristic country inference from source_name, url, title, or description.

    Order of checks:
    1. source_name -> known mapping (best-effort)
    2. URL TLD or domain contains ccTLD (e.g. .ng, .za)
    3. title/description contains country name
    Returns country name or 'Unknown'
    """
    if country_list is None:
        # use collector's configured country list if available
        country_list = [c.lower() for c in getattr(collector, 'african_countries', [])]

    # only last 7 days
    df_rss = collector.collect_african_rss_feeds(sources=sources,
                                             max_articles_per_source= 1000,
                                             days_back= 36000)

    # 1) source_name clues
    src = (row.get('source_name') or '')
    if isinstance(src, str):
        for c in country_list:
            if c in src.lower():
                return c.title()

    # 2) url/domain ccTLD
    url = (row.get('url') or '')
    if isinstance(url, str) and url:
        # check domain suffixes
        tld_map = {
            'DZ': 'Algeria', 'AO': 'Angola', 'BJ': 'Benin', 'BW': 'Botswana',
    'BF': 'Burkina Faso', 'BI': 'Burundi', 'CM': 'Cameroon', 'CV': 'Cape Verde',
    'CF': 'Central African Republic', 'TD': 'Chad', 'KM': 'Comoros',
    'CG': 'Republic of the Congo', 'CD': 'Democratic Republic of the Congo',
    'CI': 'Ivory Coast', 'DJ': 'Djibouti', 'EG': 'Egypt', 'GQ': 'Equatorial Guinea',
    'ER': 'Eritrea', 'SZ': 'Eswatini', 'ET': 'Ethiopia', 'GA': 'Gabon',
    'GM': 'Gambia', 'GH': 'Ghana', 'GN': 'Guinea', 'GW': 'Guinea-Bissau',
    'KE': 'Kenya', 'LS': 'Lesotho', 'LR': 'Liberia', 'LY': 'Libya',
    'MG': 'Madagascar', 'MW': 'Malawi', 'ML': 'Mali', 'MR': 'Mauritania',
    'MU': 'Mauritius', 'YT': 'Mayotte', 'MA': 'Morocco', 'MZ': 'Mozambique',
    'NA': 'Namibia', 'NE': 'Niger', 'NG': 'Nigeria', 'RE': 'Reunion',
    'RW': 'Rwanda', 'ST': 'Sao Tome and Principe', 'SN': 'Senegal',
    'SC': 'Seychelles', 'SL': 'Sierra Leone', 'SO': 'Somalia', 'ZA': 'South Africa',
    'SS': 'South Sudan', 'SD': 'Sudan', 'TZ': 'Tanzania', 'TG': 'Togo',
    'TN': 'Tunisia', 'UG': 'Uganda', 'EH': 'Western Sahara', 'ZM': 'Zambia',
    'ZW': 'Zimbabwe'
        }
        for suffix, country in tld_map.items():
            if suffix in url:
                return country

        # 3) url path contains country name
        for c in country_list:
            if f'/{c}/' in url.lower() or f'-{c}-' in url.lower() or c in url.lower().split('.'):
                return c.title()

    # 4) title/description contains country name
    title = (row.get('title') or '')
    desc = (row.get('description') or '')
    text = ' '.join([str(title), str(desc)]).lower()
    for c in country_list:
        if c in text:
            return c.title()

    return 'Unknown'

# Apply inference and add column
if not df_clean.empty:
    df_clean['country_inferred'] = df_clean.apply(infer_country_from_row, axis=1)
    print('\nInferred country counts:')
    print(df_clean['country_inferred'].value_counts().head(20))


# Example: search/filter articles for keywords/topics
keywords = ['investment', 'health', 'education']  # OR-style search
mask = pd.Series(False, index=df_clean.index)
for kw in keywords:
    mask = mask | df_clean['title'].str.contains(kw, case=False, na=False) \
               | df_clean['description'].astype(str).str.contains(kw, case=False, na=False)
results = df_clean[mask].copy()

print(f'Articles matching keywords ({keywords}): {len(results)}')
if not results.empty:
    print(results[['source_name','title','url','published_at']].head().to_string())

# Save outputs (guard against file write permission errors)
try:
    df_rss.to_csv('rss_raw_snapshot.csv', index=False)
except Exception as e:
    print(f"Warning: could not write rss_raw_snapshot.csv: {e}")

try:
    df_clean.to_csv('rss_preprocessed.csv', index=False)
except Exception as e:
    print(f"Warning: could not write rss_preprocessed.csv: {e}")

try:
    results.to_csv('rss_search_results.csv', index=False)
except Exception as e:
    print(f"Warning: could not write rss_search_results.csv: {e}")

print('Finished run — CSV write attempted (see warnings if any)')