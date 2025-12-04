import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import List, Dict, Optional
import json
import itertools
import logging
import xml.etree.ElementTree as ET
from html import unescape
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
try:
    import feedparser
    HAVE_FEEDPARSER = True
except Exception:
    feedparser = None
    HAVE_FEEDPARSER = False
    print("Optional dependency 'feedparser' is not installed. RSS collection will be skipped. Install with: pip install feedparser")

# Configure module logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class AfricanNewsCollector:
    """
    Collects news headlines from NewsAPI, MediaCloud, and GDELT
    focused on African countries and topics
    """
    
    def __init__(self, newsapi_key: str, mediacloud_key: Optional[str] = None):
        self.newsapi_key = newsapi_key
        self.mediacloud_key = mediacloud_key
        
        # African countries list
        self.african_countries = [
    'Algeria', 'Egypt', 'Libya', 'Morocco', 'Sudan', 'Tunisia', 'Western Sahara',     
    'Benin', 'Burkina Faso', 'Cape Verde', 'Ivory Coast', 'Gambia', 'Ghana',
    'Guinea', 'Guinea-Bissau', 'Liberia', 'Mali', 'Mauritania', 'Niger',
    'Nigeria', 'Senegal', 'Sierra Leone', 'Togo',
    'Angola', 'Cameroon', 'Central African Republic', 'Chad', 'Republic of the Congo',
    'Democratic Republic of the Congo', 'Equatorial Guinea', 'Gabon', 
    'Sao Tome and Principe',
    'Burundi', 'Comoros', 'Djibouti', 'Eritrea', 'Ethiopia', 'Kenya',
    'Madagascar', 'Malawi', 'Mauritius', 'Mozambique', 'Reunion', 'Rwanda',
    'Seychelles', 'Somalia', 'South Sudan', 'Tanzania', 'Uganda', 'Zambia',
    'Zimbabwe',
    'Botswana', 'Eswatini', 'Lesotho', 'Namibia', 'South Africa',
        ]
        
        # Topics of interest
        self.topics = [
            'infrastructure', 'investment', 'economy', 'health', 'education', 'hunger', 'aids'
            'energy', 'agriculture', 'technology', 'politics', 'trade','oil', 'mining', 'transportation'
        ]
        
         # African news sources with RSS feeds
        self.african_news_sources = {
            # Pan-African
            'AllAfrica': 'https://allafrica.com/tools/headlines/rdf/latest/headlines.rdf',
            'African Business': 'https://african.business/feed/',
            'The Africa Report': 'https://www.theafricareport.com/feed/',
            
            # Nigeria
            'Premium Times': 'https://www.premiumtimesng.com/feed',
            'The Guardian Nigeria': 'https://guardian.ng/feed/',
            'Punch Nigeria': 'https://punchng.com/feed/',
            'Vanguard Nigeria': 'https://www.vanguardngr.com/feed/',
            
            # South Africa
            'Daily Maverick': 'https://www.dailymaverick.co.za/dmrss/',
            'News24': 'https://feeds.news24.com/articles/news24/topstories/rss',
            'Mail & Guardian': 'https://mg.co.za/feed/',
            
            # Kenya
            'The Star Kenya': 'https://www.the-star.co.ke/feed',
            'Daily Nation': 'https://nation.africa/kenya/rss',
            'The Standard Kenya': 'https://www.standardmedia.co.ke/rss/headlines.php',
            
            # Ghana
            'Ghana Web': 'https://www.ghanaweb.com/GhanaHomePage/rss/news.xml',
            'Graphic Online': 'https://www.graphic.com.gh/feeds/news.rss',
            
            # Ethiopia
            'Addis Standard': 'https://addisstandard.com/feed/',
            
            # East Africa
            'The East African': 'https://www.theeastafrican.co.ke/tea/rss',
            
            # Egypt
            'Egypt Independent': 'https://www.egyptindependent.com/feed/',
            'Al-Ahram': 'https://english.ahram.org.eg/RSS/3.aspx',
            
            # Tanzania
            'The Citizen Tanzania': 'https://www.thecitizen.co.tz/tanzania/rss',
            
            # Zimbabwe
            'NewsDay Zimbabwe': 'https://www.newsday.co.zw/feed/',
            
            # Morocco
            'Morocco World News': 'https://www.moroccoworldnews.com/feed/',
        }
    
    # ==================== NEWSAPI COLLECTION ====================
    
    def collect_newsapi_data(self, 
                            countries: Optional[List[str]] = None,
                            topics: Optional[List[str]] = None,
                            days_back: int = 36000,
                            language: str = 'en,af,fr',
                            attempts_per_window: int = 50,
                            window_hours: int = 12,
                            state_path: str = 'newsapi_state.json') -> pd.DataFrame:
        """
        Collect headlines from NewsAPI
        
        Args:
            countries: List of African countries to search for
            topics: List of topics to search for
            days_back: How many days back to collect data
            language: Language code (en, ar, fr for African context)
        
        Returns:
            DataFrame with collected articles
        """
        if countries is None:
            countries = self.african_countries[:5]  # Limit to avoid rate limits
        if topics is None:
            topics = self.topics[:3]
        
        all_articles = []
        base_url = "https://newsapi.org/v2/everything"
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Build list of all country-topic pairs
        all_pairs = [(c, t) for c in countries for t in topics]

        # Load state to know which pairs have already been sampled in the current cycle
        state = {
            'covered_pairs': [],
            'last_run': None
        }
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r', encoding='utf-8') as fh:
                    state = json.load(fh)
            except Exception:
                # if file corrupted, start fresh
                state = {'covered_pairs': [], 'last_run': None}

        covered = set(tuple(x) for x in state.get('covered_pairs', []))

        # Helper to persist state
        def _save_state():
            state['covered_pairs'] = [list(x) for x in covered]
            state['last_run'] = datetime.now().isoformat()
            try:
                with open(state_path, 'w', encoding='utf-8') as fh:
                    json.dump(state, fh)
            except Exception as e:
                print(f"Warning: could not save state to {state_path}: {e}")

        # Determine remaining pairs to target (prioritize ones not covered)
        remaining = [p for p in all_pairs if tuple(p) not in covered]

        # If nothing remaining, reset cycle
        if not remaining:
            covered.clear()
            remaining = list(all_pairs)

        attempts_done = 0
        max_attempts = attempts_per_window

        # Iterate up to max_attempts; prioritize remaining pairs
        pair_iter = iter(remaining)
        while attempts_done < max_attempts:
            try:
                country, topic = next(pair_iter)
            except StopIteration:
                # If we exhausted remaining but still have attempts left, reset covered and refill
                if len(covered) == len(all_pairs):
                    covered.clear()
                remaining = [p for p in all_pairs if tuple(p) not in covered]
                if not remaining:
                    # nothing to do
                    break
                pair_iter = iter(remaining)
                country, topic = next(pair_iter)

            query = f"{country} AND {topic}"
            params = {
                'q': query,
                'apiKey': self.newsapi_key,
                'language': language,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'pageSize': 100
            }

            try:
                logger.info("Fetching NewsAPI (%d/%d): %s - %s", attempts_done+1, max_attempts, country, topic)
                response = requests.get(base_url, params=params)

                if response.status_code == 200:
                    data = response.json()

                    # Record how many articles we had before this pair so we can tell
                    # whether this pair produced any results.
                    before_count = len(all_articles)
                    for article in data.get('articles', []):
                        all_articles.append({
                            'source': 'NewsAPI',
                            'country': country,
                            'topic': topic,
                            'title': article.get('title'),
                            'description': article.get('description'),
                            'url': article.get('url'),
                            'published_at': article.get('publishedAt'),
                            'source_name': article.get('source', {}).get('name'),
                            'language': language
                        })

                    # Mark pair as covered for this cycle ONLY if at least one
                    # article was returned for the pair.
                    after_count = len(all_articles)
                    if after_count > before_count:
                        covered.add((country, topic))
                        logger.info("Marked covered: %s - %s (returned %d articles)", country, topic, after_count - before_count)

                else:
                    logger.warning("Error %s for %s-%s: %s", response.status_code, country, topic, response.text)

                attempts_done += 1

                # Respect NewsAPI rate limiting by spacing calls — aim to keep within 100/day
                time.sleep(1)

            except Exception as e:
                logger.exception("Error fetching %s-%s: %s", country, topic, e)

        # Save state after attempts
        _save_state()
        return pd.DataFrame(all_articles)

    def schedule_newsapi_collection(self,
                                    countries: Optional[List[str]] = None,
                                    topics: Optional[List[str]] = None,
                                    days_back: int = 36000,
                                    language: str = 'en,af,fr',
                                    attempts_per_window: int = 50,
                                    window_hours: int = 12,
                                    state_path: str = 'newsapi_state.json',
                                    save_path: Optional[str] = None,
                                    master_path: Optional[str] = None,
                                    per_country_target: Optional[int] = None):
        """Run NewsAPI collection repeatedly every `window_hours` hours.

        This method loops indefinitely until interrupted (KeyboardInterrupt). Each loop
        invokes `collect_newsapi_data` with the provided parameters and optionally saves
        the result to `save_path`.
        """
        try:
            while True:
                logger.info("Starting NewsAPI collection cycle at %s", datetime.now().isoformat())
                df = self.collect_newsapi_data(countries=countries,
                                               topics=topics,
                                               days_back=days_back,
                                               language=language,
                                               attempts_per_window=attempts_per_window,
                                               window_hours=window_hours,
                                               state_path=state_path)

                if isinstance(df, pd.DataFrame) and save_path:
                    try:
                        # write a timestamped CSV for this cycle
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        base, ext = os.path.splitext(save_path)
                        out_path = f"{base}_{ts}{ext or '.csv'}"
                        df.to_csv(out_path, index=False, encoding='utf-8')
                        logger.info("Saved %d rows to %s", len(df), out_path)
                    except Exception as e:
                        logger.exception("Could not save to %s: %s", save_path, e)

                    # If a master file is requested, merge and dedupe into it
                    if master_path:
                        try:
                            if os.path.exists(master_path):
                                old = pd.read_csv(master_path)
                                merged = pd.concat([old, df], ignore_index=True, sort=False)
                                # dedupe by title
                                merged = merged.drop_duplicates(subset=['title'], keep='first')
                                merged.to_csv(master_path, index=False)
                                logger.info("Appended+deduped %d rows into master %s (now %d rows)", len(df), master_path, len(merged))
                            else:
                                df.to_csv(master_path, index=False)
                                logger.info("Created master file %s with %d rows", master_path, len(df))
                        except Exception:
                            logger.exception("Failed to update master file %s", master_path)

                # After saving, check persisted state to see if target reached
                if per_country_target is not None and os.path.exists(state_path):
                    try:
                        with open(state_path, 'r', encoding='utf-8') as fh:
                            st = json.load(fh)
                        counts = st.get('country_counts', {}) or {}
                        # consider only the configured countries (if provided)
                        target_countries = countries if countries is not None else self.african_countries
                        all_reached = all(int(counts.get(c, 0)) >= per_country_target for c in target_countries)
                        if all_reached:
                            logger.info("Per-country target %d reached for all countries; stopping scheduler.", per_country_target)
                            break
                    except Exception:
                        logger.exception("Could not read state file %s to check targets", state_path)

                # Sleep until next window
                logger.info("Cycle complete. Sleeping for %s hours...", window_hours)
                time.sleep(window_hours * 3600)
        except KeyboardInterrupt:
            logger.info("NewsAPI scheduler interrupted by user; exiting loop.")
        
        
    
    # ==================== MEDIACLOUD COLLECTION ====================
    
    def collect_mediacloud_data(self,
                                query: str,
                                days_back: int = 36000) -> pd.DataFrame:
        if not self.mediacloud_key:
            print("MediaCloud API key not provided. Skipping MediaCloud collection.")
            return pd.DataFrame()
        
        # MediaCloud new API endpoint (as of 2024, verify current endpoint)
        base_url = "https://api.mediacloud.org/api/v2/stories_public/list"
        
        all_stories = []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        params = {
            'q': query,
            'fq': f'publish_date:[{start_date.strftime("%Y-%m-%d")} TO {end_date.strftime("%Y-%m-%d")}]',
            'key': self.mediacloud_key,
            'rows': 100
        }
        
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                for story in data.get('stories', []):
                    all_stories.append({
                        'source': 'MediaCloud',
                        'title': story.get('title'),
                        'url': story.get('url'),
                        'published_at': story.get('publish_date'),
                        'media_name': story.get('media_name'),
                        'language': story.get('language')
                    })
        except Exception as e:
            print(f"MediaCloud error: {str(e)}")
        
        return pd.DataFrame(all_stories)
    
    # ==================== GDELT COLLECTION ====================
    
    def collect_gdelt_data(self,
                          countries: Optional[List[str]] = None,
                          topics: Optional[List[str]] = None,
                          days_back: int = 36000) -> pd.DataFrame:
        """
        Collect from GDELT using their GKG (Global Knowledge Graph) API
        GDELT is free and doesn't require API key
        
        Args:
            countries: List of countries to filter
            topics: Topics to search for
            days_back: Days back (GDELT works best with recent data)
        
        Returns:
            DataFrame with GDELT articles
        """
        # If no countries provided, use the full configured African country list
        # (GDELT queries are broad; caller can limit if desired)
        if countries is None:
            countries = self.african_countries
        if topics is None:
            topics = self.topics[:2]
        
        all_articles = []
        
        # GDELT 2.0 DOC API endpoint
        base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        for country in countries:
            for topic in topics:
                query = f"{country} {topic}"
                
                params = {
                    'query': query,
                    'mode': 'artlist',
                    'maxrecords': 250,
                    'format': 'json',
                    'startdatetime': start_date.strftime('%Y%m%d000000'),
                    'enddatetime': end_date.strftime('%Y%m%d235959')
                }
                
                try:
                    print(f"Fetching GDELT: {country} - {topic}")
                    response = requests.get(base_url, params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for article in data.get('articles', []):
                            all_articles.append({
                                'source': 'GDELT',
                                'country': country,
                                'topic': topic,
                                'title': article.get('title'),
                                'url': article.get('url'),
                                'published_at': article.get('seendate'),
                                'domain': article.get('domain'),
                                'language': article.get('language'),
                                'tone': article.get('tone') 
                            })
                    
                    # Be respectful with GDELT API
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"GDELT error for {country}-{topic}: {str(e)}")
        
        return pd.DataFrame(all_articles)

    def _parse_rss_with_requests(self, rss_url: str, max_articles: int = 1000) -> List[Dict]:
        """Fallback RSS parser using requests + ElementTree.

        Returns list of article dicts with keys: title, link, description, published_at
        """
        entries = []

        # Use a browser-like User-Agent and simple retry/backoff to reduce rejections
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
        }
        attempts = 3
        backoff = 1.0
        content = None
        for attempt in range(attempts):
            try:
                resp = requests.get(rss_url, timeout=10, headers=headers)
                resp.raise_for_status()
                content = resp.content
                break
            except Exception as e:
                logger.debug("Attempt %d failed fetching %s: %s", attempt + 1, rss_url, e)
                time.sleep(backoff)
                backoff *= 2

        if content is None:
            logger.warning("Failed to fetch RSS %s after %d attempts", rss_url, attempts)
            return entries

        # Parse XML; try decode/unescape if raw parse fails
        try:
            root = ET.fromstring(content)
        except Exception as e:
            try:
                text = content.decode('utf-8', errors='replace')
                text = unescape(text)
                root = ET.fromstring(text)
            except Exception:
                logger.warning("Failed to parse XML from %s: %s", rss_url, e)
                return entries

        # Find item elements (RSS) or entry elements (Atom)
        items = root.findall('.//item')
        if not items:
            items = root.findall('.//{http://www.w3.org/2005/Atom}entry') or root.findall('.//entry')

        def _get_text(elem, names):
            if elem is None:
                return None
            for child in list(elem):
                tag = child.tag
                # strip namespace if present
                if '}' in tag:
                    tag = tag.rsplit('}', 1)[1]
                if tag in names:
                    # join text including nested tags
                    # If the link is an element with href attribute (Atom), return that
                    href = child.attrib.get('href') if hasattr(child, 'attrib') else None
                    if href:
                        return href
                    return ''.join(child.itertext()).strip()
            # fallback: check direct text if elem itself matches
            tag = elem.tag
            if '}' in tag:
                tag = tag.rsplit('}', 1)[1]
            if tag in names:
                return ''.join(elem.itertext()).strip()
            return None

        for it in items:
            if len(entries) >= max_articles:
                break

            title = _get_text(it, {'title'}) or ''
            link = _get_text(it, {'link', 'id'}) or ''
            # For Atom, sometimes the link is present as <link href="..."/>
            if not link:
                # check attributes on the element itself (e.g., <link href=...>)
                l = it.attrib.get('href') if hasattr(it, 'attrib') else None
                if l:
                    link = l
            description = _get_text(it, {'description', 'summary', 'content'}) or ''
            pub = _get_text(it, {'pubDate', 'published', 'updated'}) or None

            # Unescape HTML entities
            title = unescape(title)
            description = unescape(description)

            entries.append({
                'title': title,
                'link': link,
                'description': description,
                'published_at': pub
            })

        return entries
    
    def collect_african_rss_feeds(self,
                                  sources: Optional[List[str]] = None,
                                  max_articles_per_source: int = 1000,
                                  days_back: Optional[int] = None) -> pd.DataFrame:
        """
        Collect articles from African news sources via RSS feeds
        This provides LOCAL African coverage without API limits!
        
        Args:
            sources: List of source names to collect from (None = all sources)
            max_articles_per_source: Maximum articles to collect per source
        
        Returns:
            DataFrame with collected articles
        """
        all_articles = []
        use_feedparser = HAVE_FEEDPARSER and feedparser is not None
        if not use_feedparser:
            logger.info("'feedparser' not available — using fallback parser (requests + ElementTree)")
        
        # Select sources
        if sources is None:
            sources_to_fetch = self.african_news_sources
        else:
            sources_to_fetch = {k: v for k, v in self.african_news_sources.items() 
                               if k in sources}
        
        print(f"Collecting from {len(sources_to_fetch)} African news sources...")
        
        # compute cutoff if days_back provided
        cutoff = None
        if days_back is not None:
            cutoff = datetime.now() - timedelta(days=days_back)

        for source_name, rss_url in sources_to_fetch.items():
            try:
                print(f"  Fetching: {source_name}")

                articles_collected = 0

                if use_feedparser:
                    feed = feedparser.parse(rss_url)
                    for entry in feed.entries:
                        if articles_collected >= max_articles_per_source:
                            break

                        # Extract publication date
                        pub_date = None
                        if getattr(entry, 'published_parsed', None):
                            pp = entry.published_parsed
                            try:
                                pub_date = datetime(*pp[:6])
                            except Exception:
                                pub_date = None
                        elif getattr(entry, 'updated_parsed', None):
                            up = entry.updated_parsed
                            try:
                                pub_date = datetime(*up[:6])
                            except Exception:
                                pub_date = None

                        # If entry has string fields, try to coerce
                        if pub_date is None:
                            pub_str = entry.get('published') or entry.get('updated') or None
                            if pub_str:
                                try:
                                    pub_date = pd.to_datetime(pub_str, errors='coerce')
                                    if pd.isna(pub_date):
                                        pub_date = None
                                except Exception:
                                    pub_date = None

                        # Extract description/summary
                        description = None
                        if hasattr(entry, 'summary'):
                            description = entry.summary
                        elif hasattr(entry, 'description'):
                            description = entry.description

                        all_articles.append({
                            'source': 'African RSS',
                            'source_name': source_name,
                            'title': entry.get('title', ''),
                            'description': description,
                            'url': entry.get('link', ''),
                            'published_at': pub_date,
                            'language': 'unknown'
                        })

                        # filter by days_back if requested
                        if cutoff is not None:
                            pub_dt = pd.to_datetime(pub_date, errors='coerce')
                            if pd.isna(pub_dt) or pub_dt < cutoff:
                                # remove the article we just appended
                                all_articles.pop()
                                continue

                        articles_collected += 1

                else:
                    # Use fallback parser that returns simple dicts
                    parsed = self._parse_rss_with_requests(rss_url, max_articles_per_source)
                    for item in parsed:
                        if articles_collected >= max_articles_per_source:
                            break
                        # coerce published_at to datetime if string-like
                        pub_val = item.get('published_at')
                        try:
                            pub_dt = pd.to_datetime(pub_val, errors='coerce')
                        except Exception:
                            pub_dt = None

                        all_articles.append({
                            'source': 'African RSS',
                            'source_name': source_name,
                            'title': item.get('title', ''),
                            'description': item.get('description', ''),
                            'url': item.get('link', ''),
                            'published_at': pub_dt,
                            'language': 'unknown'
                        })

                        # filter by days_back if requested
                        if cutoff is not None:
                            if pd.isna(pub_dt) or pub_dt < cutoff:
                                all_articles.pop()
                                continue
                        articles_collected += 1

                print(f"    Collected {articles_collected} articles")
                time.sleep(0.5)

            except Exception as e:
                print(f"    Error fetching {source_name}: {str(e)}")
        
        return pd.DataFrame(all_articles)
    
    # ==================== COMBINED COLLECTION ====================
    
    def collect_all_sources(self,
                           countries: Optional[List[str]] = None,
                           topics: Optional[List[str]] = None,
                           days_back: int = 36000,
                           include_rss: bool = True,
                           save_path: Optional[str] = None,
                           master_path: Optional[str] = None) -> pd.DataFrame:
        """Collect from all available sources and combine.

        Args:
            countries: Countries to focus on
            topics: Topics to search for
            days_back: Days of historical data
            include_rss: Whether to include African RSS feeds
            save_path: Snapshot CSV path (optional)
            master_path: Master CSV path to append+dedupe into (optional)

        Returns:
            Combined DataFrame from all sources
        """
        print("=" * 60)
        print("Starting comprehensive data collection...")
        print("=" * 60)

        # Collect from NewsAPI
        print("\n1. Collecting from NewsAPI...")
        newsapi_df = self.collect_newsapi_data(countries, topics, days_back)
        print(f"   Collected {len(newsapi_df)} articles from NewsAPI")

        # Collect from MediaCloud
        print("\n2. Collecting from MediaCloud...")
        if countries:
            mediacloud_query = " OR ".join(countries)
            mediacloud_df = self.collect_mediacloud_data(mediacloud_query, days_back)
            print(f"   Collected {len(mediacloud_df)} articles from MediaCloud")
        else:
            mediacloud_df = pd.DataFrame()

        # Collect from GDELT (use full African list when caller didn't provide countries)
        print("\n3. Collecting from GDELT...")
        gdelt_countries = countries if countries is not None else self.african_countries
        gdelt_df = self.collect_gdelt_data(gdelt_countries, topics, days_back)
        print(f"   Collected {len(gdelt_df)} articles from GDELT")

        # Collect from African RSS feeds
        if include_rss and hasattr(self, 'collect_african_rss_feeds'):
            print("\n4. Collecting from African RSS feeds...")
            rss_df = self.collect_african_rss_feeds()
            print(f"   Collected {len(rss_df)} articles from RSS feeds")
        else:
            rss_df = pd.DataFrame()

        # Combine all dataframes
        combined_df = pd.concat([newsapi_df, mediacloud_df, gdelt_df, rss_df],
                                ignore_index=True, sort=False)

        # Remove duplicates based on title
        if 'title' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['title'], keep='first')

        # Add collection timestamp
        combined_df['collected_at'] = datetime.now()

        print("\n" + "=" * 60)
        print(f"Total unique articles collected: {len(combined_df)}")
        print("=" * 60)

        # Save snapshot if path provided
        if save_path:
            combined_df.to_csv(save_path, index=False)
            print(f"\nData saved to: {save_path}")

        # Merge into master file (append + dedupe) if requested
        if master_path is not None:
            try:
                if os.path.exists(master_path):
                    old = pd.read_csv(master_path)
                    merged = pd.concat([old, combined_df], ignore_index=True, sort=False)
                    if 'title' in merged.columns:
                        merged = merged.drop_duplicates(subset=['title'], keep='first')
                    merged.to_csv(master_path, index=False)
                    print(f"Master updated at {master_path} ({len(merged)} rows)")
                else:
                    combined_df.to_csv(master_path, index=False)
                    print(f"Master created at {master_path} ({len(combined_df)} rows)")
            except Exception as e:
                print(f"Could not update master file {master_path}: {e}")

        return combined_df
    
    # ==================== DATA PREPROCESSING ====================
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic preprocessing of collected data
        
        Args:
            df: Raw collected DataFrame
        
        Returns:
            Preprocessed DataFrame
        """
        # Remove rows with missing titles
        df = df[df['title'].notna()].copy()
        
        # Convert published_at to datetime
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['title'])
        
        # Add text length (kept for diagnostics) but do NOT drop short titles
        df['title_length'] = df['title'].str.len()
        
        print(f"Preprocessed data: {len(df)} articles remaining")
        
        return df


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Initialize collector
    NEWSAPI_KEY = "2cf75fc6d6de4e348a75c2d85a52ba57"  
    MEDIACLOUD_KEY = "ed2d3b284b34b9a58b71c7f1916422054d402648"  
    
    collector = AfricanNewsCollector(
        newsapi_key=NEWSAPI_KEY,
        mediacloud_key=MEDIACLOUD_KEY
    )
    
    # Define parameters
    selected_countries = [    'Algeria', 'Egypt', 'Libya', 'Morocco', 'Sudan', 'Tunisia', 'Western Sahara',     
    'Benin', 'Burkina Faso', 'Cape Verde', 'Ivory Coast', 'Gambia', 'Ghana',
    'Guinea', 'Guinea-Bissau', 'Liberia', 'Mali', 'Mauritania', 'Niger',
    'Nigeria', 'Senegal', 'Sierra Leone', 'Togo',
    'Angola', 'Cameroon', 'Central African Republic', 'Chad', 'Republic of the Congo',
    'Democratic Republic of the Congo', 'Equatorial Guinea', 'Gabon', 
    'Sao Tome and Principe',
    'Burundi', 'Comoros', 'Djibouti', 'Eritrea', 'Ethiopia', 'Kenya',
    'Madagascar', 'Malawi', 'Mauritius', 'Mozambique', 'Reunion', 'Rwanda',
    'Seychelles', 'Somalia', 'South Sudan', 'Tanzania', 'Uganda', 'Zambia',
    'Zimbabwe',
    'Botswana', 'Eswatini', 'Lesotho', 'Namibia', 'South Africa',]
    selected_topics = [ 'infrastructure', 'investment', 'economy', 'health', 'education',
            'energy', 'agriculture', 'technology', 'politics', 'trade']
    
    # Collect data from all sources
    data = collector.collect_all_sources(
        countries=selected_countries,
        topics=selected_topics,
        days_back=36000, 
        save_path='african_news_data.csv'
    )
    
    # Preprocess
    clean_data = collector.preprocess_data(data)
    # Ensure 'country' column exists (RSS and some sources may not set it)
    if 'country' not in clean_data.columns:
        clean_data['country'] = None

    # Display summary
    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total articles: {len(clean_data)}")
    print(f"\nArticles by source:")
    if 'source' in clean_data.columns:
        print(clean_data['source'].value_counts())
    else:
        print('No source information available')

    print(f"\nArticles by country:")
    # show counts including NaN (as 'Unknown')
    try:
        print(clean_data['country'].fillna('Unknown').value_counts())
    except Exception:
        print('Could not compute country counts')

    print(f"\nDate range: {clean_data['published_at'].min()} to {clean_data['published_at'].max()}")
    
    # Save preprocessed data
    clean_data.to_csv('african_news_clean.csv', index=False)
    print("\nClean data saved to: african_news_clean.csv")