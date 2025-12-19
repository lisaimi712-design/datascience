import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import List, Dict, Optional
import logging
import xml.etree.ElementTree as ET
from html import unescape
from collections import defaultdict
import random

# Configure module logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import feedparser
    HAVE_FEEDPARSER = True
except Exception:
    feedparser = None
    HAVE_FEEDPARSER = False
    logger.warning("Optional dependency 'feedparser' not installed; fallback parser will be used.")

class AfricanNewsCollector:
    """
    Collects news headlines from GDELT and African RSS feeds
    Enforces: 1 article per month per topic per country (12 per year)
    Selects articles randomly within each month to avoid bias
    """
    
    def __init__(self):
        # African countries list
        self.african_countries = [
            'Algeria', 'Egypt', 'Libya', 'Morocco', 'Sudan', 'Tunisia', 'Western Sahara',
            'Benin', 'Burkina Faso', 'Cape Verde', 'Ivory Coast', 'Gambia', 'Ghana', 'Guinea',
            'Guinea-Bissau', 'Liberia', 'Mali', 'Mauritania', 'Niger', 'Nigeria', 'Senegal',
            'Sierra Leone', 'Togo', 'Angola', 'Cameroon', 'Central African Republic', 'Chad',
            'Republic of the Congo', 'Democratic Republic of the Congo', 'Equatorial Guinea',
            'Gabon', 'Sao Tome and Principe', 'Burundi', 'Comoros', 'Djibouti', 'Eritrea',
            'Ethiopia', 'Kenya', 'Madagascar', 'Malawi', 'Mauritius', 'Mozambique', 'Reunion',
            'Rwanda', 'Seychelles', 'Somalia', 'South Sudan', 'Tanzania', 'Uganda', 'Zambia',
            'Zimbabwe', 'Botswana', 'Eswatini', 'Lesotho', 'Namibia', 'South Africa',
        ]
        
        # Topics of interest
        self.topics = [
            'infrastructure', 'investment', 'economy', 'health', 'education',
            'energy', 'agriculture', 'technology', 'politics', 'trade', 'oil',
            'mining', 'transportation'
        ]
        
        # African news sources with RSS feeds
        self.african_news_sources = {
            'AllAfrica': 'https://allafrica.com/tools/headlines/rdf/latest/headlines.rdf',
            'African Business': 'https://african.business/feed/',
            'The Africa Report': 'https://www.theafricareport.com/feed/',
            'Premium Times': 'https://www.premiumtimesng.com/feed',
            'The Guardian Nigeria': 'https://guardian.ng/feed/',
            'Punch Nigeria': 'https://punchng.com/feed/',
            'Vanguard Nigeria': 'https://www.vanguardngr.com/feed/',
            'Daily Maverick': 'https://www.dailymaverick.co.za/dmrss/',
            'News24': 'https://feeds.news24.com/articles/news24/topstories/rss',
            'Mail & Guardian': 'https://mg.co.za/feed/',
            'The Star Kenya': 'https://www.the-star.co.ke/feed',
            'Daily Nation': 'https://nation.africa/kenya/rss',
            'The Standard Kenya': 'https://www.standardmedia.co.ke/rss/headlines.php',
            'Ghana Web': 'https://www.ghanaweb.com/GhanaHomePage/rss/news.xml',
            'Graphic Online': 'https://www.graphic.com.gh/feeds/news.rss',
            'Addis Standard': 'https://addisstandard.com/feed/',
            'The East African': 'https://www.theeastafrican.co.ke/tea/rss',
            'Egypt Independent': 'https://www.egyptindependent.com/feed/',
            'Al-Ahram': 'https://english.ahram.org.eg/RSS/3.aspx',
            'The Citizen Tanzania': 'https://www.thecitizen.co.tz/tanzania/rss',
            'NewsDay Zimbabwe': 'https://www.newsday.co.zw/feed/',
            'Morocco World News': 'https://www.moroccoworldnews.com/feed/',
        }

    def collect_gdelt_data(self, 
                          countries: Optional[List[str]] = None,
                          topics: Optional[List[str]] = None,
                          days_back: int = 3650,
                          articles_per_month: int = 1) -> pd.DataFrame:
        """
        Collect from GDELT: 1 random article per month per topic per country
        
        Strategy:
        1. For each country-topic combination
        2. For each month in the date range
        3. Fetch ALL articles for that month
        4. Randomly select 1 article from that month
        
        Args:
            countries: List of countries to collect
            topics: List of topics to search
            days_back: How many days back to search (default: 3650 = ~10 years)
            articles_per_month: Articles per month (default: 1)
        
        Returns:
            DataFrame with collected articles
        """
        if countries is None:
            countries = self.african_countries
        if topics is None:
            topics = self.topics
            
        all_articles = []
        base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"Collecting GDELT data for {len(countries)} countries and {len(topics)} topics")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Strategy: {articles_per_month} random article per month per country-topic")
        
        # Generate list of all months in range
        months = []
        current = start_date.replace(day=1)
        while current <= end_date:
            months.append(current)
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        logger.info(f"Total months to process: {len(months)}")
        
        for country in countries:
            logger.info(f"\nProcessing country: {country}")
            
            for topic in topics:
                logger.info(f"  Topic: {topic}")
                query = f"{country} {topic}"
                
                # Process each month
                for month_start in months:
                    # Calculate last day of month
                    if month_start.month == 12:
                        month_end = month_start.replace(day=31)
                    else:
                        next_month = month_start.replace(month=month_start.month + 1)
                        month_end = next_month - timedelta(days=1)
                    
                    # Don't go beyond end_date
                    if month_end > end_date:
                        month_end = end_date
                    
                    params = {
                        'query': query,
                        'mode': 'artlist',
                        'maxrecords': 250,  # Get many to choose from
                        'format': 'json',
                        'startdatetime': month_start.strftime('%Y%m%d000000'),
                        'enddatetime': month_end.strftime('%Y%m%d235959')
                    }
                    
                    try:
                        response = requests.get(base_url, params=params, timeout=30)
                        
                        if response.status_code == 200:
                            data = response.json()
                            articles = data.get('articles', [])
                            
                            if len(articles) > 0:
                                # RANDOMLY select 1 article from this month
                                selected_articles = random.sample(articles, min(articles_per_month, len(articles)))
                                
                                for article in selected_articles:
                                    pub_raw = article.get('seendate')
                                    pub_dt = pd.to_datetime(pub_raw, errors='coerce')
                                    
                                    all_articles.append({
                                        'source': 'GDELT',
                                        'country': country,
                                        'topic': topic,
                                        'year': month_start.year,
                                        'month': month_start.month,
                                        'title': article.get('title'),
                                        'url': article.get('url'),
                                        'published_at': pub_raw,
                                        'domain': article.get('domain'),
                                        'language': article.get('language'),
                                        'tone': article.get('tone')
                                    })
                                
                                logger.debug(f"    {month_start.strftime('%Y-%m')}: Selected {len(selected_articles)} article(s)")
                            else:
                                logger.debug(f"    {month_start.strftime('%Y-%m')}: No articles found")
                        
                        # Rate limiting - be respectful
                        time.sleep(1.5)
                        
                    except Exception as e:
                        logger.error(f"    Error for {country}-{topic}-{month_start.strftime('%Y-%m')}: {str(e)}")
                        time.sleep(2)
        
        logger.info(f"\nCollected {len(all_articles)} total articles from GDELT")
        return pd.DataFrame(all_articles)

    def _parse_rss_with_requests(self, rss_url: str, max_articles: int = 100) -> List[Dict]:
        entries = []
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        for attempt in range(3):
            try:
                resp = requests.get(rss_url, timeout=10, headers=headers)
                resp.raise_for_status()
                content = resp.content
                break
            except Exception:
                time.sleep(1)
                content = None
        
        if content is None:
            return entries
        
        try:
            root = ET.fromstring(content)
        except Exception:
            try:
                text = content.decode('utf-8', errors='replace')
                root = ET.fromstring(unescape(text))
            except Exception:
                return entries
        
        items = root.findall('.//item')
        if not items:
            items = root.findall('.//{http://www.w3.org/2005/Atom}entry')
        
        for it in items[:max_articles]:
            title = ''
            link = ''
            desc = ''
            pub = None
            
            for child in it:
                tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if tag == 'title':
                    title = ''.join(child.itertext()).strip()
                elif tag == 'link':
                    link = child.attrib.get('href', ''.join(child.itertext()).strip())
                elif tag in ['description', 'summary']:
                    desc = ''.join(child.itertext()).strip()
                elif tag in ['pubDate', 'published']:
                    pub = ''.join(child.itertext()).strip()
            
            entries.append({
                'title': unescape(title),
                'link': link,
                'description': unescape(desc),
                'published_at': pub
            })
        
        return entries

    def collect_african_rss_feeds(self,
                                  sources: Optional[List[str]] = None,
                                  max_articles_per_source: int = 100,
                                  days_back: Optional[int] = None) -> pd.DataFrame:
        """
        Collect from RSS feeds (these are recent articles, not historical)
        """
        all_articles = []
        use_feedparser = HAVE_FEEDPARSER and feedparser is not None
        
        if sources is None:
            sources_to_fetch = self.african_news_sources
        else:
            sources_to_fetch = {k: v for k, v in self.african_news_sources.items() if k in sources}
        
        logger.info(f"Collecting from {len(sources_to_fetch)} African news sources...")
        
        cutoff = None
        if days_back:
            cutoff = datetime.now() - timedelta(days=days_back)
        
        source_country_map = {
            'Premium Times': 'Nigeria', 'The Guardian Nigeria': 'Nigeria',
            'Punch Nigeria': 'Nigeria', 'Vanguard Nigeria': 'Nigeria',
            'Daily Maverick': 'South Africa', 'News24': 'South Africa',
            'Mail & Guardian': 'South Africa', 'The Star Kenya': 'Kenya',
            'Daily Nation': 'Kenya', 'The Standard Kenya': 'Kenya',
            'Ghana Web': 'Ghana', 'Graphic Online': 'Ghana',
            'Addis Standard': 'Ethiopia', 'The East African': 'Kenya',
            'Egypt Independent': 'Egypt', 'Al-Ahram': 'Egypt',
            'The Citizen Tanzania': 'Tanzania', 'NewsDay Zimbabwe': 'Zimbabwe',
            'Morocco World News': 'Morocco',
        }
        
        for source_name, rss_url in sources_to_fetch.items():
            try:
                logger.info(f"  Fetching: {source_name}")
                
                if use_feedparser:
                    feed = feedparser.parse(rss_url)
                    for entry in feed.entries[:max_articles_per_source]:
                        pub_date = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            try:
                                pub_date = datetime(*entry.published_parsed[:6])
                            except:
                                pass
                        
                        if cutoff and pub_date and pub_date < cutoff:
                            continue
                        
                        year = pub_date.year if pub_date else None
                        month = pub_date.month if pub_date else None
                        
                        all_articles.append({
                            'source': 'African RSS',
                            'source_name': source_name,
                            'country': source_country_map.get(source_name),
                            'topic': None,
                            'year': year,
                            'month': month,
                            'title': entry.get('title', ''),
                            'description': getattr(entry, 'summary', ''),
                            'url': entry.get('link', ''),
                            'published_at': pub_date,
                            'language': 'unknown'
                        })
                else:
                    parsed = self._parse_rss_with_requests(rss_url, max_articles_per_source)
                    for item in parsed:
                        pub_str = item.get('published_at')
                        pub_date = pd.to_datetime(pub_str, errors='coerce')
                        year = pub_date.year if pd.notna(pub_date) else None
                        month = pub_date.month if pd.notna(pub_date) else None
                        
                        all_articles.append({
                            'source': 'African RSS',
                            'source_name': source_name,
                            'country': source_country_map.get(source_name),
                            'topic': None,
                            'year': year,
                            'month': month,
                            'title': item.get('title', ''),
                            'description': item.get('description', ''),
                            'url': item.get('link', ''),
                            'published_at': pub_str,
                            'language': 'unknown'
                        })
                
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"    Error fetching {source_name}: {str(e)}")
        
        return pd.DataFrame(all_articles)

    def collect_all_sources(self,
                           countries: Optional[List[str]] = None,
                           topics: Optional[List[str]] = None,
                           days_back: int = 3650,
                           include_rss: bool = True,
                           output_folder: Optional[str] = None,
                           articles_per_month: int = 1,
                           max_articles_per_rss_source: Optional[int] = 100) -> pd.DataFrame:
        """
        Collect from all sources with monthly sampling strategy
        
        Args:
            countries: List of countries
            topics: List of topics
            days_back: Days of historical data (default: 3650 = ~10 years)
            include_rss: Whether to include RSS feeds
            output_folder: Where to save data
            articles_per_month: Articles per month per country-topic (default: 1 = 12/year)
            max_articles_per_rss_source: Max articles per RSS source
        
        Returns:
            Combined DataFrame
        """
        print("=" * 60)
        print("Starting comprehensive data collection...")
        print(f"STRATEGY: {articles_per_month} random article per MONTH per country-topic")
        print(f"Expected: {articles_per_month * 12} articles per YEAR per country-topic")
        print("=" * 60)
        
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            logger.info(f"Output folder: {output_folder}")
        
        # Collect from GDELT with monthly sampling
        print("\n1. Collecting from GDELT...")
        gdelt_df = self.collect_gdelt_data(
            countries=countries,
            topics=topics,
            days_back=days_back,
            articles_per_month=articles_per_month
        )
        print(f"   Collected {len(gdelt_df)} articles from GDELT")
        
        if output_folder:
            gdelt_path = os.path.join(output_folder, 'gdelt_data.csv')
            gdelt_df.to_csv(gdelt_path, index=False, encoding='utf-8')
            logger.info(f"Saved GDELT data to {gdelt_path}")
        
        # Collect from RSS feeds
        if include_rss:
            print("\n2. Collecting from African RSS feeds...")
            rss_df = self.collect_african_rss_feeds(
                days_back=days_back if days_back <= 365 else 365,  # RSS only has recent
                max_articles_per_source=max_articles_per_rss_source
            )
            print(f"   Collected {len(rss_df)} articles from RSS feeds")
            
            if output_folder:
                rss_path = os.path.join(output_folder, 'rss_data.csv')
                rss_df.to_csv(rss_path, index=False, encoding='utf-8')
                logger.info(f"Saved RSS data to {rss_path}")
        else:
            rss_df = pd.DataFrame()
        
        # Combine data
        combined_df = pd.concat([gdelt_df, rss_df], ignore_index=True, sort=False)
        
        # Remove duplicates
        if 'title' in combined_df.columns:
            before = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['title'], keep='first')
            after = len(combined_df)
            logger.info(f"Removed {before - after} duplicate titles")
        
        # Add collection timestamp
        combined_df['collected_at'] = datetime.now()
        
        # Convert published_at to datetime
        if 'published_at' in combined_df.columns:
            combined_df['published_at'] = pd.to_datetime(combined_df['published_at'], errors='coerce')
        
        print("\n" + "=" * 60)
        print(f"Total unique articles collected: {len(combined_df)}")
        print("=" * 60)
        
        # Save combined data
        if output_folder:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_path = os.path.join(output_folder, f'combined_data_{timestamp}.csv')
            combined_df.to_csv(final_path, index=False, encoding='utf-8')
            print(f"\nCombined data saved to: {final_path}")
        
        return combined_df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess collected data"""
        df = df[df['title'].notna()].copy()
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        df = df.drop_duplicates(subset=['title'])
        df['title_length'] = df['title'].str.len()
        logger.info(f"Preprocessed data: {len(df)} articles remaining")
        return df
    
    def generate_collection_report(self, df: pd.DataFrame, output_folder: str = None):
        """Generate detailed collection statistics"""
        print("\n" + "=" * 60)
        print("COLLECTION REPORT")
        print("=" * 60)
        
        # Overall stats
        print(f"\nTotal articles: {len(df)}")
        
        # By source
        if 'source' in df.columns:
            print(f"\nBy source:")
            print(df['source'].value_counts())
        
        # GDELT-specific analysis
        gdelt_df = df[df['source'] == 'GDELT']
        if len(gdelt_df) > 0:
            print(f"\n--- GDELT DETAILED ANALYSIS ---")
            print(f"Total GDELT articles: {len(gdelt_df)}")
            
            # Articles per country-topic-year
            if all(col in gdelt_df.columns for col in ['country', 'topic', 'year']):
                year_summary = gdelt_df.groupby(['country', 'topic', 'year']).size()
                print(f"\nArticles per country-topic-YEAR:")
                print(f"  Mean: {year_summary.mean():.2f}")
                print(f"  Median: {year_summary.median():.2f}")
                print(f"  Max: {year_summary.max()}")
                print(f"  Min: {year_summary.min()}")
                print(f"\nDistribution of articles per year:")
                print(year_summary.value_counts().sort_index())
                
                # Check expected target (12 per year)
                target_per_year = 12
                at_target = (year_summary == target_per_year).sum()
                total_combos = len(year_summary)
                print(f"\nCombinations with exactly {target_per_year} articles: {at_target}/{total_combos} ({at_target/total_combos*100:.1f}%)")
            
            # Articles per country-topic-month
            if all(col in gdelt_df.columns for col in ['country', 'topic', 'year', 'month']):
                month_summary = gdelt_df.groupby(['country', 'topic', 'year', 'month']).size()
                print(f"\nArticles per country-topic-MONTH:")
                print(f"  Mean: {month_summary.mean():.2f}")
                print(f"  Max: {month_summary.max()}")
                
                over_one = month_summary[month_summary > 1]
                if len(over_one) > 0:
                    print(f"\n⚠️  WARNING: {len(over_one)} country-topic-months have >1 article")
                    print("First 5 cases:")
                    print(over_one.head())
                else:
                    print(f"\n✓ All country-topic-months have exactly 1 article (as expected)")
            
            # Articles per country
            if 'country' in gdelt_df.columns:
                print(f"\nTop 10 countries by article count:")
                country_counts = gdelt_df['country'].value_counts().head(10)
                print(country_counts)
                
                # Calculate expected
                if 'year' in gdelt_df.columns:
                    years = gdelt_df['year'].nunique()
                    expected_per_country = 13 * 12 * years  # 13 topics × 12 articles/year × N years
                    print(f"\nExpected per country: ~{expected_per_country} (13 topics × 12/year × {years} years)")
        
        # Save report
        if output_folder:
            report_path = os.path.join(output_folder, 'collection_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("AFRICAN NEWS COLLECTION REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Total articles: {len(df)}\n\n")
                if 'source' in df.columns:
                    f.write("By source:\n")
                    f.write(str(df['source'].value_counts()) + "\n\n")
                if len(gdelt_df) > 0 and 'country' in gdelt_df.columns:
                    f.write("By country:\n")
                    f.write(str(gdelt_df['country'].value_counts()) + "\n")
            print(f"\n📄 Report saved to: {report_path}")


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("AFRICAN NEWS COLLECTOR")
    print("Strategy: 1 random article per MONTH per country-topic")
    print("Result: 12 articles per YEAR per country-topic")
    print("="*60 + "\n")
    
    collector = AfricanNewsCollector()
    
    selected_countries = [
       'Algeria', 'Egypt', 'Libya', 'Morocco', 'Sudan', 'Tunisia', 'Western Sahara',
            'Benin', 'Burkina Faso', 'Cape Verde', 'Ivory Coast', 'Gambia', 'Ghana', 'Guinea',
            'Guinea-Bissau', 'Liberia', 'Mali', 'Mauritania', 'Niger', 'Nigeria', 'Senegal',
            'Sierra Leone', 'Togo', 'Angola', 'Cameroon', 'Central African Republic', 'Chad',
            'Republic of the Congo', 'Democratic Republic of the Congo', 'Equatorial Guinea',
            'Gabon', 'Sao Tome and Principe', 'Burundi', 'Comoros', 'Djibouti', 'Eritrea',
            'Ethiopia', 'Kenya', 'Madagascar', 'Malawi', 'Mauritius', 'Mozambique', 'Reunion',
            'Rwanda', 'Seychelles', 'Somalia', 'South Sudan', 'Tanzania', 'Uganda', 'Zambia',
            'Zimbabwe', 'Botswana', 'Eswatini', 'Lesotho', 'Namibia', 'South Africa',
        ]
    
    selected_topics = [
        'infrastructure', 'investment', 'economy', 'health', 'education',
        'energy', 'agriculture', 'technology', 'politics', 'trade', 'oil',
        'mining', 'transportation'
    ]
    
    print(f"Countries: {len(selected_countries)}")
    print(f"Topics: {len(selected_topics)}")
    print(f"Expected per country: 13 topics × 12 articles/year × N years")
    print()
    
    # Collect data
    data = collector.collect_all_sources(
        countries=selected_countries,
        topics=selected_topics,
        days_back=3650,  # ~10 years
        include_rss=True,
        output_folder='african_news_output',
        articles_per_month=1,  # 1 article per month = 12 per year
        max_articles_per_rss_source=10
    )
    
    # Preprocess
    clean_data = collector.preprocess_data(data)
    
    # Save clean data
    clean_path = os.path.join('african_news_output', 'african_news_clean.csv')
    clean_data.to_csv(clean_path, index=False, encoding='utf-8')
    
    # Generate detailed report
    collector.generate_collection_report(clean_data, output_folder='african_news_output')
    
    print(f"\n📁 All files saved in: african_news_output/")
    print("="*60)