"""
Advanced year inference using multiple methods:
1. Improved date parsing (collected_at/published_at)
2. Relative date extraction ("last year", "X months ago")
3. Source-based median year assignment
4. Collection date fallback
"""

import pandas as pd
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def parse_date_flexible(date_str):
    """
    Flexibly parse various date formats using pandas + dateutil
    """
    if pd.isna(date_str) or str(date_str).strip() == '':
        return None
    
    try:
        # Try pandas' smart parsing first
        dt = pd.to_datetime(date_str, infer_datetime_format=True)
        return dt
    except:
        pass
    
    # Try common formats
    formats = [
        '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y', '%b %d, %Y',
        '%d %B %Y', '%d %b %Y', '%Y', '%Y-%m'
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(str(date_str).strip(), fmt)
            return pd.Timestamp(dt)
        except:
            pass
    
    return None

def extract_relative_date_year(text, reference_date):
    """
    Extract relative dates like "last year", "X months ago", "this year"
    Uses reference_date (collected_at) to calculate implied year
    """
    if pd.isna(text) or pd.isna(reference_date):
        return None
    
    text = str(text).lower()
    
    # Ensure reference_date is a datetime
    if isinstance(reference_date, str):
        reference_date = parse_date_flexible(reference_date)
    if not isinstance(reference_date, pd.Timestamp):
        return None
    
    ref_year = reference_date.year
    
    # Pattern: "last year"
    if re.search(r'\blast\s+year\b', text):
        return ref_year - 1
    
    # Pattern: "this year"
    if re.search(r'\bthis\s+year\b', text):
        return ref_year
    
    # Pattern: "next year"
    if re.search(r'\bnext\s+year\b', text):
        return ref_year + 1
    
    # Pattern: "X years ago"
    match = re.search(r'(\d+)\s+years?\s+ago', text)
    if match:
        years_ago = int(match.group(1))
        return ref_year - years_ago
    
    # Pattern: "X months ago"
    match = re.search(r'(\d+)\s+months?\s+ago', text)
    if match:
        months_ago = int(match.group(1))
        years_ago = months_ago // 12
        if years_ago > 0:
            return ref_year - years_ago
    
    # Pattern: "X weeks ago" (rough estimate)
    match = re.search(r'(\d+)\s+weeks?\s+ago', text)
    if match:
        weeks_ago = int(match.group(1))
        if weeks_ago > 26:  # More than 6 months
            years_ago = weeks_ago // 52
            if years_ago > 0:
                return ref_year - years_ago
    
    # Pattern: "in YYYY" (e.g., "in 2023")
    match = re.search(r'\bin\s+(20\d{2}|19\d{2})\b', text)
    if match:
        year = int(match.group(1))
        if 1900 <= year <= 2100:
            return year
    
    return None

def calculate_source_median_year(df):
    """
    Calculate median year for each source
    Returns dict: {source: median_year}
    """
    source_medians = {}
    
    if 'source' not in df.columns or 'year' not in df.columns:
        return source_medians
    
    # Get rows with valid years
    df_valid = df[df['year'].notna() & (df['year'].astype(str).str.strip() != '')].copy()
    df_valid['year'] = pd.to_numeric(df_valid['year'], errors='coerce')
    df_valid = df_valid[df_valid['year'].notna()]
    
    if len(df_valid) == 0:
        return source_medians
    
    # Calculate median per source
    for source in df_valid['source'].unique():
        years = df_valid[df_valid['source'] == source]['year']
        if len(years) > 0:
            median_year = years.median()
            source_medians[source] = int(median_year)
    
    return source_medians

def infer_year_advanced(row, date_col, source_medians=None):
    """
    Advanced year inference with priority:
    1. Keep existing valid year
    2. Parse date column carefully
    3. Extract relative dates from combined text
    4. Use source median
    5. Fallback to collection date
    """
    # 1. Keep existing valid year
    if pd.notna(row.get('year')) and str(row.get('year', '')).strip() != '':
        try:
            year_val = int(row['year'])
            if 2010 <= year_val <= 2030:
                return year_val
        except:
            pass
    
    # 2. Try parsing date column carefully
    if date_col and date_col in row:
        dt = parse_date_flexible(row[date_col])
        if dt and dt.year:
            year = int(dt.year)
            if 1900 <= year <= 2100:
                return year
    
    # 3. Extract relative dates from combined text
    combined_text = None
    for col in ['combined_text', 'processed_text', 'description', 'title']:
        if col in row and pd.notna(row[col]):
            combined_text = row[col]
            break
    
    ref_date = row.get(date_col) if date_col and date_col in row else None
    if combined_text and ref_date:
        year = extract_relative_date_year(combined_text, ref_date)
        if year and 1900 <= year <= 2100:
            return year
    
    # 4. Use source median year
    if source_medians and 'source' in row:
        source = row['source']
        if source in source_medians:
            return source_medians[source]
    
    # 5. Fallback to collection date (collected_at)
    if 'collected_at' in row:
        dt = parse_date_flexible(row['collected_at'])
        if dt and dt.year:
            year = int(dt.year)
            if 1900 <= year <= 2100:
                return year
    
    return None

def process_file_advanced(input_path, output_path, date_col):
    """
    Process a CSV file with advanced year inference
    """
    print(f"\n{'='*80}")
    print(f"Processing: {input_path}")
    print(f"{'='*80}")
    
    df = pd.read_csv(input_path)
    
    print(f"Total rows: {len(df)}")
    
    # Count missing years before
    missing_before = df['year'].isna().sum() + (df['year'].astype(str).str.strip() == '').sum()
    print(f"Missing years before: {missing_before} ({100.0*missing_before/len(df):.1f}%)")
    
    # Calculate source medians
    print("\nCalculating source median years...")
    source_medians = calculate_source_median_year(df)
    print(f"  Found {len(source_medians)} sources with median years")
    if len(source_medians) > 0:
        print(f"  Sample: {list(source_medians.items())[:5]}")
    
    # Step 1: Better date parsing
    print("\n[Step 1/4] Parsing date columns...")
    inferred_1 = 0
    for idx in df.index:
        if pd.isna(df.loc[idx, 'year']) or str(df.loc[idx, 'year']).strip() == '':
            dt = parse_date_flexible(df.loc[idx, date_col])
            if dt and dt.year:
                year = int(dt.year)
                if 1900 <= year <= 2100:
                    df.loc[idx, 'year'] = year
                    inferred_1 += 1
    print(f"  Filled {inferred_1} from improved date parsing")
    
    # Step 2: Relative date extraction
    print("[Step 2/4] Extracting relative dates...")
    inferred_2 = 0
    for idx in df.index:
        if pd.isna(df.loc[idx, 'year']) or str(df.loc[idx, 'year']).strip() == '':
            combined = None
            for col in ['combined_text', 'processed_text', 'description', 'title']:
                if col in df.columns and pd.notna(df.loc[idx, col]):
                    combined = df.loc[idx, col]
                    break
            
            ref_date = df.loc[idx, date_col] if date_col in df.columns else None
            if combined and ref_date:
                year = extract_relative_date_year(combined, ref_date)
                if year and 1900 <= year <= 2100:
                    df.loc[idx, 'year'] = year
                    inferred_2 += 1
    print(f"  Filled {inferred_2} from relative dates ('last year', 'X months ago', etc.)")
    
    # Step 3: Source median
    print("[Step 3/4] Using source median years...")
    inferred_3 = 0
    for idx in df.index:
        if pd.isna(df.loc[idx, 'year']) or str(df.loc[idx, 'year']).strip() == '':
            if 'source' in df.columns and df.loc[idx, 'source'] in source_medians:
                df.loc[idx, 'year'] = source_medians[df.loc[idx, 'source']]
                inferred_3 += 1
    print(f"  Filled {inferred_3} from source median year")
    
    # Step 4: Collection date fallback
    print("[Step 4/4] Using collection date fallback...")
    inferred_4 = 0
    for idx in df.index:
        if pd.isna(df.loc[idx, 'year']) or str(df.loc[idx, 'year']).strip() == '':
            if 'collected_at' in df.columns:
                dt = parse_date_flexible(df.loc[idx, 'collected_at'])
                if dt and dt.year:
                    year = int(dt.year)
                    if 1900 <= year <= 2100:
                        df.loc[idx, 'year'] = year
                        inferred_4 += 1
    print(f"  Filled {inferred_4} from collection date (collected_at)")
    
    # Count missing years after
    missing_after = df['year'].isna().sum() + (df['year'].astype(str).str.strip() == '').sum()
    total_filled = missing_before - missing_after
    
    print(f"\nSummary:")
    print(f"  Total filled: {total_filled} ({100.0*total_filled/missing_before:.1f}% of missing)")
    print(f"  Missing years after: {missing_after} ({100.0*missing_after/len(df):.1f}%)")
    
    # Year distribution
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    valid_years = df[df['year'].notna()]['year'].astype(int)
    
    print(f"\nYear distribution (top 20):")
    top_years = valid_years.value_counts().sort_index(ascending=False).head(20)
    for year, count in top_years.items():
        print(f"  {int(year)}: {count}")
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    return df

# ============================================================================
# PROCESS FILES
# ============================================================================

print("\n" + "="*80)
print("ADVANCED YEAR INFERENCE")
print("Methods: Date parsing + Relative dates + Source median + Collection date")
print("="*80)

# Process world_newsE_topics_assignments.csv
df_world = process_file_advanced(
    input_path='data_world/world_newsE_topics_assignments.csv',
    output_path='data_world/world_newsE_topics_assignments_final.csv',
    date_col='published'
)

# Process ldamulti.csv
df_lda = process_file_advanced(
    input_path='data_africa/ldamulti.csv',
    output_path='data_africa/ldamulti_final.csv',
    date_col='published_at'
)

print("\n" + "="*80)
print("✅ COMPLETE - Advanced year inference finished")
print("="*80)
print("\nNew files created:")
print("  • data_world/world_newsE_topics_assignments_final.csv")
print("  • data_africa/ldamulti_final.csv")
