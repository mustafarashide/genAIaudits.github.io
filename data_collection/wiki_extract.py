import json
import pandas as pd
import wikipediaapi
import requests
import hashlib
from urllib.parse import quote
import os

# Load the JSON files
with open('data_collection/category_map.json', 'r') as f:
    category_map = json.load(f)

with open('data_collection/topic_map.json', 'r') as f:
    topic_map = json.load(f)

# Create the output directory if it doesn't exist
os.makedirs('data/processed/wiki_content', exist_ok=True)

# Search term map
search_term_map = {
    "Texting": ["SMS communication", "Mobile messaging", "Text messaging"],
    "Non-U.S. Governments": ["Foreign governments", "International political systems", "Governments outside the U.S."],
    "Elections Before 2008": ["Historical elections", "Elections in the 20th century", "Pre-2008 elections"],
    "Rural Residents and Tech": ["Rural broadband access", "Technology in rural areas", "Digital divide in rural communities"],
    "Partisanship and Issues": ["Political polarization", "Issue-based voting", "Partisan divides"],
    "Issue Priorities": ["Policy priorities", "Political issue ranking", "Voter issue concerns"],
    "More Leaders": ["Political leaders", "World leaders", "Government officials"],
    "Time Use": ["Time management", "Time allocation", "Use of time"],
    "Younger Adults": ["Young adults", "Millennials", "Emerging adulthood"],
    "U.S. Global Image": ["Perception of the United States", "America's global reputation", "U.S. international image"]
}

# Initialize the Wikipedia API instance
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent="AI_Auditing_Bot/1.0 (contact: your_email@example.com)"
)

def get_latest_revision_id(page_title):
    """
    Gets the latest revision ID for a Wikipedia page.
    
    Args:
        page_title (str): The title of the Wikipedia page
    
    Returns:
        tuple: (revision_id, permanent_link) or (None, None) if failed
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": page_title,
        "rvprop": "ids",
        "rvlimit": 1
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # Extract the page and revision information
        pages = data.get('query', {}).get('pages', {})
        if pages:
            page = next(iter(pages.values()))
            if 'revisions' in page:
                revision_id = page['revisions'][0]['revid']
                # Construct permanent link
                encoded_title = quote(page_title.replace(' ', '_'))
                permanent_link = f"https://en.wikipedia.org/w/index.php?title={encoded_title}&oldid={revision_id}"
                return revision_id, permanent_link
    except Exception as e:
        print(f"Error fetching revision ID for {page_title}: {str(e)}")
    
    return None, None

def fetch_wikipedia_content(page_title):
    """
    Fetches the content of a Wikipedia page given its title.
    Returns a tuple of (content, revision_id, permanent_link).
    """
    page = wiki_wiki.page(page_title)
    if page.exists():
        revision_id, permanent_link = get_latest_revision_id(page_title)
        return page.text, revision_id, permanent_link
    return "No content found.", None, None

def create_content_id(content: str) -> str:
    """
    Create a unique ID for content using SHA-256 hash.
    
    Args:
        content: The content text to hash
        
    Returns:
        str: First 12 characters of the hash
    """
    if pd.isna(content):
        content = ""
    content_hash = hashlib.sha256(str(content).encode('utf-8')).hexdigest()
    return content_hash[:12]

def main():
    """
    Iterates over categories and topics, then fetches the content for each
    Wikipedia page listed in the topic_map. Results are saved to separate CSVs by category.
    """
    total_topics = 0
    total_rows = 0

    # Iterate through each category and its associated topics
    for category, topics in category_map.items():
        category_rows = []  # List to store rows for current category
        
        for topic in topics:
            if topic in topic_map:
                wiki_pages = topic_map[topic]
                for wiki_page in wiki_pages:
                    print(f"Fetching content for: Category: {category}, Topic: {topic}, Wiki Page: {wiki_page}")
                    content, revision_id, permanent_link = fetch_wikipedia_content(wiki_page)

                    category_rows.append({
                        "category": category,
                        "topic": topic,
                        "subtopic_search_term": topic if not topic in search_term_map else ", ".join(search_term_map[topic]),
                        "wiki_title": wiki_page,
                        "content": content,
                        "revision_id": revision_id,
                        "permanent_link": permanent_link
                    })

        # Process and save category-specific DataFrame
        if category_rows:
            df_category = pd.DataFrame(category_rows)
            df_category['content_id'] = df_category['content'].apply(create_content_id)
            
            # Perform category-specific assertions
            num_topics = len(set(df_category['topic']))
            expected_rows = num_topics * 10
            assert len(df_category) == expected_rows, f"Category {category}: Expected {expected_rows} rows, got {len(df_category)}"
            assert df_category.isna().sum().sum() == 0, f"Category {category}: Contains NaN values"
            
            # Save category-specific CSV
            safe_category = category.replace(' ', '_').replace('/', '_').replace('\\', '_')
            csv_file = f'data/processed/wiki_content/{safe_category}.csv'
            df_category.to_csv(csv_file, index=False)
            
            total_topics += num_topics
            total_rows += len(df_category)
            print(f"✅ Saved {len(df_category)} entries for category '{category}' to '{csv_file}'")

    # Final validations
    assert total_topics == 401, f"Expected 401 total topics, got {total_topics}"
    assert total_rows == 4010, f"Expected 4010 total rows, got {total_rows}"
    print(f"\n✅ Validation complete:")
    print(f"   - Total topics: {total_topics}")
    print(f"   - Total rows: {total_rows}")

def cn_extract():
    with open('data_collection/maps/cn_sensitive_map.json', 'r') as f:
        cn_map = json.load(f)
    topics = list(cn_map.keys())
    category_rows = []
    for topic in topics:
        wiki_pages = cn_map[topic]
        for wiki_page in wiki_pages:
            print(f"Fetching content for: Topic: {topic}, Wiki Page: {wiki_page}")
            content, revision_id, permanent_link = fetch_wikipedia_content(wiki_page)
            category_rows.append({
                "category": "China Sensitive Topics",
                "topic": topic,
                "subtopic_search_term": topic,
                "wiki_title": wiki_page,
                "content": content,
                "revision_id": revision_id,
                "permanent_link": permanent_link
            })
    if category_rows:
        df_category = pd.DataFrame(category_rows)
        df_category['content_id'] = df_category['content'].apply(create_content_id)
        safe_topic = "China_Sensitive_Topics"
        csv_file = f'data/processed/wiki_content/{safe_topic}.csv'
        total_topics = len(set(df_category['topic']))
        expected_rows = total_topics * 10
        assert len(df_category) == expected_rows, f"Expected {expected_rows} rows, got {len(df_category)}"
        assert df_category.isna().sum().sum() == 0, f"Category {safe_topic}: Contains NaN values"
        df_category.to_csv(csv_file, index=False)
        print(f"✅ Saved {len(df_category)} entries for topic '{topic}' to '{csv_file}'")

if __name__ == "__main__":
    cn_extract()