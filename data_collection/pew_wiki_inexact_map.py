import requests
from bs4 import BeautifulSoup
import time
import json
import wikipediaapi

# List of exact topic titles to exclude
excluded_topics = {
    "American News Pathways 2020 Project",
    "American Trends Panel",
    "Future of the Internet (Project)",
    "Pew-Templeton Global Religious Futures Project",
    "State of the News Media (Project)",
    'Other Topics',
    'Test'
}

# Fetch topics from Pew Research website
url = 'https://www.pewresearch.org/topics/'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract topics, skipping excluded ones
topics = []
for section in soup.find_all('div', class_='wp-block-prc-block-taxonomy-index-az-list'):
    for li in section.find_all('li'):
        topic = li.find('a').text.strip()
        if topic not in excluded_topics:
            topics.append(topic)


def clean_topic_for_wikipedia(topic):
    topic = topic.replace('&', 'and')
    return topic


# Clean topics
cleaned_topics = set([clean_topic_for_wikipedia(topic) for topic in topics])
num_topics = len(cleaned_topics)
print(f"Found {num_topics} topics from Pew Research website.")

# Wikipedia API setup
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent="AI_Auditing_Bot/1.0 (contact: daviddai@example.com)"
)

def search_term(term):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': term,  
        'format': 'json',
        'srlimit': 15  # Buffer size to ensure we get enough results
    }

    response = requests.get(url, params=params)
    data = response.json()

    suggestions = []
    if 'query' in data and 'search' in data['query']:
        for result in data['query']['search']:
            title = result['title']
            # Skip disambiguation pages
            if '(disambiguation)' not in title:
                page = wiki_wiki.page(title)
                if page.exists():
                    suggestions.append(title)
                    # Break if we have 10 valid suggestions
                    if len(suggestions) == 10:
                        break
    return suggestions

def search_wikipedia_page(topic):
    # First, try searching the full topic
    suggestions = search_term(topic)

    # Print results
    if suggestions:
        print(f"✅ Found valid pages for '{topic}': {suggestions}")
    else:
        print(f"❌ No valid pages found for topic: '{topic}'")

    return suggestions


# Mapping topics
mapped_topics = {}

for topic in cleaned_topics:
    results = search_wikipedia_page(topic)
    if not results:
        results = ["NA"]

    mapped_topics[topic] = results

    time.sleep(0.1)

# Validate topic count
if len(set(mapped_topics.keys())) != num_topics:
    print(f"⚠️ Warning: Mapped topics count ({len(mapped_topics)}) does not match original topics count ({num_topics}).")
    raise ValueError("Mismatch in topic counts. Please check the mapping process.")

# Validate each topic has exactly 10 pages
for topic, pages in mapped_topics.items():
    if len(pages) != 10:
        print(f"⚠️ Warning: Topic '{topic}' does not have exactly 10 pages. Found {len(pages)} pages.")
        raise ValueError(f"Topic '{topic}' does not have exactly 10 pages.")

# Save mapping
with open('data_collection/maps/inexact_map.json', 'w') as f:
    json.dump(mapped_topics, f, indent=4)

print(f"\n✅ Successfully mapped {len(mapped_topics)} topics and wrote to topic_map.json.")