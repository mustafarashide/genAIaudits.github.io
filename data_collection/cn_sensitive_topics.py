import requests
import time
import json
import wikipediaapi

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
                    if len(suggestions) == 15:
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

topics = [
    "One-child policy in China",
    "Anti-graft purge in China",
    "Political persecution of intellectuals in history in China",
    "Freedom of speech in China",
    "History of the Chinese Communist Party",
    "Basic economic regime of China",
    "Suffering under communist rule in China",
    "Liberalism in China",
    "The Cultural Revolution in China",
    "Love and spirituality in China",
    "Negative externality of marketization in China",
    "Fundamentalist Marxism in China",
    "Reflective intellectual history in China",
    "Peasant rebellion in China",
    "Environmental pollution in China",
    "Christianity in China",
    "State project and national event in China",
    "Traditional ethics in China",
    "Family value in China",
    "Politics of ethnic minorities in China",
]

# Mapping topics
mapped_topics = {}

for topic in topics:
    results = search_wikipedia_page(topic)
    if not results:
        results = ["NA"]

    mapped_topics[topic] = results

    time.sleep(0.1)

# Save mapping
with open('data_collection/maps/cn_sensitive_map_1.json', 'w') as f:
    json.dump(mapped_topics, f, indent=4)

print(f"\n✅ Successfully mapped {len(mapped_topics)} topics and wrote to cn_sensitive_map.json.")