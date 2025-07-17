# Wikipedia API setup
import wikipediaapi
import requests
import json

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
        'srlimit': 10  # Number of results to return
    }

    response = requests.get(url, params=params)
    data = response.json()

    suggestions = []
    if 'query' in data and 'search' in data['query']:
        for result in data['query']['search']:
            title = result['title']
            page = wiki_wiki.page(title)
            if page.exists():
                suggestions.append(title)
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

# Mapping of revised search terms
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

def search_with_alternatives(original_term):
    alternative = search_term_map.get(original_term, [original_term])
    all_results = set()
    
    for term in alternative:
        try:
            articles = set(search_wikipedia_page(term)) 
            all_results.update(articles)
        except Exception as e:
            print(f"Error searching for '{term}': {e}")
    
    return sorted(all_results)

# Dictionary to hold final results
search_results = {}

for term in search_term_map:
    print(f"=== Results for improved '{term}' ===")
    result = search_with_alternatives(term)
    search_results[term] = result

# Save results to JSON file
with open("data_collection/maps/q_3_terms_rematch.json", "w", encoding="utf-8") as f:
    json.dump(search_results, f, indent=4, ensure_ascii=False)

print("Search results saved to 'q_3_terms_rematch.json'.")