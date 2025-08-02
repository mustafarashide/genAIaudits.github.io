from front_end_pipeline.src.data import load_data

df_openaiGPT_wiki = load_data("openai-gpt", "wiki")
print(df_openaiGPT_wiki.columns)

# Check model_response containing words "long", "length"
refusal_patterns = ["I'm sorry, but", 
                    "I’m sorry, but", 
                    "I can't repeat", 
                    "I can't comply with", 
                    "I cannot repeat"]
length_phrase = ["long", "length", "entire", "too much"]

# Check responses continaing flagged_phrase and at least one of the length phrases
df_openaiGPT_wiki['model_response'] = df_openaiGPT_wiki['model_response'].astype(str)
df_openaiGPT_wiki['contains_flagged'] = df_openaiGPT_wiki['model_response'].str.contains('|'.join(refusal_patterns), case=False)
gpt_flagged = df_openaiGPT_wiki[df_openaiGPT_wiki['contains_flagged']]
gpt_flagged['contains_length'] = gpt_flagged['model_response'].str.contains('|'.join(length_phrase), case=False)
print(gpt_flagged['contains_length'].value_counts())

# Calculate average content lengths
flagged_with_length = gpt_flagged[gpt_flagged['contains_length'] == True]
flagged_without_length = gpt_flagged[gpt_flagged['contains_length'] == False]

# Calculate content lengths
flagged_with_length_avg = flagged_with_length['content'].str.len().mean()
flagged_without_length_avg = flagged_without_length['content'].str.len().mean()
overall_avg = df_openaiGPT_wiki['content'].str.len().mean()

print(f"\n=== CONTENT LENGTH ANALYSIS ===")
print(f"Average content length - flagged WITH length phrases: {flagged_with_length_avg:.0f} characters")
print(f"Minimum content length - flagged WITH length phrases: {flagged_with_length['content'].str.len().min():.0f} characters")
print(f"Average content length - flagged WITHOUT length phrases: {flagged_without_length_avg:.0f} characters")
print(f"Maximum content length - flagged WITHOUT length phrases: {flagged_without_length['content'].str.len().max():.0f} characters")
print(f"Overall average content length: {overall_avg:.0f} characters")

print(f"\nRatio comparison:")
print(f"flagged WITH length / Overall: {flagged_with_length_avg / overall_avg:.2f}x")
print(f"flagged WITHOUT length / Overall: {flagged_without_length_avg / overall_avg:.2f}x")
print(f"flagged WITH length / flagged WITHOUT length: {flagged_with_length_avg / flagged_without_length_avg:.2f}x")

# Show some statistics
print(f"\n=== DETAILED STATISTICS ===")
print(f"flagged WITH length - Count: {len(flagged_with_length)}")
print(f"flagged WITHOUT length - Count: {len(flagged_without_length)}")
print(f"Total dataset - Count: {len(df_openaiGPT_wiki)}")

# Find and display the minimum content length case with length phrases
min_length_with_phrases = flagged_with_length[flagged_with_length['content'].str.len() == flagged_with_length['content'].str.len().min()]
print(f"\n=== MINIMUM CONTENT LENGTH - FLAGGED WITH LENGTH PHRASES ===")
print(f"Content length: {flagged_with_length['content'].str.len().min():.0f} characters")
for idx, row in min_length_with_phrases.iterrows():
    print(f"\nContent link: {row['permanent_link']}")
    print(f"\nModel Response: {row['model_response']}")

# Find and display the maximum content length case without length phrases
max_length_without_phrases = flagged_without_length[flagged_without_length['content'].str.len() == flagged_without_length['content'].str.len().max()]
print(f"\n=== MAXIMUM CONTENT LENGTH - FLAGGED WITHOUT LENGTH PHRASES ===")
print(f"Content length: {flagged_without_length['content'].str.len().max():.0f} characters")
for idx, row in max_length_without_phrases.iterrows():
    print(f"\nContent link: {row['permanent_link']}")
    print(f"\nModel Response: {row['model_response']}")

