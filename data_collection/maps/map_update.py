import json

# File paths
rematch_path = "data_collection/maps/inexact_manual_rematch.json"
inexact_path = "data_collection/maps/inexact_map.json"
output_path = "data_collection/topic_map.json"

# Load data
with open(rematch_path, "r", encoding="utf-8") as f:
    rematch_map = json.load(f)

with open(inexact_path, "r", encoding="utf-8") as f:
    inexact_map = json.load(f)

# Replace values where keys match
for key in inexact_map:
    if key in rematch_map:
        inexact_map[key] = rematch_map[key]

# Save the updated map
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(inexact_map, f, indent=4, ensure_ascii=False)

print(f"Updated map saved to '{output_path}'")