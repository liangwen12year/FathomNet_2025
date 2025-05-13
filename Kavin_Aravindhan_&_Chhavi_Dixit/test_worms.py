"""
Script to query FathomNet's WoRMS search API for taxonomic information.

Example:
---------
Queries the WoRMS database for the term "Funiculina" and prints the result.
"""

import requests
import urllib.parse

# --------------------------------------------------------
# Define search term (e.g., genus or family name)
# --------------------------------------------------------
term = "Funiculina"

# Encode search term for URL compatibility
encoded_term = urllib.parse.quote_plus(term)

# Construct the FathomNet WoRMS API URL
url = f"https://fathomnet.org/api/worms/search?term={encoded_term}"

# --------------------------------------------------------
# Send GET request to the API
# --------------------------------------------------------
resp = requests.get(url)

# --------------------------------------------------------
# Output Results
# --------------------------------------------------------
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    # Pretty-print JSON response if request is successful
    print(resp.json())
else:
    print("Failed to retrieve data from WoRMS API.")
