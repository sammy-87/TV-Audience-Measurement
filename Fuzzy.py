from fuzzywuzzy import fuzz
from fuzzywuzzy import process
query = 'Man vs Wild'
choices = ['manoj', 'wildul ', 'Man Wild','wildlife','Wild Man','Manish']
# Get a list of matches ordered by score, default limit to 5
process.extract(query, choices)
# [('Barack H Obama', 95), ('Barack H. Obama', 95), ('B. Obama', 85)]
 
# If we want only the top one
# print(process.extractOne(query, choices))
# ('Barack H Obama', 95)
# print(fuzz.token_sort_ratio('Barack Obama', 'Barack H. Obama'))

print(fuzz.token_set_ratio('Taarzan The wonder Car ', 'Ananaya Taarlan the wonder car Man  Wild Hill Tata Charvi '))
