from util import *
import os
import json
from PIL import Image
import pickle
import re

a = "m\u00b2"
b = a
a = re.sub(r'[^\x00-\x7F]', '', a)
print(a)
print(b)

def remove_sources_section(text):
    return re.sub(r'sources:.*', '', text, flags=re.IGNORECASE | re.DOTALL)

# Example usage
text = "This is some important content. qweqw \nsources:\nData from 2023, study XYZ, etc. Next sentence etc"
cleaned_text = remove_sources_section(text)
print(cleaned_text) 