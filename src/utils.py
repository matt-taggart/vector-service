import re

def linkify(text):
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, lambda url: '<a href="{}" target="_blank" rel="noopener noreferrer">{}</a>.'.format(url.group(0), url.group(0)), text)
    return text
