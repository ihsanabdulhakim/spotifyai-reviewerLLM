import json
import pandas as pd
import openai
import numpy as np
import torch
import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel
from metadata import (
    calculate_avg_rating_by_version_descending_funct,
    calculate_avg_rating_by_version_ascending_funct,
    count_appkeywords_in_reviews_funct,
    get_latest_version_funct,
    get_oldest_version_funct,
    get_rating_list_funct,
    calculate_avg_rating_funct,
    get_review_likes_funct,
    get_review_text_funct,
    analyze_reviews_funct
)

MODEL = "ft:gpt-4o-mini-2024-07-18:personal:spotify-chatbot-reviewer-updated:AO57Uu5E"

from huggingface_hub import login


hf_token = "your-huggingface-token"
login(hf_token, add_to_git_credential=False)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
hbmodel = AutoModel.from_pretrained("distilbert-base-uncased")
context_list = []

def get_embedding(text):
    # Tokenize and get the model outputs
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = hbmodel(**inputs)
    # Use mean pooling on the token embeddings to get a sentence embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def quality_score(user_message, assistant_message):
    # Get embeddings for both messages
    user_embedding = get_embedding(user_message)
    assistant_embedding = get_embedding(assistant_message)

    # Calculate dot product
    dot_product = np.dot(user_embedding, assistant_embedding)

    # Calculate magnitudes
    user_magnitude = np.linalg.norm(user_embedding)
    assistant_magnitude = np.linalg.norm(assistant_embedding)

    # Calculate cosine similarity manually
    if user_magnitude == 0 or assistant_magnitude == 0:
        cosine_similarity = 0.0  # Avoid division by zero
    else:
        cosine_similarity = dot_product / (user_magnitude * assistant_magnitude)

    # Convert similarity score to a quality score (scale it between 0 and 100)
    score = float(cosine_similarity) * 100
    return score

# Open the JSON Lines file and load its contents into context_list
with open('./dataset/spotifyreviewscombined.jsonl', 'r') as file:
    for line in file:
        # Load each line as a JSON object and append it to the context_list
        json_object = json.loads(line)
        context_list.append(json_object)

message_empty = "I'm sorry I can't find suitable answer from our database, Could you ask another question?"

# List of keywords to count in each review_text
keywords = ['pandora', 'youtube music', 'amazon', 'shazam', 'joox', 'apple music', 'deezer', 'soundcloud', 'iheartradio', 'soundhound']

# Function to count occurrences of each keyword in review_text
def count_appkeywords_in_reviews(context_list):
    # Initialize a dictionary to store keyword counts
    keyword_counts = {keyword: 0 for keyword in keywords}

    for item in context_list:
        review_text = item['review_text'].lower()  # Convert text to lowercase for case insensitivity

        # Check for each keyword and increment count only once if keyword is found
        for keyword in keywords:
            if keyword in review_text:
                keyword_counts[keyword] += 1  # Count only once per row

    # Sort the dictionary by count values in descending order
    sorted_counts = dict(sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True))
    return sorted_counts

def calculate_avg_rating_by_version_ascending(context_list):
    # Initialize a dictionary to hold the sum of ratings and count of reviews for each version
    version_ratings = {}
    
    # Function to convert version string to a tuple of integers for sorting
    def version_key(version):
        try:
            return tuple(map(int, version.split('.')))
        except ValueError:
            return (float('inf'),)  # Return a tuple that will sort to the end

    # Iterate through each item in the context_list
    for item in context_list:
        version = item.get('author_app_version')  # Use get() to avoid KeyError
        rating = item.get('review_rating')  # Use get() to avoid KeyError
        
        # Check if the version is valid and the rating is a number
        if version and rating is not None and isinstance(rating, (int, float)):
            # If the version is not in the dictionary, initialize it
            if version not in version_ratings:
                version_ratings[version] = {'total_rating': 0, 'count': 0}
            
            # Update the total rating and count for the version
            version_ratings[version]['total_rating'] += rating
            version_ratings[version]['count'] += 1

    # Calculate the average rating for each version, rounded to 2 decimal places
    avg_rating_by_version = {
        version: {
            'average_rating': round(data['total_rating'] / data['count'], 2),  # Round to 2 decimal places
            'user_count': data['count']  # Count of users for this version
        }
        for version, data in version_ratings.items()
        if data['count'] > 0  # Ensure there are reviews to avoid division by zero
    }

    # Sort the dictionary by average rating (ascending), user count (descending), and then by version (latest first)
    sorted_avg_rating_by_version = dict(sorted(
        avg_rating_by_version.items(),
        key=lambda item: (item[1]['average_rating'], -item[1]['user_count'], version_key(item[0])),
        reverse=False  # Sort by average rating first (ascending)
    ))

    # Return only the first 5 items
    return dict(list(sorted_avg_rating_by_version.items())[:5])


def calculate_avg_rating_by_version_descending(context_list):
    # Initialize a dictionary to hold the sum of ratings and count of reviews for each version
    version_ratings = {}
    
    # Function to convert version string to a tuple of integers for sorting
    def version_key(version):
        try:
            return tuple(map(int, version.split('.')))
        except ValueError:
            return (float('inf'),)  # Return a tuple that will sort to the end

    # Iterate through each item in the context_list
    for item in context_list:
        version = item.get('author_app_version')  # Use get() to avoid KeyError
        rating = item.get('review_rating')  # Use get() to avoid KeyError
        
        # Check if the version is valid and the rating is a number
        if version and rating is not None and isinstance(rating, (int, float)):
            # If the version is not in the dictionary, initialize it
            if version not in version_ratings:
                version_ratings[version] = {'total_rating': 0, 'count': 0}
            
            # Update the total rating and count for the version
            version_ratings[version]['total_rating'] += rating
            version_ratings[version]['count'] += 1

def get_latest_version(context_list):
    # Initialize a variable to hold the latest version found
    latest_version = None
    
    # Function to convert version string to a tuple of integers for comparison
    def version_key(version):
        try:
            return tuple(map(int, version.split('.')))
        except ValueError:
            return (float('-inf'),)  # Return a tuple that will sort to the beginning

    # Iterate through each item in the context_list
    for item in context_list:
        version = item.get('author_app_version')  # Use get() to avoid KeyError
        
        # Check if the version is valid
        if version:
            # Update the latest version if it is not set or is newer than the current latest_version
            if latest_version is None or version_key(version) > version_key(latest_version):
                latest_version = version

    return latest_version


def get_oldest_version(context_list):
    # Initialize a variable to hold the oldest version found
    oldest_version = None
    
    # Function to convert version string to a tuple of integers for comparison
    def version_key(version):
        try:
            return tuple(map(int, version.split('.')))
        except ValueError:
            return (float('inf'),)  # Return a tuple that will sort to the end

    # Iterate through each item in the context_list
    for item in context_list:
        version = item.get('author_app_version')  # Use get() to avoid KeyError
        
        # Check if the version is valid
        if version:
            # Update the oldest version if it is not set or is older than the current oldest_version
            if oldest_version is None or version_key(version) < version_key(oldest_version):
                oldest_version = version

    return oldest_version

def get_rating_list(
    author_app_version=None,
    review_date=None,
    review_date_time=None,
    review_date_authorname=None,
    review_datetime_authorname=None,
    author_name=None
):
    # Normalize filter inputs to lowercase if they are strings
    author_app_version = author_app_version.lower() if isinstance(author_app_version, str) else author_app_version
    review_date = review_date.lower() if isinstance(review_date, str) else review_date
    review_date_time = review_date_time.lower() if isinstance(review_date_time, str) else review_date_time
    review_date_authorname = review_date_authorname.lower() if isinstance(review_date_authorname, str) else review_date_authorname
    review_datetime_authorname = review_datetime_authorname.lower() if isinstance(review_datetime_authorname, str) else review_datetime_authorname
    author_name = author_name.lower() if isinstance(author_name, str) else author_name

    ratings = []

    # Iterate through each item in the context_list
    for item in context_list:
        # Create combinations for review_date_authorname and review_datetime_authorname, converting to lowercase
        review_date_authorname_combo1 = f"{item.get('author_name', '').lower()} {item.get('review_date', '').lower()}"
        review_date_authorname_combo2 = f"{item.get('review_date', '').lower()} {item.get('author_name', '').lower()}"
        
        review_datetime_authorname_combo1 = f"{item.get('author_name', '').lower()} {item.get('review_date', '').lower()} {item.get('review_time', '').lower()}"
        review_datetime_authorname_combo2 = f"{item.get('review_date', '').lower()} {item.get('review_time', '').lower()} {item.get('author_name', '').lower()}"

        # Filter by specified parameters
        if (author_app_version is None or item.get('author_app_version', '').lower() == author_app_version) and \
           (review_date is None or item.get('review_date', '').lower() == review_date) and \
           (review_date_time is None or f"{item.get('review_date', '').lower()} {item.get('review_time', '').lower()}" == review_date_time) and \
           (review_date_authorname is None or review_date_authorname in [review_date_authorname_combo1, review_date_authorname_combo2]) and \
           (review_datetime_authorname is None or review_datetime_authorname in [review_datetime_authorname_combo1, review_datetime_authorname_combo2]) and \
           (author_name is None or item.get('author_name', '').lower() == author_name):
            
            rating = item.get('review_rating')
            
            # Check if the rating is a valid number
            if rating is not None and isinstance(rating, (int, float)):
                ratings.append(rating)

    return ratings  # Return the list of ratings


def calculate_avg_rating(
    author_app_version=None,
    review_date=None,
    review_date_time=None,
    review_date_authorname=None,
    review_datetime_authorname=None,
    author_name=None,
    message_empty="No ratings found for the specified filters."
):
    # Get the list of ratings based on filters
    ratings = get_rating_list(
        author_app_version,
        review_date,
        review_date_time,
        review_date_authorname,
        review_datetime_authorname,
        author_name
    )

    # Calculate and return the average rating if there are valid ratings
    if ratings:
        total_rating = sum(ratings)
        count = len(ratings)
        return round(total_rating / count, 2)
    else:
        return message_empty  # Return message if no ratings match the specified filters
    
def get_review_likes(
    review_date=None,
    review_date_time=None,
    author_name=None,
    authorname_reviewdatetime=None,
    reviewdatetime_authorname=None,
    authorname_reviewdate=None,
    authorname_reviewtime=None
):
    # Normalize filter inputs to lowercase if they are strings
    review_date = review_date.lower() if isinstance(review_date, str) else review_date
    review_date_time = review_date_time.lower() if isinstance(review_date_time, str) else review_date_time
    author_name = author_name.lower() if isinstance(author_name, str) else author_name
    authorname_reviewdatetime = authorname_reviewdatetime.lower() if isinstance(authorname_reviewdatetime, str) else authorname_reviewdatetime
    reviewdatetime_authorname = reviewdatetime_authorname.lower() if isinstance(reviewdatetime_authorname, str) else reviewdatetime_authorname
    authorname_reviewdate = authorname_reviewdate.lower() if isinstance(authorname_reviewdate, str) else authorname_reviewdate
    authorname_reviewtime = authorname_reviewtime.lower() if isinstance(authorname_reviewtime, str) else authorname_reviewtime

    review_likes = []
    max_likes = 10

    # Iterate through each item in the context_list
    for item in context_list:
        # Create combinations for authorname_reviewdate, authorname_reviewtime, and reviewdatetime_authorname
        authorname_reviewdate_combo = f"{item.get('author_name', '').lower()} {item.get('review_date', '').lower()}"
        authorname_reviewtime_combo = f"{item.get('author_name', '').lower()} {item.get('review_time', '').lower()}"
        
        review_datetime_authorname_combo = f"{item.get('review_date', '').lower()} {item.get('review_time', '').lower()} {item.get('author_name', '').lower()}"

        # Filter by specified parameters
        if (review_date is None or item.get('review_date', '').lower() == review_date) and \
           (review_date_time is None or f"{item.get('review_date', '').lower()} {item.get('review_time', '').lower()}" == review_date_time) and \
           (author_name is None or item.get('author_name', '').lower() == author_name) and \
           (authorname_reviewdatetime is None or authorname_reviewdatetime in [review_datetime_authorname_combo]) and \
           (reviewdatetime_authorname is None or reviewdatetime_authorname in [review_datetime_authorname_combo]) and \
           (authorname_reviewdate is None or authorname_reviewdate in [authorname_reviewdate_combo]) and \
           (authorname_reviewtime is None or authorname_reviewtime in [authorname_reviewtime_combo]):
            
            likes = item.get('review_likes')  # Assuming the like count is stored in 'review_likes'
            
            # Check if the likes is a valid number
            if likes is not None and isinstance(likes, (int, float)):
                review_likes.append(likes)
            
            if len(review_likes) >= max_likes:
                break

    return review_likes  # Return the list of review likes

def get_review_text(
    author_name=None,
    author_app_version=None,
    review_date=None,
    review_time=None,
    review_rating=None,
    review_date_time=None,
    review_time_date=None,
    max_output=5
):
    results = []

    # Normalize filter inputs to lowercase if they are strings
    author_name = author_name.lower() if isinstance(author_name, str) else author_name
    author_app_version = author_app_version.lower() if isinstance(author_app_version, str) else author_app_version
    review_date = review_date.lower() if isinstance(review_date, str) else review_date
    review_time = review_time.lower() if isinstance(review_time, str) else review_time
    review_date_time = review_date_time.lower() if isinstance(review_date_time, str) else review_date_time
    review_time_date = review_time_date.lower() if isinstance(review_time_date, str) else review_time_date

    # Iterate through each item in the context_list
    for item in context_list:
        # Create combinations for review_date_time and review_time_date, converting to lowercase
        review_date_time_combo = f"{item.get('review_date', '').lower()} {item.get('review_time', '').lower()}"
        review_time_date_combo = f"{item.get('review_time', '').lower()} {item.get('review_date', '').lower()}"

        # Filter by specified parameters
        if (author_name is None or item.get('author_name', '').lower() == author_name) and \
           (author_app_version is None or item.get('author_app_version', '').lower() == author_app_version) and \
           (review_date is None or item.get('review_date', '').lower() == review_date) and \
           (review_time is None or item.get('review_time', '').lower() == review_time) and \
           (review_rating is None or item.get('review_rating') == review_rating) and \
           (review_date_time is None or review_date_time == review_date_time_combo) and \
           (review_time_date is None or review_time_date == review_time_date_combo):
            
            review_text = item.get('review_text')
            
            # Add review text to results if it exists
            if review_text:
                results.append(review_text)
                
                # Limit results to max_output
                if len(results) >= max_output:
                    break

    # Return results if any matched, otherwise a default message
    return results if results else "No matching reviews found."
    return results if results else message_empty

def analyze_reviews(context_list):
    total_data = len(context_list)
    
    # Initialize counts and sums for averages
    count_author_app_version = 0
    count_review_text = 0
    sum_review_rating = 0
    count_review_rating = 0
    count_author_name = 0
    unique_author_app_versions = set()

    for review in context_list:
        # Count non-'NaN' author_app_version
        if review.get('author_app_version') != 'NaN':
            count_author_app_version += 1
            unique_author_app_versions.add(review.get('author_app_version'))
            
        # Count non-'NaN' review_text
        if review.get('review_text') != 'NaN':
            count_review_text += 1
            
        # Sum review_rating and count valid ratings
        if review.get('review_rating') != 'NaN':
            sum_review_rating += review.get('review_rating')
            count_review_rating += 1
            
        # Count non-'NaN' author_name
        if review.get('author_name') != 'NaN':
            count_author_name += 1


            
    # Calculate averages where applicable
    average_review_rating = sum_review_rating / count_review_rating if count_review_rating > 0 else 0
    average_review_rating_not_nan = average_review_rating  # Same as above since we are only counting valid ratings

    return {
        "total_data": total_data,
        "count_author_app_version": count_author_app_version,
        "count_review_text": count_review_text,
        "average_review_rating": round((average_review_rating),2),
        # "average_review_rating_not_nan": average_review_rating_not_nan,
        "count_author_name": count_author_name,
        "group_of_author_app_version": len(unique_author_app_versions)
    }
result = analyze_reviews(context_list)

def takekeywordfromlist(data, SYSTEMPROMPT):
    """Generate a sentence from the list."""
    try:
        # Convert the input data to a string representation
        data_str = str(data)

        # Create the completion request
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"{SYSTEMPROMPT}: '{data_str}'"
                }
            ],
            max_tokens=100,  # Adjust based on desired output length
            temperature=0.7,  # Controls randomness; adjust as needed
        )

        # Extract the generated text from the response
        generatetext = response.choices[0].message.content
        return generatetext

    except Exception as e:
        print(f"Error: {e}")
        return "I'm sorry I can't answer that question; could you ask anything else?"

# Handle tool calls in the conversation
def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)

    if tool_call.function.name == "calculate_avg_rating_by_version_descending":
        context_list = arguments.get("context_list")
        listresult = calculate_avg_rating_by_version_descending(context_list)
        response = listresult
    elif tool_call.function.name == "calculate_avg_rating_by_version_ascending":
        context_list = arguments.get("context_list")
        listresult = calculate_avg_rating_by_version_ascending(context_list)
        response = listresult
    elif "competitors" in message or "compete" in message or "rival" in message:
        # Handle the case for counting app keywords in reviews
        context_list = arguments.get("context_list")
        SYSTEMPROMPT = "Analyze the many usage of Spotify competitors from this list of streaming app music names and how many found in our database."
        results = count_appkeywords_in_reviews(context_list)
        response = takekeywordfromlist(results, SYSTEMPROMPT)
    elif tool_call.function.name == "get_latest_version":
        context_list = arguments.get("context_list")
        SYSTEMPROMPT = "You have to make sentence that state that this number is the latest version of the app"
        results = get_latest_version(context_list)
        response = takekeywordfromlist(results,SYSTEMPROMPT)
    elif tool_call.function.name == "get_oldest_version":
        context_list = arguments.get("context_list")
        SYSTEMPROMPT = "You have to make sentence that state that this number is the oldest version of the app"
        results = get_oldest_version(context_list)
        response = takekeywordfromlist(results,SYSTEMPROMPT)
    elif tool_call.function.name == "get_rating_list":
        context_list = arguments.get("context_list")
        review_date = arguments.get("review_date")
        review_date_time = arguments.get("review_date_time")
        author_name = arguments.get("author_name")
        authorname_reviewdatetime = arguments.get("authorname_reviewdatetime")
        reviewdatetime_authorname = arguments.get("reviewdatetime_authorname")
        authorname_reviewdate = arguments.get("authorname_reviewdate")
        authorname_reviewtime = arguments.get("authorname_reviewtime")
        SYSTEMPROMPT = "You have to make sentence analyze of number or list of rating that found in the database"
        # Create a dictionary to hold the parameters to pass
        params = {}
        if author_name: 
            params['author_name'] = author_name
        if review_date: 
            params['review_date'] = review_date
        if review_date_time: 
            params['review_date_time'] = review_date_time
        if authorname_reviewdatetime: 
            params['authorname_reviewdatetime'] = authorname_reviewdatetime
        if reviewdatetime_authorname: 
            params['reviewdatetime_authorname'] = reviewdatetime_authorname
        if authorname_reviewdate: 
            params['authorname_reviewdate'] = authorname_reviewdate
        if authorname_reviewtime: 
            params['authorname_reviewtime'] = authorname_reviewtime
        
        results = get_rating_list(**params)  
        response = takekeywordfromlist(results, SYSTEMPROMPT)
            
    # elif tool_call.function.name == "calculate_avg_rating":
    #     response = 
    # elif tool_call.function.name == "get_review_likes":
    #     response = 
    # elif tool_call.function.name == "get_review_text":
    #     response = 
    elif tool_call.function.name == "analyze_reviews":
        context_list = arguments.get("context_list")
        SYSTEMPROMPT = "You have to make sentence for analysis this statistics data"
        results = analyze_reviews(context_list)
        response = takekeywordfromlist(results,SYSTEMPROMPT)
    
    return {
        "role": "tool",
        "content": json.dumps({"response": response}),
        "tool_call_id": message.tool_calls[0].id
    }


tools = [
    {"type": "function", "function": calculate_avg_rating_by_version_descending_funct},
    {"type": "function", "function": calculate_avg_rating_by_version_ascending_funct},
    {"type": "function", "function": count_appkeywords_in_reviews_funct},
    {"type": "function", "function": get_latest_version_funct},
    {"type": "function", "function": get_oldest_version_funct},
    {"type": "function", "function": get_rating_list_funct},
    {"type": "function", "function": calculate_avg_rating_funct},
    {"type": "function", "function": get_review_likes_funct},
    {"type": "function", "function": get_review_text_funct},
    {"type": "function", "function": analyze_reviews_funct}
]