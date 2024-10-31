# Function to calculate average rating by app version in descending order
calculate_avg_rating_by_version_descending_funct = {
    "name": "calculate_avg_rating_by_version_descending",
    "description": "Calculate the average rating for each version of an app, sorted by highest rating, user count , and version (latest first). Returns the first 5 results.",
    "parameters": {
        "type": "object",
        "properties": {
            "context_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "author_app_version": {
                            "type": "string",
                            "description": "The version of the app from the review."
                        },
                        "review_rating": {
                            "type": "number",
                            "description": "The rating given in the review."
                        }
                    },
                    "required": ["author_app_version", "review_rating"],
                    "additionalProperties": False
                },
                "description": "List of dictionaries, each containing an app version and its associated review rating."
            }
        },
        "required": ["context_list"],
        "additionalProperties": False
    }
}

calculate_avg_rating_by_version_ascending_funct = {
    "name": "calculate_avg_rating_by_version_ascending",
    "description": "Calculate the average rating for each version of an app, sorted by lowest rating, user count , and version (latest first). Returns the first 5 results.",
    "parameters": {
        "type": "object",
        "properties": {
            "context_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "author_app_version": {
                            "type": "string",
                            "description": "The version of the app from the review."
                        },
                        "review_rating": {
                            "type": "number",
                            "description": "The rating given in the review."
                        }
                    },
                    "required": ["author_app_version", "review_rating"],
                    "additionalProperties": False
                },
                "description": "List of dictionaries, each containing an app version and its associated review rating."
            }
        },
        "required": ["context_list"],
        "additionalProperties": False
    }
}

count_appkeywords_in_reviews_funct = {
    "name": "count_appkeywords_in_reviews",
    "description": "Count the occurrences of Spotify App Competitors that being talked in the review ",
    "parameters": {
        "type": "object",
        "properties": {
            "context_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "review_text": {
                            "type": "string",
                            "description": "The text of the review containing keywords."
                        }
                    },
                    "required": ["review_text"],
                    "additionalProperties": False
                },
                "description": "List of dictionaries, each containing a review text."
            }
        },
        "required": ["context_list"],
        "additionalProperties": False
    }
}


get_latest_version_funct = {
    "name": "get_latest_version",
    "description": "Retrieve the latest version of an app from a list of app version entries, based on the semantic versioning scheme.",
    "parameters": {
        "type": "object",
        "properties": {
            "context_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "author_app_version": {
                            "type": "string",
                            "description": "The version of the app as specified by the author."
                        }
                    },
                    "required": ["author_app_version"],
                    "additionalProperties": False
                },
                "description": "List of dictionaries, each containing an app version."
            }
        },
        "required": ["context_list"],
        "additionalProperties": False
    }
}

get_oldest_version_funct = {
    "name": "get_oldest_version",
    "description": "Retrieve the oldest version of an app from a list of app version entries, based on the semantic versioning scheme.",
    "parameters": {
        "type": "object",
        "properties": {
            "context_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "author_app_version": {
                            "type": "string",
                            "description": "The version of the app as specified by the author."
                        }
                    },
                    "required": ["author_app_version"],
                    "additionalProperties": False
                },
                "description": "List of dictionaries, each containing an app version."
            }
        },
        "required": ["context_list"],
        "additionalProperties": False
    }
}


get_rating_list_funct = {
    "name": "get_rating_list",
    "description": "Retrieve a list of ratings based on specified filters, including app version, review date, and author name.",
    "parameters": {
        "type": "object",
        "properties": {
            "author_app_version": {
                "type": "string",
                "description": "The version of the app for which to retrieve ratings. If not specified, all versions will be included."
            },
            "review_date": {
                "type": "string",
                "description": "The date of the review to filter by. If not specified, all dates will be included."
            },
            "review_date_time": {
                "type": "string",
                "description": "The date and time of the review to filter by. If not specified, all times will be included."
            },
            "review_date_authorname": {
                "type": "string",
                "description": "The combination of review date and author name to filter by. If not specified, all combinations will be included."
            },
            "review_datetime_authorname": {
                "type": "string",
                "description": "The combination of review date, time, and author name to filter by. If not specified, all combinations will be included."
            },
            "author_name": {
                "type": "string",
                "description": "The name of the author whose reviews to filter. If not specified, all authors will be included."
            }
        },
        "required": [],
        "additionalProperties": False
    }
}

calculate_avg_rating_funct = {
    "name": "calculate_avg_rating",
    "description": "Calculate the average rating based on specified filters, including app version, review date, and author name. Returns a message if no ratings match the specified filters.",
    "parameters": {
        "type": "object",
        "properties": {
            "author_app_version": {
                "type": "string",
                "description": "The version of the app for which to calculate the average rating. If not specified, all versions will be included."
            },
            "review_date": {
                "type": "string",
                "description": "The date of the review to filter by. If not specified, all dates will be included."
            },
            "review_date_time": {
                "type": "string",
                "description": "The date and time of the review to filter by. If not specified, all times will be included."
            },
            "review_date_authorname": {
                "type": "string",
                "description": "The combination of review date and author name to filter by. If not specified, all combinations will be included."
            },
            "review_datetime_authorname": {
                "type": "string",
                "description": "The combination of review date, time, and author name to filter by. If not specified, all combinations will be included."
            },
            "author_name": {
                "type": "string",
                "description": "The name of the author whose reviews to filter. If not specified, all authors will be included."
            },
            "message_empty": {
                "type": "string",
                "description": "Message to return if no ratings are found matching the specified filters."
            }
        },
        "required": [],
        "additionalProperties": False
    }
}


get_review_likes_funct = {
    "name": "get_review_likes",
    "description": "Retrieve a list of review likes based on specified filter criteria such as review date, review time, and author name. Returns up to 10 likes.",
    "parameters": {
        "type": "object",
        "properties": {
            "context_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "author_name": {
                            "type": "string",
                            "description": "The name of the author who wrote the review."
                        },
                        "review_date": {
                            "type": "string",
                            "description": "The date the review was written (format: YYYY-MM-DD)."
                        },
                        "review_time": {
                            "type": "string",
                            "description": "The time the review was written (format: HH:MM)."
                        },
                        "review_likes": {
                            "type": "number",
                            "description": "The number of likes the review received."
                        }
                    },
                    "required": ["author_name", "review_date", "review_time", "review_likes"],
                    "additionalProperties": False
                },
                "description": "List of dictionaries, each containing review details including author name, date, time, and likes."
            },
            "review_date": {
                "type": "string",
                "description": "The date of the review to filter likes (optional)."
            },
            "review_date_time": {
                "type": "string",
                "description": "The combination of review date and time to filter likes (optional)."
            },
            "author_name": {
                "type": "string",
                "description": "The name of the author whose reviews to filter by (optional)."
            },
            "authorname_reviewdatetime": {
                "type": "string",
                "description": "Combination of author name and review date/time to filter likes (optional)."
            },
            "reviewdatetime_authorname": {
                "type": "string",
                "description": "Combination of review date/time and author name to filter likes (optional)."
            },
            "authorname_reviewdate": {
                "type": "string",
                "description": "Combination of author name and review date to filter likes (optional)."
            },
            "authorname_reviewtime": {
                "type": "string",
                "description": "Combination of author name and review time to filter likes (optional)."
            }
        },
        "required": ["context_list"],
        "additionalProperties": False
    }
}

get_review_text_funct = {
    "name": "get_review_text",
    "description": "Retrieve a list of review texts based on specified filter criteria such as author name, app version, review date, and review time. Returns up to 5 review texts.",
    "parameters": {
        "type": "object",
        "properties": {
            "context_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "author_name": {
                            "type": "string",
                            "description": "The name of the author who wrote the review."
                        },
                        "author_app_version": {
                            "type": "string",
                            "description": "The version of the app from which the review was made."
                        },
                        "review_date": {
                            "type": "string",
                            "description": "The date the review was written (format: YYYY-MM-DD)."
                        },
                        "review_time": {
                            "type": "string",
                            "description": "The time the review was written (format: HH:MM)."
                        },
                        "review_rating": {
                            "type": "number",
                            "description": "The rating given in the review."
                        },
                        "review_text": {
                            "type": "string",
                            "description": "The text content of the review."
                        }
                    },
                    "required": ["author_name", "author_app_version", "review_date", "review_time", "review_rating", "review_text"],
                    "additionalProperties": False
                },
                "description": "List of dictionaries, each containing review details including author name, app version, date, time, rating, and text."
            },
            "author_name": {
                "type": "string",
                "description": "The name of the author whose reviews to filter by (optional)."
            },
            "author_app_version": {
                "type": "string",
                "description": "The version of the app to filter reviews by (optional)."
            },
            "review_date": {
                "type": "string",
                "description": "The date of the review to filter texts (optional)."
            },
            "review_time": {
                "type": "string",
                "description": "The time of the review to filter texts (optional)."
            },
            "review_rating": {
                "type": "number",
                "description": "The rating of the review to filter texts (optional)."
            },
            "review_date_time": {
                "type": "string",
                "description": "The combination of review date and time to filter texts (optional)."
            },
            "review_time_date": {
                "type": "string",
                "description": "The combination of review time and date to filter texts (optional)."
            },
            "max_output": {
                "type": "integer",
                "description": "The maximum number of review texts to return (default is 5)."
            }
        },
        "required": ["context_list"],
        "additionalProperties": False
    }
}

analyze_reviews_funct = {
    "name": "analyze_reviews",
    "description": "Analyze a list of app reviews to calculate total reviews, count of non-NaN author app versions, non-NaN review texts, average review rating, count of non-NaN author names, and the number of unique author app versions.",
    "parameters": {
        "type": "object",
        "properties": {
            "context_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "review_id": {
                            "type": "string",
                            "description": "Unique identifier for the review."
                        },
                        "author_name": {
                            "type": "string",
                            "description": "Name of the author who wrote the review."
                        },
                        "review_date": {
                            "type": "string",
                            "description": "Date when the review was submitted."
                        },
                        "review_time": {
                            "type": "string",
                            "description": "Time when the review was submitted."
                        },
                        "review_likes": {
                            "type": "number",
                            "description": "Number of likes received by the review."
                        },
                        "review_rating": {
                            "type": "number",
                            "description": "Rating given in the review, usually between 1 and 5."
                        },
                        "author_app_version": {
                            "type": "string",
                            "description": "The version of the app from the review."
                        },
                        "review_text": {
                            "type": "string",
                            "description": "Text content of the review."
                        }
                    },
                    "required": [
                        "review_id",
                        "author_name",
                        "review_date",
                        "review_time",
                        "review_likes",
                        "review_rating",
                        "author_app_version",
                        "review_text"
                    ],
                    "additionalProperties": False
                },
                "description": "List of dictionaries, each containing information about an app review."
            }
        },
        "required": ["context_list"],
        "additionalProperties": False
    },
    "returns": {
        "type": "object",
        "properties": {
            "total_data": {
                "type": "number",
                "description": "Total number of reviews in the context_list."
            },
            "count_author_app_version": {
                "type": "number",
                "description": "Count of author app versions that are not 'NaN'."
            },
            "count_review_text": {
                "type": "number",
                "description": "Count of review texts that are not 'NaN'."
            },
            "average_review_rating": {
                "type": "number",
                "description": "Average rating of the reviews, rounded to two decimal places."
            },
            "count_author_name": {
                "type": "number",
                "description": "Count of author names that are not 'NaN'."
            },
            "group_of_author_app_version": {
                "type": "number",
                "description": "Count of unique author app versions excluding 'NaN'."
            }
        },
        "required": [
            "total_data",
            "count_author_app_version",
            "count_review_text",
            "average_review_rating",
            "count_author_name",
            "group_of_author_app_version"
        ],
        "additionalProperties": False
    }
}
