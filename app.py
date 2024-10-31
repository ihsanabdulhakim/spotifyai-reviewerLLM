import openai
import streamlit as st
import pandas as pd
import os
from PIL import Image


from backend import (
    count_appkeywords_in_reviews, 
    calculate_avg_rating_by_version_ascending,
    calculate_avg_rating_by_version_descending, 
    calculate_avg_rating, get_rating_list,
    get_review_text, 
    analyze_reviews, 
    get_review_likes, 
    get_latest_version, 
    get_oldest_version,
    handle_tool_call,
    tools,
    MODEL,
    get_embedding,
    hbmodel,
    quality_score
)
from metadata import (
    calculate_avg_rating_by_version_descending_funct,
    calculate_avg_rating_by_version_ascending_funct,
    count_appkeywords_in_reviews_funct,
    get_latest_version_funct,
    get_oldest_version_funct,
    get_rating_list_funct,
    calculate_avg_rating_funct,
    get_review_likes_funct,
    get_review_text_funct
)

serialkey = "your-openai-api-key"
openai.api_key = serialkey


system_message = "Your name is AI-Spotify"
system_message += "You are a Spotify App reviewer based on the dataset from Google Play comments (already train you)"
system_message += "You only may answer in English Language or Bahasa Indonesia (preferably in English)"
system_message += "Apologize if you can't answer it"

# Example chat function integrating tools with error handling
def chat(message, history):
    if message is None:
        return "I'm sorry I can't answer that, could you please try asking another question?"
    
    messages = [{"role": "system", "content": system_message}]
    
    for human, assistant in history:
        messages.append({"role": "user", "content": human if human is not None else ""})
        messages.append({"role": "assistant", "content": assistant if assistant is not None else ""})

    messages.append({"role": "user", "content": message})

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )

        # Check if a tool call is required
        if response.choices[0].finish_reason == "tool_calls":
            tool_message = response.choices[0].message
            tool_response = handle_tool_call(tool_message)
            messages.append(tool_message)
            messages.append(tool_response)
            response = openai.ChatCompletion.create(model=MODEL, messages=messages)

        return response.choices[0].message.content
    
    except Exception as e:
        # Log the error if necessary
        print(f"Error occurred: {e}")
        return "I'm sorry I can't answer that, could you please try asking another question?"



# Example chat function integrating tools with error handling
def chat(message, history):
    if message is None:
        return "I'm sorry I can't answer that, could you please try asking another question?"
    
    messages = [{"role": "system", "content": system_message}]
    
    for human, assistant in history:
        messages.append({"role": "user", "content": human if human is not None else ""})
        messages.append({"role": "assistant", "content": assistant if assistant is not None else ""})

    messages.append({"role": "user", "content": message})

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )

        # Check if a tool call is required
        if response.choices[0].finish_reason == "tool_calls":
            tool_message = response.choices[0].message
            tool_response = handle_tool_call(tool_message)
            messages.append(tool_message)
            messages.append(tool_response)
            response = openai.ChatCompletion.create(model=MODEL, messages=messages)

        assistant_response = response.choices[0].message.content

        qualityscore = quality_score(message, assistant_response)
        # Create a result dictionary for the current chat
        current_result = {
            "user_message": message,
            "assistant_message": assistant_response,
            "quality_score": qualityscore
        }

        # Load existing data if the CSV file exists
        csv_file_path = "./results/resultspotifyai.csv"
        if os.path.exists(csv_file_path):
            existing_df = pd.read_csv(csv_file_path)
            # Create a new DataFrame for the current result
            new_df = pd.DataFrame([current_result])
            # Append the new result to the existing DataFrame
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            # Create a new DataFrame if the file does not exist
            combined_df = pd.DataFrame([current_result])

        # Save combined DataFrame back to CSV
        combined_df.to_csv(csv_file_path, index=False)

        return assistant_response  # Return the assistant's response
    
    except Exception as e:
        # Log the error if necessary
        print(f"Error occurred: {e}")
        return "I'm sorry I can't answer that, could you please try asking another question?"



# Function to add user input to history
def add_text(history, text):
    if text.strip():  # Ensure the text is not empty or just whitespace
        history.append((text, None))  # Append the user's message with no response yet
    return history

# Function to generate bot's response
def bot(history):
    user_message = history[-1][0]  # Get the last user message
    if user_message is not None:  # Ensure there's a valid user message
        bot_response = chat(user_message, history[:-1])  # Get bot response
        history[-1] = (user_message, bot_response)  # Add bot response to the last history item
    return history

# Initial greeting for the bot
initial_greeting = "Hello, AI-Spotify here! How may I assist you?"

# Streamlit app layout
st.title("AI-Spotify")

bot_image = Image.open("AI-Spotify.jpeg") 
st.image(bot_image, caption="AI-Spotify", width=100)

# Initialize chat history in session state
if 'history' not in st.session_state:
    st.session_state.history = [(None, initial_greeting)]

# Display chat history
for human, assistant in st.session_state.history:
    if human is not None:
        st.write(f"**User:** {human}")
    st.write(f"**AI-Spotify:** {assistant}")

# Function to send user input and handle bot response
def send_message():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.history = add_text(st.session_state.history, user_input)
        st.session_state.history = bot(st.session_state.history)
        # Reset the input field value for new input
        del st.session_state.user_input  # This will clear the input without modifying the widget after instantiation

# Text input for user message
st.text_input("User input", key="user_input", on_change=send_message)

# Clear chat history
if st.button("Clear"):
    st.session_state.history = [(None, initial_greeting)]