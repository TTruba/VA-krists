from fastapi import FastAPI
import openai
import requests
import time

app = FastAPI()

# Set your API keys securely (replace with your own secure method)
openai.api_key = ""
alpha_vantage_api_key = ""
exchange_rate_api_key = "" 

# Base URLs for APIs
alpha_vantage_url = "https://www.alphavantage.co/query"
exchange_rate_url = "https://v6.exchangerate-api.com/v6"

# Function to get stock data from Alpha Vantage
def get_stock_data(symbol):
    try:
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': '5min',
            'apikey': alpha_vantage_api_key
        }
        response = requests.get(alpha_vantage_url, params=params)
        data = response.json()

        if "Time Series (5min)" in data:
            latest_timestamp = sorted(data["Time Series (5min)"].keys())[-1]
            stock_info = data["Time Series (5min)"][latest_timestamp]
            close_price = float(stock_info['4. close'])
            return close_price, latest_timestamp
        else:
            return None, None
    except Exception as e:
        print(f"Error retrieving stock data for {symbol}: {str(e)}")
        return None, None

# Function to convert USD to EUR
def convert_to_euro(amount):
    try:
        response = requests.get(f"{exchange_rate_url}/{exchange_rate_api_key}/latest/USD")
        data = response.json()
        if 'conversion_rates' in data and 'EUR' in data['conversion_rates']:
            rate = data['conversion_rates']['EUR']
            return amount * rate
        else:
            print("Error fetching conversion rate.")
            return None
    except Exception as e:
        print(f"Error converting to EUR: {str(e)}")
        return None

# Create the assistant using OpenAI's beta API
assistant = openai.beta.assistants.create(
    name="Custom Stock Assistant",
    instructions="You are an assistant with access to custom tools that analyzes text to extract stock symbols.",
    model="gpt-4o-mini",
)

# Route to receive and process a user message
@app.post("/send-message/")
async def process_message_and_respond(thread_id: str, message: str):
    """
    Receive a message from the user and return a response from the virtual assistant.
    
    Args:
        thread_id (str): The ID of the conversation thread.
        message (str): The message sent by the user.
    
    Returns:
        dict: A dictionary containing the thread ID, the assistant's response, and the original message.
    """
    
    # Create a new thread for the conversation if not already present
    thread = openai.beta.threads.create()
    
    # Add the received message to the conversation thread
    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"Analyze the following text and respond with any stock symbols mentioned in one word: \"{message}\"."
    )
    
    # Run the assistant
    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    
    # Wait for the assistant's response
    while run.status != "completed":
        run = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        time.sleep(2)
    
    # Retrieve the assistant's response
    messages = openai.beta.threads.messages.list(thread_id=thread.id)
    final_answer = messages.data[0].content[0].text.value.strip()

    # If the assistant provided stock symbols, retrieve stock prices
    response_data = []
    if final_answer:
        symbols = final_answer.split(",")
        for symbol in symbols:
            symbol = symbol.strip().upper()
            close_price, timestamp = get_stock_data(symbol)
            if close_price is not None:
                price_in_euro = convert_to_euro(close_price)
                if price_in_euro:
                    response_data.append(f"{symbol} Stock Price in EUR: €{price_in_euro:.2f} (as of {timestamp})")
                else:
                    response_data.append(f"Error converting {symbol} price to EUR.")
            else:
                response_data.append(f"Stock information for {symbol} is not available.")
    
    return {
        "thread_id": thread_id,
        "response": "\n".join(response_data) if response_data else "No stock symbols found.",
        "message_received": message
    }

# Route to retrieve conversation history
@app.get("/conversation-history/")
async def conversation_history(thread_id: str):
    """
    Retrieve the conversation history for a specific thread.

    Args:
        thread_id (str): The ID of the conversation thread.

    Returns:
        dict: A dictionary containing the thread ID and a list of conversation messages.
    """
    # Mock response for conversation history; implement actual retrieval if needed
    return {
        "thread_id": thread_id,
        "conversation_history": [
            {"sender": "user", "content": "What is the price of AAPL?"},
            {"sender": "assistant", "content": "The stock price for AAPL is €150.00 (as of 2024-01-01 10:00:00)."}
        ]
    }
