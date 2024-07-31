# Step 1: Import Necessary Libraries and Initialize Vertex AI
import vertexai
import streamlit as st
from vertexai.preview import generative_models
from vertexai.preview.generative_models import ToolConfig

from vertexai.preview.generative_models import GenerativeModel, Tool, Part, Content, ChatSession, FunctionDeclaration, GenerationConfig
from flight_manager import search_flights, book_flight

# Initialize Vertex AI
project = "gemini-flights-mission-430419"
vertexai.init(project=project)

# Step 2: Define the Function Declaration
get_search_flights2 = FunctionDeclaration(
    name="get_search_flights2",
    description="Tool for searching a flight with origin, destination, departure date, seat type, and maximum cost",
    parameters={
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "The airport of departure for the flight given in airport code such as LAX, SFO, BOS, etc."
            },
            "destination": {
                "type": "string",
                "description": "The airport of destination for the flight given in airport code such as LAX, SFO, BOS, etc."
            },
            "departure_date": {
                "type": "string",
                "format": "date",
                "description": "The date of departure for the flight in YYYY-MM-DD format"
            },
            "max_cost": {
                "type": "string",
                "description": "The cost of the flight should be always less than max_cost, max_cost is a number"
            },
            "seat_type": {
                "type": "string",
                "description": "choose the type of the seat from the 3 options: 'economy', 'business', 'first_class'"
            }
        },
        "required": [
            "origin",
            "destination",
            "departure_date",
            "max_cost",
            "seat_type"
        ]
    },
)


book_flight_declaration = FunctionDeclaration(
    name="book_flight_declaration",
    description="Tool for booking a flight with a flight id, seat type and number of seats",
    parameters={
        "type": "object",
        "properties": {
            "flight_id": {
                "type": "integer",
                "description": "The id or number of the flight"
            },
            "num_seats": {
                "type": "integer",
                "description": "The number of seats to book"
            },
            "seat_type": {
                "type": "string",
                "description": "choose the type of the seat from the 3 options: 'economy', 'business', 'first_class'"
            }
        },
        "required": [
            "flight_id",
            "seat_type"
        ]
    },
)


# Step 3: Instantiate the Tool Class
search_tool = Tool(
    function_declarations=[get_search_flights2, book_flight_declaration],
)


tool_config = ToolConfig(
    function_calling_config=ToolConfig.FunctionCallingConfig(
        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,  # ANY mode forces the model to predict a function call from a subset of function names.
        allowed_function_names=[
        ],  # Allowed functions to call when mode is ANY, if empty any one of the provided functions will be called.
    )
)

# Step 4: Configure Generation Settings and Initialize the Model
config = GenerationConfig(temperature=0.4)

# Load Model with Configuration
model = GenerativeModel(
    "gemini-1.5-pro",
    tools=[search_tool],
    generation_config=config,
)


# helper function to unpack responses
def handle_response(response):
    response_args ={}
    n=0
    # Check for function call with intermediate step, always return response
    if response.candidates[0].content.parts[0].function_call:
        print("response.candidates[0].content.parts[0].function_call=\n", response.candidates[0].content.parts[0].function_call, "#######################################\n")
        response_args = response.candidates[0].content.parts[0].function_call.args
    elif len(response.candidates[0].content.parts) == 2:
        response_args = response.candidates[0].content.parts[1].function_call.args
        n=1

    if len(response_args) > 0:
        # Function call exists, unpack and load into a function
        #response_args = response.candidates[0].content.parts[0].function_call.args
        print("response_args=/n", response_args,"/n####################################")
        function_params = {}
        for key in response_args:
            value = response_args[key]
            function_params[key] = value
        print("function_params=\n", function_params, "#########################################\n")
        if response.candidates[0].content.parts[n].function_call.name == "get_search_flights2":
            results = search_flights(**function_params)
            print("results from search_flights=\n", results, "#########################################\n")
            if results:
                intermediate_response = chat.send_message(
                    Part.from_function_response(
                        name="get_search_flights2",
                        response=results
                    )
                )
                return intermediate_response.candidates[0].content.parts[0].text
        else:
            results = book_flight(**function_params)
            print("results from book_flight=\n", results, "#########################################\n")
            if results:
                intermediate_response = chat.send_message(
                    Part.from_function_response(
                        name="book_flight_declaration",
                        response=results
                    )
                )
                return intermediate_response.candidates[0].content.parts[0].text
        print("results =\n", results, "#########################################\n")
        return "Search Failed"
    else:
        # Return just text
        return response.candidates[0].content.parts[0].text


# helper function to display and send streamlit messages
def llm_function(chat: ChatSession, query):
    print("query=\n", query,"\n#######################################\n")
    response = chat.send_message(query)
    print("response from chat.send_message(query)=\n", response, "#######################################\n")
    output = handle_response(response)

    with st.chat_message("model"):
        st.markdown(output)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )
    st.session_state.messages.append(
        {
            "role": "model",
            "content": output
        }
    )


st.title("Gemini Flights")

chat = model.start_chat()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display and load to chat history
for index, message in enumerate(st.session_state.messages):
    content = Content(
        role=message["role"],
        parts=[Part.from_text(message["content"])]
    )

    if index != 0:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chat.history.append(content)

# For Initial message startup
if len(st.session_state.messages) == 0:
    # Invoke initial message
    initial_prompt = "Introduce yourself as a flights management assistant, ReX, powered by Google Gemini and designed to search/book flights. You use emojis to be interactive. For reference, the year for dates is 2024"

    llm_function(chat, initial_prompt)

# For capture user input
query = st.chat_input("Gemini Flights")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    llm_function(chat, query)
