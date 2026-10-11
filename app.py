
import os
import certifi
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain.agents import create_agent


# -------------------- Configuration --------------------

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="centered",
)


# -------------------- Weather Tool --------------------

@tool
def get_weather_data(city: str) -> str:
    """Fetch current weather information for a given city."""

    if not WEATHERSTACK_API_KEY:
        return "Error: WEATHERSTACK_API_KEY is missing."

    try:
        response = requests.get(
            "https://api.weatherstack.com/current",
            params={
                "access_key": WEATHERSTACK_API_KEY,
                "query": city,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if "current" not in data:
            return f"Weather unavailable for {city}: {data.get('error', {}).get('info', 'Unknown error')}"

        current = data["current"]

        return (
            f"City: {data.get('location', {}).get('name', city)}\n"
            f"Temperature: {current['temperature']}°C\n"
            f"Condition: {current['weather_descriptions'][0]}\n"
            f"Humidity: {current['humidity']}%"
        )

    except requests.RequestException as error:
        return f"Weather API request failed: {error}"


# -------------------- Initialize AI Agent --------------------

@st.cache_resource
def get_agent():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing from your .env file.")

    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY is missing from your .env file.")

    search_tool = TavilySearch(
        max_results=2,
        tavily_api_key=TAVILY_API_KEY,
    )

    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        api_key=GROQ_API_KEY,
        temperature=0,
    )

    tools = [
        search_tool,
        get_weather_data,
    ]

    return create_agent(
        model=llm,
        tools=tools,
    )


# -------------------- Streamlit UI --------------------

st.title("🤖 AI Research Assistant")

st.write(
    "Ask questions and get answers using AI, "
    "web search, and current weather information."
)

st.divider()

with st.form("question_form"):
    question = st.text_area(
        "Your question",
        value=(
            "Find the capital of India and find its weather. "
            "Answer in only 2 lines."
        ),
        placeholder="Example: Find the latest Python updates.",
        height=120,
    )

    submitted = st.form_submit_button(
        "🔍 Get Answer",
        type="primary",
        use_container_width=True,
    )


# -------------------- Process Question --------------------

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("AI is processing your question..."):
                agent = get_agent()

                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": question.strip(),
                            }
                        ]
                    }
                )

                answer = response["messages"][-1].content

            st.subheader("Answer")
            st.markdown(answer)

        except Exception as error:
            st.error(f"Unable to generate an answer: {error}")
