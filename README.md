# News Feed (Powered by Gemini)

Welcome to the **Global Insight Hub**, a Streamlit-based application powered by Gemini AI that aggregates news feeds from multiple sources, analyzes their content, and provides personalized updates, summaries, or recommendations. This tool utilizes advanced natural language processing and caching to deliver real-time insights, making it perfect for staying updated or curating news for research or personal use.

## Features

- Aggregate news feeds from RSS sources or APIs using `feedparser` and `requests` libraries.
- Generate embeddings and cache them in a Redis store for quick access and personalization.
- Offer three display modes: **Latest Updates**, **Customized Recommendations**, and **Topic Summaries** using Gemini AI.
- Save and reload cached data for offline access or faster reloads.
- Interactive interface built with Streamlit for seamless navigation.

## Tech Stack

- **Frontend**: Streamlit for a responsive and user-friendly interface.
- **Backend**: Python with `feedparser` for RSS aggregation and `requests` for API calls.
- **AI**: Gemini AI (via Google Generative AI API) for content analysis and recommendations.
- **Storage**: Redis for caching embeddings and FAISS for vector storage (optional).
- **Utilities**: `dotenv` for environment variable management, `tenacity` for retry logic.

## Prerequisites

- Python 3.8 or higher
- Git installed on your system
- A Google API key from [Google AI Studio](https://aistudio.google.com/) for Gemini AI access

## Installation

### Clone the Repository
```bash
git clone https://github.com/Mohitt029/news-feed.git
cd news-feed
```
### Set Up a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the Streamlit app
```bash
streamlit run main.py
```
