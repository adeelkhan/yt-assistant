# 🎬 YouTube Agent

A computer science assistant that helps one get good suggested videos YouTube videos on any topic

---

## 🚀 Tech Stack

- **Python** 3.11
- **LangGraph**
- **LangChain**
- **yt-dlp**
- **youtube-search** `==2.1.2`
- **youtube-transcript-api** `==1.2.3`

---

## 🧠 Overview

The core logic is defined in **`src/agent/graph.py`**, showcasing logic that helps users discover top videos based on criteria such as **views**, **duration**, and more.

This agent can be extended into more advanced workflows, which can be visualized and debugged using **LangGraph Studio**.

---

### 1. Install Dependencies

Make sure to install required dependencies along with the LangGraph CLI, which is used to run the local development server.

```bash
cd path/to/your/app
pip install -e . "langgraph-cli[inmem]"

# Install Dependencies
pip install -r requirements.txt

# setup personal tokens for llm, tools or langsmith
cp .env.example .env

```
