from __future__ import annotations

import yt_dlp
import ast
import operator

from langchain.chat_models import init_chat_model 
from dotenv import load_dotenv 

from langgraph.graph import StateGraph
from langgraph.graph import MessagesState
from langchain_core.tools import tool
from langchain_community.tools import YouTubeSearchTool
from langgraph.types import Send
from langgraph.graph import MessagesState 
from langchain_core.messages import SystemMessage 

from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Annotated, List, Dict, Any

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# setting up llm 
load_dotenv() 
llm = init_chat_model("google_genai:gemini-2.5-flash-lite")


# setting up state and input structures
class SearchInput(BaseModel):
  search_input: str = Field(description="Search input string for youtube videos")
  no_of_vidoes: int = Field(description="No of vidoes to search")

class State(MessagesState): 
  youtube_videos: Annotated[List, operator.add]
  processed_metadata: Annotated[List[Dict[Any, Any]], operator.add]
  best_videos: str  
  no_of_vidoes: int

class UrlState(TypedDict): 
  url: str

# prompts
system_msg = """You are helpful youtube assitant that would help in searching videos specially related to topics of computer science."""
search_prompt = """Search atleast 10 youtube videos for the below search string

{search_string}
"""
final_prompt = """Now we have metadata information below from processing youtube vidoes. Based on the metadata info please return the 
youtube links for the best 4 videos that should not be shorts.

Some of the basis for selecting videos are 
1. view count
2. likes
3. video duration and should not be more than 5 hour.

Below is the video metadata

{metadata}
"""

# setting up nodes/helpers
def get_video_info(url: str):
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return info

def get_transcript_text(url: str):
    video_id = url.split("v=")[-1]

    ytt_api = YouTubeTranscriptApi()
    try: 
        transcript = ytt_api.fetch(video_id)
    except Exception as e: 
        print("Error fetching transcript:", e)
        return None

    formatter = TextFormatter()
    text_formatted = formatter.format_transcript(transcript)

    return text_formatted

def process_each_video(state: UrlState): 
    url = state["url"]
    info = get_video_info(url)
    # transcript_text = get_transcript_text(url)

    record = {
        "youtube_url": url,
        "video_id": info.get("id"),
        "title": info.get("title"),
        "description": info.get("description"),
        "channel": info.get("channel"),
        "upload_date": info.get("upload_date"),
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        # "transcript": transcript_text if transcript_text else "", # not adding transcription for now
    }
    return {"processed_metadata": [record]}

def final_process(state: State):
    output = ""
    i =0
    for record in state["processed_metadata"]: 
        output += str(i)
        output  += f", video url: {record['youtube_url']}"
        output  += f", video title: {record['title']}"
        output  += f", video description: {record['description']}"
        output  += f", video view count: {record['view_count']}"
        output  += f", video like count: {record['like_count']}"
        output += "\n"
        i+=1

    prompt = final_prompt.format(metadata=output)
    response = llm.invoke(prompt)

    return {"best_videos": response}

def yt_assistant(state: State): 
  sys_msg = SystemMessage(content=system_msg)
  return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

def yt_extract_metadata_and_transcript(state: State):
  videos_content = state["messages"][-1].content
  if type(videos_content) == str: # only would work for string content
    videos_list = ast.literal_eval(videos_content)
    if type(videos_list) == list:
      return {"youtube_videos":videos_list}
          
def route_urls(state: State):
  videos_list = state["youtube_videos"]
  return [Send("process_each_video", {"url": url}) for url in videos_list]

# setting up tools
@tool 
def yt_search(search_str: str) -> str:
  """Search the youtube for video content

  Args: 
    search_str: search string 
  """
  search_tool = YouTubeSearchTool()
  args_str = search_str + ",10"
  return search_tool.run(args_str)

tools = [yt_search]
llm_with_tools = llm.bind_tools(tools)

# dummy context
class Context(TypedDict):
    """Context parameters for the agent."""
    my_configurable_param: str

# setting up graph 
from langgraph.graph import START, StateGraph 
from langgraph.prebuilt import tools_condition 
from langgraph.prebuilt import ToolNode 

builder = StateGraph(State, context_schema=Context)

# nodes 
builder.add_node("yt_assistant", yt_assistant)
builder.add_node("yt_extract_metadata_and_transcript", yt_extract_metadata_and_transcript)
builder.add_node("process_each_video", process_each_video)
builder.add_node("final_process", final_process)
builder.add_node("tools", ToolNode(tools))

# edges
builder.add_edge(START, "yt_assistant")
builder.add_conditional_edges(
  "yt_assistant", 
  tools_condition,
)

builder.add_edge("tools", "yt_extract_metadata_and_transcript")
builder.add_conditional_edges(
    "yt_extract_metadata_and_transcript",
    route_urls,
    ["process_each_video"]
)
builder.add_edge("process_each_video", "final_process")

# build graph
graph = builder.compile(name="YT-Graph")
