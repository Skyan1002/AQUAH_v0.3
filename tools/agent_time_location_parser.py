import json
import os
import re
from typing import Optional

from crewai import Agent, Task, Crew
from openai import OpenAI


def _should_search_flash_flood(input_text: str) -> bool:
    lowered = input_text.lower()
    return "flash flood" in lowered or "flashflood" in lowered


def fetch_flash_flood_context(input_text: str) -> Optional[dict]:
    if not _should_search_flash_flood(input_text):
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    prompt = f"""
You have access to live web search.

Task:
Use the user's request to find the specific flash flood event(s), time window, and affected locations.
Summarize relevant news or official sources that confirm the timing and locations.

User input:
{input_text}

Output JSON only, with this structure:
{{
  "event_summary": "short summary of the flash flood event",
  "time_window": "string with start/end time range as found in sources",
  "locations": [
    {{
      "name": "location name",
      "latitude": number or null,
      "longitude": number or null,
      "impact": "short impact summary",
      "sources": ["url"]
    }}
  ],
  "sources": ["url"]
}}
"""
    response = client.responses.create(
        model=model_name,
        input=prompt,
        tools=[{"type": "web_search"}],
    )
    text_output = response.output_text
    try:
        return json.loads(text_output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text_output, re.DOTALL)
        if not match:
            return {"event_summary": text_output.strip(), "time_window": "", "locations": [], "sources": []}
        return json.loads(match.group(0))


def fetch_flash_flood_focus_coords(
    input_text: str,
    basin_name: str,
    event_context: Optional[dict] = None,
) -> Optional[tuple[float, float]]:
    if not _should_search_flash_flood(input_text):
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    context_note = ""
    if event_context:
        context_note = f"\nExisting context:\n{json.dumps(event_context, ensure_ascii=False)}\n"
    prompt = f"""
You have access to live web search.

Task:
Find the flash flood locations mentioned in the user's request and identify the single most impacted
location (city, county, river community, or specific area). Return its latitude and longitude.
Use official sources or news reports to justify the location.

User input:
{input_text}

Parsed basin name (may be wrong):
{basin_name}
{context_note}

Output JSON only:
{{
  "name": "most impacted location name",
  "latitude": number,
  "longitude": number,
  "impact": "short impact summary",
  "sources": ["url"]
}}
"""
    response = client.responses.create(
        model=model_name,
        input=prompt,
        tools=[{"type": "web_search"}],
    )
    text_output = response.output_text
    try:
        result = json.loads(text_output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text_output, re.DOTALL)
        if not match:
            return None
        result = json.loads(match.group(0))
    try:
        return (float(result["latitude"]), float(result["longitude"]))
    except (KeyError, TypeError, ValueError):
        return None
## Agents
# Fixed parse simulation info
def fixed_parse_simulation_info(input_text: str, agents_config: dict, tasks_config: dict):
    search_context = fetch_flash_flood_context(input_text)
    search_context_text = ""
    if search_context:
        search_context_text = (
            "\n\nWeb search findings (flash flood context):\n"
            f"{json.dumps(search_context, ensure_ascii=False)}\n"
        )
    # Create agents
    location_parser = Agent(
        role=agents_config['location_parser_agent']['role'],
        goal=agents_config['location_parser_agent']['goal'],
        backstory=agents_config['location_parser_agent']['backstory'],
        verbose=agents_config['location_parser_agent']['verbose']
    )

    time_parser = Agent(
        role=agents_config['time_parser_agent']['role'],
        goal=agents_config['time_parser_agent']['goal'],
        backstory=agents_config['time_parser_agent']['backstory'],
        verbose=agents_config['time_parser_agent']['verbose']
    )

    # Create tasks with context as a list
    parse_location_description = (
        tasks_config['parse_location']['description'].format(input_text=input_text)
        + search_context_text
    )
    parse_location_task = Task(
        description=parse_location_description,
        expected_output=tasks_config['parse_location']['expected_output'],
        agent=location_parser
    )

    parse_time_description = (
        tasks_config['parse_time_period']['description'].format(input_text=input_text)
        + search_context_text
    )
    parse_time_task = Task(
        description=parse_time_description,
        expected_output=tasks_config['parse_time_period']['expected_output'],
        agent=time_parser
    )

    # Create crew
    crew = Crew(
        agents=[location_parser, time_parser],
        tasks=[parse_location_task, parse_time_task],
        verbose=True
    )

    # Run the crew
    result = crew.kickoff()
    
    # Extract the basin name from the output of the location parsing task, removing any surrounding quotes
    basin_name = parse_location_task.output.raw.strip('"\'')
    # Extract the time period from the output of the time parsing task (as a string, e.g., "[datetime(...)]")
    time_period = parse_time_task.output.raw

    # Return both pieces of information as a dictionary
    return {
        "basin_name": basin_name,
        "time_period": time_period,
        "event_context": search_context
    }

def get_basin_center_coords(
    basin_name,
    input_text,
    agents_config: dict,
    tasks_config: dict,
    event_context: Optional[dict] = None,
):
    focus_coords = fetch_flash_flood_focus_coords(input_text, basin_name, event_context)
    if focus_coords:
        return focus_coords
    basin_center_agent = Agent(
        role=agents_config['basin_center_agent']['role'],
        goal=agents_config['basin_center_agent']['goal'],
        backstory=agents_config['basin_center_agent']['backstory'],
        verbose=agents_config['basin_center_agent']['verbose']
    )
    basin_center_task = Task(
        description=tasks_config['get_basin_center']['description'].format(basin_name=basin_name, input_text=input_text),
        expected_output=tasks_config['get_basin_center']['expected_output'],
        agent=basin_center_agent
    )
    crew = Crew(
        agents=[basin_center_agent],
        tasks=[basin_center_task],
        verbose=True
    )
    crew.kickoff()
    # Extract the output and try to parse the tuple using regex
    import re
    output = basin_center_task.output.raw
    match = re.search(r"\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?", output)
    if match:
        lat = float(match.group(1))
        lon = float(match.group(2))
        return (lat, lon)
    else:
        return output  # fallback: return raw output if parsing fails
