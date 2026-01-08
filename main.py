from importlib import reload
import argparse
import getpass
import os

import tools.aquah_run
reload(tools.aquah_run)
from tools.aquah_run import aquah_run

def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--llm_model_name", default="gpt-4o")
    parser.add_argument("--vision_model_name", default=None)
    parser.add_argument("--input_text", default=None)
    parser.add_argument("--basin_shp_path", default="shpFile/Basin_selected.shp")
    parser.add_argument("--basin_level", type=int, default=5)
    parser.add_argument("--gauge_meta_path", default="EF5_tools/gauge_meta.csv")
    parser.add_argument("--figure_path", default="figures")
    parser.add_argument("--basic_data_path", default="BasicData")
    parser.add_argument("--basic_data_clip_path", default="BasicData_Clip")
    parser.add_argument("--usgs_data_path", default="USGS_gauge")
    parser.add_argument("--mrms_data_path", default="MRMS_data")
    parser.add_argument("--crest_input_mrms_path", default="CREST_input/MRMS/")
    parser.add_argument("--mrms_2min_data_path", default="MRMS_2min_data")
    parser.add_argument("--crest_input_mrms_2min_path", default="CREST_input/MRMS_2min/")
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--pet_data_path", default="PET_data")
    parser.add_argument("--crest_input_pet_path", default="CREST_input/PET/")
    parser.add_argument("--crest_output_path", default="CREST_output")
    parser.add_argument("--control_file_path", default="control.txt")
    parser.add_argument("--report_path", default="report")
    parser.add_argument("--time_step", default="2u")
    parser.add_argument("--warmup_time_step", default="1h")
    parser.add_argument("--water_balance_type", default="crest")
    parser.add_argument("--warmup_flag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup_days", type=int, default=30)
    parser.add_argument("--skip_download", action="store_true", default=False)
    parser.add_argument("--skip_basic_data", action="store_true", default=False)
    return parser


args = build_parser().parse_args()

load_env_file()

api_keys = [
    ("OPENAI_API_KEY", "Please enter your OpenAI API Key (input will be hidden): "),
    ("ANTHROPIC_API_KEY", "Please enter your Anthropic API Key (input will be hidden): "),
    ("GOOGLE_API_KEY", "Please enter your Google API Key (input will be hidden): "),
    ("DEEPSEEK_API_KEY", "Please enter your DeepSeek API Key (input will be hidden): "),
]

for key_name, prompt in api_keys:
    if not os.getenv(key_name):
        print(f"{key_name} not found in environment variables.")
        value = getpass.getpass(prompt)
        if value:
            os.environ[key_name] = value

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No API Key provided. Exiting.")
aquah_run(args)
