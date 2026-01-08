from importlib import reload
import argparse
import getpass
import os

import tools.aquah_run
reload(tools.aquah_run)
from tools.aquah_run import aquah_run

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

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("OpenAI API Key not found in environment variables.")
    api_key = getpass.getpass("Please enter your OpenAI API Key (input will be hidden): ")

if not api_key:
    raise ValueError("No API Key provided. Exiting.")
os.environ["OPENAI_API_KEY"] = api_key
aquah_run(args)
