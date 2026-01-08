# Disable OpenTelemetry warnings and tracing
import os, re
os.environ["OTEL_PYTHON_DISABLED"] = "true"
os.environ["OTEL_PYTHON_TRACER_PROVIDER"] = "none"
import pypandoc

import yaml
from crewai import Agent, Task, Crew
import warnings
from opentelemetry import trace

# Disable the TracerProvider warning by setting the environment variable
os.environ["OTEL_PYTHON_TRACER_PROVIDER"] = "none"

# Feedback control
class DirtyFlags:
    def __init__(self):
        self.gauge_id = False
        self.crest_args = False

import json

from tools.prompt_loader import load_prompts

def feedback_agent(feedback: str):
    prompts = load_prompts()
    feedback_prompts = prompts["feedback_agent"]
    feedback_parser_agent = Agent(
        role=feedback_prompts["role"],
        goal=feedback_prompts["goal"],
        backstory=feedback_prompts["backstory"],
        verbose=False  # Set to True to view LLM reasoning logs
    )

    # Extensible CREST parameter set (converted to lowercase for matching)
    CREST_PARAMS = {
        "alpha", "alpha0", "b", "beta", "fc", "grid_on", "im", "isu", "iwu",
        "ke", "leaki", "th", "under", "wm"
    }
    
    # Let Agent parse feedback and output JSON ------------------
    feedback_task = Task(
        description=feedback_prompts["task_description"].format(
            feedback=feedback,
            crest_params=sorted(CREST_PARAMS),
        ),
        expected_output=feedback_prompts["expected_output"],
        agent=feedback_parser_agent
    )
    crew = Crew(agents=[feedback_parser_agent], tasks=[feedback_task], verbose=False)
    crew.kickoff()

    # Parse Agent output ------------------------------------
    raw = feedback_task.output.raw.strip()
    
    # Try multiple cleaning strategies to extract valid JSON
    json_text = None
    
    # Strategy 1: Look for JSON object in the raw output
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, flags=re.S)
    if json_match:
        json_text = json_match.group(0)
    
    # Strategy 2: Clean markdown code blocks
    if not json_text:
        clean = raw
        clean = re.sub(r'```(?:json)?\s*', '', clean, flags=re.I)  # Remove ```json or ```
        clean = re.sub(r'```\s*$', '', clean)  # Remove trailing ```
        clean = clean.strip()
        
        # Look for JSON object again
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', clean, flags=re.S)
        if json_match:
            json_text = json_match.group(0)
    
    # Strategy 3: If still no JSON found, try the entire cleaned text
    if not json_text:
        json_text = clean
    
    # Try to parse JSON with multiple fallback strategies
    result = None
    
    try:
        result = json.loads(json_text)
    except json.JSONDecodeError:
        try:
            # Try fixing common JSON issues
            fixed_json = json_text.replace("'", '"')  # Replace single quotes with double quotes
            fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)  # Quote unquoted keys
            result = json.loads(fixed_json)
        except json.JSONDecodeError:
            try:
                # Try extracting just the content between braces
                brace_content = re.search(r'\{(.*)\}', json_text, flags=re.S)
                if brace_content:
                    content = brace_content.group(1).strip()
                    # Build a minimal valid JSON
                    result = {
                        "gauge_id_dirty": False,
                        "gauge_id_new": None,
                        "crest_args_dirty": False,
                        "crest_args_new": {},
                        "explanation": f"Parsed from content: {content[:100]}..."
                    }
                else:
                    raise json.JSONDecodeError("No valid JSON structure found", json_text, 0)
            except:
                # Final fallback: return default structure
                print(f"Warning: All JSON parsing strategies failed. Using default structure.\nRaw output: {raw}")
                result = {
                    "gauge_id_dirty": False,
                    "gauge_id_new": None,
                    "crest_args_dirty": False,
                    "crest_args_new": {},
                    "explanation": "Failed to parse feedback, no changes applied."
                }

    # Ensure result is a dictionary before using setdefault
    if not isinstance(result, dict):
        print(f"Warning: Result is not a dictionary, got {type(result)}. Using default structure.\nResult: {result}")
        result = {
            "gauge_id_dirty": False,
            "gauge_id_new": None,
            "crest_args_dirty": False,
            "crest_args_new": {},
            "explanation": "Failed to parse feedback, no changes applied."
        }

    result.setdefault("gauge_id_dirty", False)
    result.setdefault("gauge_id_new", None)
    result.setdefault("crest_args_dirty", False)
    result.setdefault("crest_args_new", {})
    result.setdefault("explanation", "No changes detected.")
    
    # Assemble executable code string -------------------------------
    code_lines = [
        "",

    ]
    if result["gauge_id_dirty"]:
        code_lines += [
            "dirty.gauge_id = True",
            f"args_new.gauge_id = '{result['gauge_id_new']}'",
        ]
    if result["crest_args_dirty"]:
        code_lines.append("dirty.crest_args = True")
        for k, v in result["crest_args_new"].items():
            # Keep Python syntax for bool values
            val_repr = str(v).lower() if isinstance(v, bool) else v
            code_lines.append(f"crest_args_new.{k.lower()} = {val_repr}")

    code_str = "\n".join(code_lines)
    explanation_str = result["explanation"]

    return code_str, explanation_str


# Main
def aquah_run(cli_args):
    # Warning control
    warnings.filterwarnings('ignore')
    import logging
    logging.getLogger("opentelemetry.trace").setLevel(logging.ERROR)

    
    # os.environ['OPENAI_MODEL_NAME'] = llm_model_name
    # os.environ['OPENAI_API_KEY'] = gpt_key
    

    # Define file paths for YAML configurations
    files = {
        'agents': 'config/agents.yaml',
        'tasks': 'config/tasks.yaml'
    }
    # Load configurations from YAML files
    configs = {}
    for config_type, file_path in files.items():
        with open(file_path, 'r') as file:
            configs[config_type] = yaml.safe_load(file)

    # Assign loaded configurations to specific variables
    agents_config = configs['agents']
    tasks_config = configs['tasks']
    input_text = cli_args.input_text
    if not input_text:
        input_text = input("Please enter the simulation information (e.g., 'I want to simulate basin Fort Cobb, from 2022 June to July'): ")
    print('User input: ', input_text)
    
    print('\n\033[1;31m\033[1m------------------------------------------------')
    print('Step 1: Determine Location and Time Period')
    print('------------------------------------------------\033[0m\033[0m\n')
    from tools.agent_time_location_parser import fixed_parse_simulation_info, get_basin_center_coords
    try:
        result = fixed_parse_simulation_info(input_text, agents_config, tasks_config)
        # print(result)
    except Exception as e:
        print(f"Error: {e}")
        
    print('\n\033[1;31m\033[1m------------------------------------------------')
    print('Step 2: Find the Basin')
    print('------------------------------------------------\033[0m\033[0m\n')
    basin_name = result["basin_name"]
    center_coords = get_basin_center_coords(basin_name, input_text, agents_config, tasks_config)
    # print(f"Basin '{basin_name}' center coordinates: {center_coords}")

    # Construct the args
    from types import SimpleNamespace
    from datetime import datetime
    import re

    args = SimpleNamespace()
    args.llm_model_name = cli_args.llm_model_name
    args.vision_model_name = cli_args.vision_model_name
    args.basin_name = result["basin_name"]

    # Helper to parse a string like "[datetime(2022, 6, 1), datetime(2022, 7, 1)]"
    def parse_time_period_string(s):
        # Find all datetime(...) patterns
        dt_matches = re.findall(r"datetime\((.*?)\)", s)
        if len(dt_matches) != 2:
            raise ValueError("Could not parse two datetime objects from time_period string.")
        dt_objs = []
        for dt_str in dt_matches:
            # Split by comma, strip, and convert to int
            parts = [int(x.strip()) for x in dt_str.split(",")]
            dt_objs.append(datetime(*parts))
        return dt_objs

    time_period = result["time_period"]
    if isinstance(time_period, str):
        try:
            time_period = parse_time_period_string(time_period)
        except Exception as e:
            raise ValueError(f"Failed to parse time_period string: {e}")

    args.time_start = time_period[0]
    args.time_end = time_period[1]
    args.selected_point = center_coords
    # default args
    args.basin_shp_path = cli_args.basin_shp_path
    args.basin_level = cli_args.basin_level
    args.gauge_meta_path = cli_args.gauge_meta_path
    args.figure_path = cli_args.figure_path
    args.basic_data_path = cli_args.basic_data_path
    args.basic_data_clip_path = cli_args.basic_data_clip_path
    args.usgs_data_path = cli_args.usgs_data_path
    args.mrms_data_path = cli_args.mrms_data_path
    args.crest_input_mrms_path = cli_args.crest_input_mrms_path
    args.mrms_2min_data_path = cli_args.mrms_2min_data_path
    args.crest_input_mrms_2min_path = cli_args.crest_input_mrms_2min_path
    args.num_processes = cli_args.num_processes
    args.pet_data_path = cli_args.pet_data_path
    args.crest_input_pet_path = cli_args.crest_input_pet_path
    args.crest_output_path = cli_args.crest_output_path
    args.control_file_path = cli_args.control_file_path
    args.report_path = cli_args.report_path
    args.time_step = cli_args.time_step
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.figure_path = os.path.join(args.figure_path, current_time)
    args.input_text = input_text
    args.water_balance_type = cli_args.water_balance_type
    args.warmup_flag = cli_args.warmup_flag
    args.skip_download = cli_args.skip_download
    args.skip_basic_data = cli_args.skip_basic_data
    args.warmup_time_step = cli_args.warmup_time_step
    args.warmup_days = cli_args.warmup_days
    args.warmup_time_end = args.time_start
    args.warmup_state_folder = os.path.join('warmup_state', current_time)
    if args.warmup_flag:
        # Create warmup state folder if it doesn't exist
        if not os.path.exists(args.warmup_state_folder):
            os.makedirs(args.warmup_state_folder)
        print(f"Warmup is enabled, {args.warmup_days} days, created warmup state folder: {args.warmup_state_folder}")
        from datetime import datetime, timedelta
        args.warmup_time_start = args.time_start - timedelta(days=args.warmup_days)
    else:
        args.warmup_time_start = args.time_start
    
    

    def run_basin_processing():
        import importlib
        basin_processor_module = importlib.import_module("tools.basin_processor")
        importlib.reload(basin_processor_module)
        basin_processor = basin_processor_module.basin_processor
        Basin_Area, Basin_Name, gauges_list, gauges_description = basin_processor(args)
        args.basin_area = Basin_Area
        args.basin_name = Basin_Name
        args.gauges_list = gauges_list
        args.gauges_description = gauges_description
        print(args.gauges_description)

    # Basin shp and basic data download
    run_basin_processing()

    print('\n\033[1;31m\033[1m--------------------------------------------------------')
    print('Step 2.5: Verify Basin Map Against User Request')
    print('--------------------------------------------------------\033[0m\033[0m\n')
    from tools.agent_basin_map_verifier import verify_basin_map_by_image
    basin_map_path = os.path.join(args.figure_path, "basin_map_with_gauges.png")
    verifier_model = args.vision_model_name or args.llm_model_name
    decision, reason = verify_basin_map_by_image(
        input_text=args.input_text,
        image_path=basin_map_path,
        model_name=verifier_model,
        temperature=0.2,
    )
    decision = decision.strip().lower()
    print(f"Basin verification decision: {decision}")
    print(f"Basin verification reason: {reason}")
    if decision == "expand" and args.basin_level != 4:
        print(f"Expanding basin level from {args.basin_level} to 4 (HUC8) and rebuilding basin data.")
        args.basin_level = 4
        run_basin_processing()
    basin_name_current_time = args.basin_name + '_' + current_time
    basin_name_current_time = basin_name_current_time.replace(' ', '_')
    args.crest_output_path = os.path.join(args.crest_output_path, basin_name_current_time)
    args.report_path = os.path.join(args.report_path, basin_name_current_time)
    
    # return

    print('\n\033[1;31m\033[1m--------------------------------------------------------')
    print('Step 3: Find the Outlet Gauge Representing the Basin')
    print('--------------------------------------------------------\033[0m\033[0m\n')

    from tools.agent_select_outlet_gauge import select_outlet_gauge
    args.gauge_id, args.why_select_outlet_gauge = select_outlet_gauge(args)
    print("Selected Outlet Gauge ID:", args.gauge_id)
    print("Why Selected Outlet Gauge:\n", args.why_select_outlet_gauge)

    import importlib
    import tools.gauge_processor
    importlib.reload(tools.gauge_processor)
    args.latitude_gauge, args.longitude_gauge = tools.gauge_processor.gauge_processor(args)


    import pickle
    args_output_path = os.path.join(args.figure_path, f'simulation_args.pkl')
    with open(args_output_path, 'wb') as f:
        pickle.dump({'args': args}, f)
    print(f"Saved simulation arguments to: {args_output_path}")
    # return

    print('\n\033[1;31m\033[1m--------------------------------------------------------')
    print('Step 4: Download Precipitation and Potential Evapotranspiration Data')
    print('--------------------------------------------------------\033[0m\033[0m\n')
    # Input data download
    if cli_args.skip_download:
        print("Skipping data download because --skip_download is enabled.")
    else:
        import importlib
        import tools.precipitation_processor
        importlib.reload(tools.precipitation_processor)
        tools.precipitation_processor.precipitation_processor(args)

        import importlib
        import tools.pet_processor
        importlib.reload(tools.pet_processor)
        tools.pet_processor.pet_processor(args)

    # print('\n\033[1;31m\033[1m--------------------------------------------------------')
    # print('Step 5: Get Initial Parameters and Run CREST')
    # print('--------------------------------------------------------\033[0m\033[0m\n')


    # import importlib
    # import tools.agent_parameter_initial_guess
    # importlib.reload(tools.agent_parameter_initial_guess)
    # crest_args, why_parameter_initial_guess = tools.agent_parameter_initial_guess.get_initial_crest_args(args)
    # crest_args.grid_on = False
    # crest_args.grid_on = True
    # args.grid_on = crest_args.grid_on
    # args.why_parameter_initial_guess = why_parameter_initial_guess

    # import importlib
    # import tools.crest_run
    # importlib.reload(tools.crest_run)
    # # # Override crest_args with specified parameters
    # # crest_args.wm = 180
    # # crest_args.b = 2.5
    # # crest_args.im = 0.03
    # # crest_args.ke = 0.7
    # # crest_args.fc = 80
    # # crest_args.iwu = 25
    # # crest_args.th = 50
    # # crest_args.under = 1.5
    # # crest_args.leaki = 0.1
    # # crest_args.isu = 0
    # # crest_args.alpha = 1.2
    # # crest_args.beta = 0.6
    # # crest_args.alpha0 = 0.8
    # print("Using manually specified CREST parameters")

    # os.makedirs(args.crest_output_path, exist_ok=True)
    # args_output_path = os.path.join(args.crest_output_path, f'simulation_args_initial.pkl')
    # with open(args_output_path, 'wb') as f:
    #     pickle.dump({'args': args, 'crest_args': crest_args}, f)
    # print(f"Saved initial simulation arguments to: {args_output_path}")
    # tools.crest_run.crest_run(args, crest_args)
    
    # default_flag = False
    # # default_flag = True
    # if default_flag:
    #     tools.crest_run.crest_run_default(args)
    print('\n\033[1;31m\033[1m--------------------------------------------------------')
    print('Step 5: Run CREST')
    print('--------------------------------------------------------\033[0m\033[0m\n')

    import importlib
    import tools.crest_run
    importlib.reload(tools.crest_run)
    tools.crest_run.crest_run_cali(args)
    

    return
    print('\n\033[1;31m\033[1m--------------------------------------------------------')
    print('Step 6: Generate a Report')
    print('--------------------------------------------------------\033[0m\033[0m\n')
    
    import importlib
    import tools.agent_report_writer
    importlib.reload(tools.agent_report_writer)
    tools.agent_report_writer.final_report_writer(args, crest_args, agents_config, tasks_config, iteration_num=0)


    # Save simulation arguments to a pickle file for future reference
    import pickle
    
    iteration_num = 0
    args_output_path = os.path.join(args.crest_output_path, f'simulation_args_{iteration_num}.pkl')
    with open(args_output_path, 'wb') as f:
        pickle.dump({'args': args, 'crest_args': crest_args}, f)
    print(f"Saved simulation arguments to: {args_output_path}")
    
    
    # feedback module from user
    while True:
        iteration_num += 1
        print('This is iteration: ', iteration_num)
        print('Please enter the feedback for the simulation: ')
        feedback = input("Please enter feedback on the results, or 'q' to quit: ")
        
        print('User feedback: ', feedback)
        if feedback == 'q' or feedback == 'Q' or feedback == 'quit' or feedback == 'Quit' or feedback == 'exit' or feedback == 'Exit':
            break
        if feedback == 'r' or feedback == 'R':
            import importlib
            import tools.agent_parameter_initial_guess
            importlib.reload(tools.agent_parameter_initial_guess)
            crest_args, why_parameter_initial_guess = tools.agent_parameter_initial_guess.get_initial_crest_args(args)
            crest_args.grid_on = False
            args.why_parameter_initial_guess = why_parameter_initial_guess
                    
            import importlib
            import tools.crest_run
            importlib.reload(tools.crest_run)
            tools.crest_run.crest_run(args, crest_args)

            import importlib
            import tools.agent_report_writer
            importlib.reload(tools.agent_report_writer)
            tools.agent_report_writer.final_report_writer(args, crest_args, agents_config, tasks_config, iteration_num)

            # Save simulation arguments to a pickle file for future reference
            import pickle
            args_output_path = os.path.join(args.crest_output_path, f'simulation_args_{iteration_num}.pkl')
            with open(args_output_path, 'wb') as f:
                pickle.dump({'args': args, 'crest_args': crest_args}, f)
            print(f"Saved simulation arguments to: {args_output_path}")


            continue
        # Initialize dirty flags
        dirty = DirtyFlags()
        code_snippet, explain = feedback_agent(feedback)

        print('Code snippet: ', code_snippet)
        print('Explain: ', explain)
        import copy
        # Copy args to args_new and crest_args to crest_args_new
        args_new = copy.deepcopy(args)
        crest_args_new = copy.deepcopy(crest_args)
        # Execute the generated code snippet
        exec(code_snippet)


        if dirty.gauge_id:
            args_new.latitude_gauge, args_new.longitude_gauge = tools.gauge_processor.gauge_processor(args_new)

        tools.crest_run.crest_run(args_new, crest_args_new)

        import importlib
        import tools.agent_report_writer
        importlib.reload(tools.agent_report_writer)
        tools.agent_report_writer.final_report_writer(args_new, crest_args_new, agents_config, tasks_config, iteration_num)

        # Save simulation arguments to a pickle file for future reference
        # iteration_num = 0
        args_output_path = os.path.join(args.crest_output_path, f'simulation_args_{iteration_num}.pkl')
        with open(args_output_path, 'wb') as f:
            pickle.dump({'args': args, 'crest_args': crest_args}, f)
        print(f"Saved simulation arguments to: {args_output_path}")
        args = copy.deepcopy(args_new)
        crest_args = copy.deepcopy(crest_args_new)


