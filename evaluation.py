from utils import *
from generate_layout_openai import *
from generate_layout_ollama import *
from generate_layout_ollama import generate_layout as ollama_layout
from generate_layout_openai import generate_layout as openai_layout
import pandas as pd
from tqdm import tqdm
import numpy as np

def eval(model,type,prompts,prompt_type):
    names,boxes = [],[]
    total_prompts = len(prompts)
    num_correct_format = 0
    num_valid_layout = 0

    for prompt in tqdm(prompts, total=total_prompts, desc=prompt_type, unit="prompt"):
        if type == "openai":
            names,boxes = openai_layout(prompt,model,eval_mode=True)
        elif type == "ollama":
            names,boxes = ollama_layout(prompt,model,eval_mode=True)
        
        # checking if format is valid -> "four coordinates for each box along with its label"
        correct_format = True

        # first if lists are same size
        if len(names) == len(boxes):
            # then checking if each box has 4 coordinates
            for box in boxes:
                if len(box) != 4:
                    correct_format = False
        else:
            correct_format = False
        
        # if format is still valid then increment correct format counter
        if correct_format:
            num_correct_format += 1
        
        # checking if layout is valid -> given layout (x0,y0,x1,y1), all coordinates must be in range [0,512] and x0<=x1, y0<=y1
        valid_layout = True
        for box in boxes:
            # if any coordinate is not in range layout not valid
            if max(box) > 512 or min(box) < 0:
                valid_layout = False
                
            # if any of the first set of coords is larger than the second set layout not valid
            if box[0] > box[2] or box[1] > box[3]:
                valid_layout = False
    
        if valid_layout:
            num_valid_layout += 1
    
    format_accuracy = num_correct_format/total_prompts
    valid_accuracy = num_valid_layout/total_prompts

    return format_accuracy,valid_accuracy,total_prompts


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="The ollama model to use"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="openai",
        help="what type of model: openai or ollama"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to sample from dataset"
    )

    args = parser.parse_args()
    model = args.model
    type = args.type
    samples = args.samples
    seed = 42

    # using pandas to read each dataset and sampling random 50 prompts from each (default is 50 each can change with samples)
    colors = pd.read_csv("./prompts/colors_composition_prompts.csv").sample(n=samples,random_state=seed)
    counting = pd.read_csv("./prompts/counting_prompts.csv").sample(n=samples,random_state=seed)
    size = pd.read_csv("./prompts/size_compositions_prompts.csv").sample(n=samples,random_state=seed)
    spatial = pd.read_csv("./prompts/spatial_compositions_prompts.csv").sample(n=samples,random_state=seed)

    # Convert the correct prompt columns to a list (using synthetic prompts for counting)
    color_prompts = colors['meta_prompt'].tolist()
    counting_prompts = counting['synthetic_prompt'].tolist()     
    size_prompts = size['meta_prompt'].tolist()     
    spatial_prompts = spatial['meta_prompt'].tolist()     

    # Creating dictionary of prompt lists for evaluation    
    prompts = {
        "Colors & Composition": color_prompts,
        "Counting": counting_prompts,
        "Size Compositions": size_prompts,
        "Spatial Compositions": spatial_prompts
    }

    overall_format_accuracy = 0.0
    overall_valid_accuracy = 0.0
    total_samples = 0
    overall_start_time = time.time()

    # Evaluating each prompt set
    for prompt_type,prompt_list in prompts.items():
        start_time = time.time()
        
        # evaluating the specified model
        format_accuracy,valid_accuracy,total_prompts = eval(model,type,prompt_list,prompt_type)

        end_time = time.time()
        runtime = end_time - start_time

        # keeping track of overall accuracy
        overall_format_accuracy += format_accuracy
        overall_valid_accuracy += valid_accuracy
        total_samples += len(prompt_list)
        
        # printing out the results
        print(f"\n-----Results for {prompt_type} Prompt Set from HRS-----")
        print(f"\nRuntime: {runtime:.2f} seconds")
        print(f"Format Accuracy: {format_accuracy * 100:.2f}%")
        print(f"Validness Accuracy: {valid_accuracy * 100:.2f}%")
    
    overall_format_accuracy/=4
    overall_valid_accuracy/=4
    overall_end_time = time.time()
    overall_runtime = overall_end_time-overall_start_time

    print(f"\n-----OVERALL RESULTS-----\n")
    print(f"Overall Runtime: {runtime:.2f} seconds")
    print(f"Overall Format Accuracy: {overall_format_accuracy * 100:.2f}%")
    print(f"Overall Validness Accuracy: {overall_valid_accuracy * 100:.2f}%")
