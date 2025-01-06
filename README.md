# LLM Spatial Layout

Using LLM structured output capabilities to generate reliable spatial layouts from image descriptions. An extension of the GPT4 based box layout generation from the amazing [Grounded Text-to-Image Synthesis with Attention Refocusing](https://attention-refocusing.github.io/) paper.

## Approach & Motivation

In the original Attention Refocusing paper, layouts in the form of bounding boxes were generated by prompting GPT4. This was implemented through using in-context examples to ask the model to generate box coordinates for each object in a description of an image.

As mentioned in the limitations section of the paper, **GPT4 sometimes produces invalid layouts or very small bounding boxes.** This is because at the time the paper was written, structured output capability of models was not yet refined or very robust.
Looking at the code, despite extensive prompting and in-context examples, **the model isn't actually strictly enforced to follow a specific format in its output**, leading to unexpected results at times.

To mitigate this issue, I re-implemented the layout generation scripts using both [OpenAI's structured outputs beta](https://platform.openai.com/docs/guides/structured-outputs?context=ex1#how-to-use) as well as [Ollama structured outputs](https://ollama.com/blog/structured-outputs) to enable use of open-source models.

### Improvements

* Almost always ensures consistent output format to allow for **reliable layout generation**
* **Simplifies prompting and code** in general, reduces the need for extensive in-context examples to enforce output structure
* Use of **open-source models** makes the attention refocusing method more accessible (free and doesn't require API subscription) to allow more users to experiment locally

## Running the code

### Setup

Create a conda environment:
    
    conda create -n "llm-layout" python=3.13

Install the required packages
    
    pip install -r requirements.txt

If using Ollama, first check that your Ollama version is >= 0.5.1 because structured outputs are only available in newer versions
    
    ollama --version

If your version is older, try to upgrade your version as specified here: [Ollama docs](https://github.com/ollama/ollama/blob/main/docs/faq.md). I had to manually uninstall and re-install a new version (0.5.4) from the Ollama website.

Finally, if using Ollama be sure to pull the models you want to use first before running the scripts.

    ollama pull [model name]

I used the following for my short experiment:
* llama3:8b
* llama3.1:8b
* qwen2.5:7b


### Generating layouts

Both scripts (OpenAI and Ollama) run the same, simply specify the model as a command line argument. Example here shown for qwen2.5:7b through ollama.

        python3 generate_layout_ollama.py --model qwen2.5:7b
    
When using the OpenAI script **please only use the following models** as specified in the [OpenAI API Docs](https://platform.openai.com/docs/guides/structured-outputs?context=ex1#how-to-use)
* o1-2024-12-17 and later
* gpt-4o-mini-2024-07-18 and later
* gpt-4o-2024-08-06 and later

## Results




## Conclusions & Further Work

