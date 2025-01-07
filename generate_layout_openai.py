from pydantic import BaseModel
from openai import OpenAI
import argparse
import time

from utils import *

def generate_layout(description,model,eval_mode=False):
  client = OpenAI()
  # a single object (using tuple)
  # class Object(BaseModel):
  #     name: str
  #     box_coordinates: tuple[int,int,int,int]

  # using separate fields
  class Object(BaseModel):
      name: str
      x0: int
      y0: int
      x1: int
      y1: int

  # overall layout structure
  class ObjectLayout(BaseModel):
      objects: list[Object]

  # the instruction prompt
  instruction = "Provide box coordinates for an image with "
  # instruction = "Provide box coordinates (x0,y0,x1,y1) for an image with "

  # setting up messages following the example json files in the attention refocusing repo but without in-context example (modified for gpt-4o)
  messages=[
      {
        "role": "system",
        "content": "You are gpt-4o, a large language model trained by OpenAI. Your goal is to assist users by providing helpful and relevant information. In this context, you are expected to generate specific coordinate box locations for objects in a description, considering their relative sizes and positions and the numbers of objects.Size of image is 512*512"
      },
  ]

  # appending the message
  message = instruction + description
  messages.append(
      {"role": "user", "content": message},
  )

  completion = client.beta.chat.completions.parse(
    model=model,
    messages=messages,
    response_format=ObjectLayout,
  )

  layout = completion.choices[0].message

  # If the model refuses to respond, you will get a refusal message
  if (layout.refusal):
    print(layout.refusal)

  # parse the layout (auto parsing from openai)
  layout = layout.parsed

  # printing out the description and model output if not in eval mode
  if not eval_mode:
    print("\n-----Image Description-----\n")
    print(description)
    print("\n-----Model Output-----\n")
    print(layout)


  # returning the names of objects and their corresponding bounding boxes
  object_names = []
  object_boxes = []

  # iterate through object list of the returned layout
  for object in layout.objects:
    # appending object name
    object_names.append(object.name)

    # creating box from coordinates and appending
    box = [object.x0,object.y0,object.x1,object.y1]
    object_boxes.append(box)
  
  return object_names, object_boxes


# running the script
if __name__ == "__main__":
  # getting command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--model",
      type=str,
      default="llama3",
      help="The ollama model to use"
  )

  args = parser.parse_args()
  model = args.model

  # getting inputs
  description = input("Please your describe image: ")
  image_name = input("Enter a name for your image to save the layout: ")

  start_time = time.time()

  # getting layout from gpt-4o
  names,boxes = generate_layout(description,model)

  end_time = time.time()
  runtime = end_time - start_time
  print(f"\nRuntime: {runtime:.2f} seconds")

  output_folder = "./outputs"
  draw_box(names,boxes,output_folder,image_name+".jpg")
