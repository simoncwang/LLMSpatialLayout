from PIL import Image, ImageDraw, ImageFont
import os

def draw_box(text, boxes, output_folder, img_name):
    width, height = 512, 512
    image = Image.new('RGB', (width, height), 'gray')
    
    draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype("Roboto-LightItalic.ttf", size=20)
    for i, box in enumerate(boxes):
        t = text[i]
        draw.rectangle([(box[0], box[1]),(box[2], box[3])], outline=128, width=2)
        mean_box_x, mean_box_y = (
            int((box[0] + box[2]) / 2),
            int((box[1] + box[3]) / 2)
        )
        draw.text((mean_box_x, mean_box_y), t, fill=200)
    image.save(os.path.join(output_folder, img_name))