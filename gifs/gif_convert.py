from PIL import Image
import imageio
import os


folder_path = 'gif_wi_4'  


crop_box = (50, 50, 150, 150)

frames = sorted([
    f for f in os.listdir(folder_path)
    if f.endswith('.png')
])

cropped_images = []

for frame in frames:
    img_path = os.path.join(folder_path, frame)
    with Image.open(img_path) as im:
        cropped = im.crop(crop_box)
        cropped_images.append(cropped)

output_gif = os.path.join(folder_path, 'output.gif')
cropped_images[0].save(
    output_gif,
    save_all=True,
    append_images=cropped_images[1:],
    duration=100,  
    loop=0
)

print(f"GIF saved to: {output_gif}")
