import openai
import configparser
import pandas as pd
import requests


# function openai_api_client returns Open AI API client
def openai_api_client():
    config = configparser.ConfigParser() # reading data from config file
    config.read("config.ini")
    api_key = config['openai-api']['key']

    client = openai.OpenAI(api_key=api_key)
    return client

client = openai_api_client()

golf_courses_df = pd.read_csv("Golf AI Avatars Project - courses.csv") # Loading DataFrame with Golf Courses in North Carolina. Df has data about Golf Course Name, and Image of Golf Course

for index, row in golf_courses_df.iterrows(): # Looping through all golf courses from df, and generating AI avatars based on provided data
    # Using Vision Model and asking it to describe what's in image
    response_about_image = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": row['Image'],
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    about_image = str(response_about_image.choices[0].message.content) # storing response in about_image variable


    # Asking Dall-e-3 Model to generate avatar on described golf course based on Golf Course Name(provided from df), and based on image(provided description from Open AI Vision Model)
    response = client.images.generate(
      model="dall-e-3",
      prompt=f"How does AI imagine Human Golf Male Avatar on{row['Course Name']} Golf Course. More details about image : {about_image}. And it is located in North Carolina.",
      size="1024x1024",
      quality="hd",
      n=1,
    )
    image_url = response.data[0].url # storing URL of image

    img_response = requests.get(image_url) # Downloading image


    if img_response.status_code == 200: # if download was success, write it to .jpg file
        with open(f'imgs/{row["Course Name"]}.jpg','wb') as f:
            f.write(img_response.content)
        print('Image downloaded succesfully.')
    else:
        print(f'Failed to download image : {row["Course Name"]}')
    print(image_url)
