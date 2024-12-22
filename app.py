import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
from serpapi import GoogleSearch

st.set_page_config(page_title="Image Caption & Search", layout="centered")

st.title("Image Caption Generator & Image Search")

image_url = st.text_input("Enter the image URL:")

if image_url:

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


    headers = {
        "User-Agent": "MyBot/1.0 (MyContactInformation@example.com)"
    }

    try:
    
        response = requests.get(image_url, stream=True, headers=headers)
        response.raise_for_status() 

    
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Uploaded Image", use_container_width=True)

    
        inputs = processor(images=img, return_tensors="pt")

    
        outputs = model.generate(**inputs)
        summary = processor.decode(outputs[0], skip_special_tokens=True)

    
        st.write(f"Generated summary: *{summary}*")

    
        query = st.text_input("Enter your search query:")

        if query:
        
            web_query = summary + " " + query

        
            api_key = "c12477161d5d16158a7155444a7a336b557bfc342ab3a148948a841ec468da33"

        
            params = {
                "q": web_query, 
                "tbm": "isch", 
                "api_key": api_key 
            }

        
            search = GoogleSearch(params)

        
            results = search.get_dict()

        
        
            st.subheader("Search Results:")
            if 'images_results' in results:
                images = results['images_results']
                
            
                num_columns = 3
                rows = [images[i:i + num_columns] for i in range(0, len(images), num_columns)]
                
                for row in rows:
                    cols = st.columns(num_columns)
                    for col, image_result in zip(cols, row):
                        with col:
                            st.image(image_result['original'], caption=image_result['title'], use_container_width=True)
                            st.write(f"[Image URL]({image_result['original']})")
            else:
                st.write("No image results found.")


    except Exception as e:
        st.error(f"Error fetching the image: {e}")
else:
    st.write("Please enter an image URL to get started.")
