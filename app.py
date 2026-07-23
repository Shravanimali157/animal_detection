import pandas as pd 
import numpy as np
import streamlit as st 
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model




interpreter = tf.lite.Interpreter( model_path="animal_final.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()





#model=load_model("animal_final.keras")

st.set_page_config(
    page_title="Animal Recognition",
    page_icon="🐾",
    layout="wide"
)


st.markdown("""
<style>

/* Main background */
.stApp{
    background-color:#F5F7FA;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color:#1E293B;
}

/* Sidebar text */
[data-testid="stSidebar"] *{
    color:white;
}

/* Buttons */
.stButton>button{
    background-color:#2563EB;
    color:white;
    border-radius:10px;
    border:none;
}

.stButton>button:hover{
    background-color:#1D4ED8;
}

/* File uploader */
[data-testid="stFileUploader"]{
    background-color:white;
    border-radius:10px;
    padding:10px;
}

/* Radio buttons */
.stRadio{
    background:white;
    padding:10px;
    border-radius:10px;
}

/* Progress bar */
.stProgress > div > div > div{
    background-color:#22C55E;
}

</style>
""", unsafe_allow_html=True)








st.sidebar.title("🐾 Animal Recognition")
st.sidebar.markdown("---")

st.sidebar.subheader("📖 About")
st.sidebar.write(
    "This application uses a Convolutional Neural Network (CNN) to classify an uploaded image into one of three categories: Cat, Dog, or Wild Animal."
)

st.sidebar.write("The system accepts an image as input and classifies it into one of three predefined classes: Cat, Dog, or Wild Animal. It demonstrates the practical application of CNNs for image classification and can serve as a foundation for more advanced multi-class animal recognition systems.")





if "animal" not in st.session_state:
    st.session_state.predicted_animal = None




st.title("🐾 Animal Recognition Using CNN")

    #st.write("🐾 Animal Recognition Using CNN")

    # Navigation
page = st.radio(
    "Choose a Section",
    ["🐾 Animal Detection", "📚 Animal Information"],
    horizontal=True
)


if page == "🐾 Animal Detection":

    st.write("                                                                                                                                                  ")
    st.write("                                                                                                                                                  ")


    st.markdown(
        "<p style='text-align:center; font-size:18px; color:gray;'>"
        "Classify images as Cat, Dog, or Wild Animal"
        "</p>",
        unsafe_allow_html=True
    )

    st.write("                                                                                                                                                  ")

    
    uploaded_file=st.file_uploader("Upload an image to identify whether it is a **Cat**, **Dog**, or **Wild Animal**.",type=['jpg','jepg','png'])


    if uploaded_file is not None:
        st.image(uploaded_file)
        
        #img = Image.open(uploaded_file)

        img = Image.open(image).convert("RGB")

        img=img.resize((128,128))

        img_array=image.img_to_array(img)/255.0

        mg = img.astype( "float32")/255.0
        
        img_array=np.expand_dims(img_array, axis = 0)

        interpreter.set_tensor(input_details[0]["index"],img)
                    
        interpreter.invoke()

        
        prediction=interpreter.get_tensor(output_details[0]["index"])
        
        class_names = ['Cat', 'Dog','Wild Animal']


        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = prediction[0][predicted_index] * 100

        st.success(f"Prediction: {predicted_class.upper()}")

        st.subheader("Prediction Probabilities")

        st.write(f"**Confidence:** {confidence:.2f}%")

        # Progress bar only for predicted class
        st.progress(float(confidence / 100))
        
        st.subheader("🎯 Features")
        st.write("""
            - 🐱 Detects Cats
            - 🐶 Detects Dogs
            - 🦁 Detects Wild Animals
            - ⚡ Fast AI Prediction
            - 📊 Displays Confidence Score
            """)

        st.subheader("🖼️ Supported Formats")
        st.write("""
            - JPG
            - JPEG
            - PNG
            """) 
        
        st.session_state.animal = predicted_class
        

elif page == "📚 Animal Information":

    st.header("📚 Animal Information")

    if st.session_state.animal is None:
        st.info("⚠️ Please detect an animal first from the Animal Detection section.")


    elif st.session_state.animal == "Cat":

        st.subheader("🐱 Cat")

        st.write("""
                    Cats are small, carnivorous mammals commonly kept as pets.
                    They are known for their agility, curiosity, and independence.
                   
                    Domestic cats (Felis catus) are small, agile, and independent carnivorous mammals. Highly evolved predators, they are famous for their excellent night vision, 
                    retractable claws, and ability to squeeze into tight spaces. As obligate carnivores, they require meat-based diets to survive. 

                    Essential Care Guide for Cats
                    To take proper care of a domestic cat, you must provide a balanced, meat-based diet and constant access to fresh water to support their needs as obligate carnivores. 
                    Maintaining a pristine environment is equally important, which requires scooping their litter box daily and keeping it in a quiet, accessible location. To fulfill their 
                    natural instincts, offer vertical spaces like cat trees, scratching posts to protect your furniture, and daily interactive playtime to keep them mentally and physically sharp. 
                    Regular grooming, including brushing their fur to prevent hairballs and trimming their claws, keeps them comfortable, while keeping them strictly indoors protects them from outside hazards. 
                    Finally, ensure their long-term health by scheduling annual veterinarian checkups for essential vaccinations, spaying or neutering them early, and microchipping them in case they ever escape.

                  """)

    elif st.session_state.animal == "Dog":

        st.subheader("🐶 Dog")

        st.write("""
                    To take proper care of a domestic cat, you must provide a balanced, meat-based diet and constant access to fresh water to support their needs as obligate carnivores. 
                    Maintaining a pristine environment is equally important, which requires scooping their litter box daily and keeping it in a quiet, accessible location. 
                    To fulfill their natural instincts, offer vertical spaces like cat trees, scratching posts to protect your furniture, and daily interactive playtime to keep them mentally and physically sharp. 
                    Regular grooming, including brushing their fur to prevent hairballs and trimming their claws, keeps them comfortable, while keeping them strictly indoors protects them from outside hazards. 
                    Finally, ensure their long-term health by scheduling annual veterinarian checkups for essential vaccinations, spaying or neutering them early, and microchipping them in case they ever escape.
                 
                    Essential Care Guide for Dogs
                    To take proper care of a domestic dog, you must provide a balanced diet tailored to their specific age, size, and activity level, alongside constant access to clean drinking water. 
                    Dogs require daily physical exercise and mental stimulation, which you can easily achieve through regular walks, interactive play like fetch, and positive reinforcement training to ensure good behavior. 
                    Their grooming routine should include brushing their coat a few times a week to control shedding, monthly baths with dog-safe shampoo, and regular nail trims to prevent discomfort. 
                    For their physical safety and community health, it is essential to keep them securely leashed in public, provide them with a collar and microchip for identification, and spay or neuter them. 
                    Finally, maintain their medical well-being by scheduling annual veterinarian checkups for core vaccinations, dental care, and routine preventative treatments for fleas, ticks, and heartworms.
                
                """)

    elif st.session_state.animal == "Wild Animal":

        st.subheader("🦁 Wild Animal")

        st.write("""
                    Wild animals are undomesticated species that live and thrive in natural ecosystems independently of human intervention. Unlike companion pets, they are finely adapted to specific habitats—such as 
                    forests, oceans, deserts, and grasslands—where they must hunt, forage, and protect themselves to survive. Because they are not adapted to live with humans, keeping them as pets is dangerous to 
                    people and harmful to the animal's physical and psychological well-being. True "care" for wild animals does not mean feeding or handling them, but rather respecting their boundaries, preserving their 
                    natural habitats, and observing them safely from a distance to let them remain wild.When humans encounter wild animals in everyday life, care means coexisting responsibly and minimizing human interference. 
                    You can protect local wildlife by securing household trash, keeping domestic pets indoors or leashed, and avoiding the use of harmful chemical pesticides in your yard. If you discover a wild animal that is clearly injured, 
                    sick, or orphaned, the best course of action is to avoid touching it directly and immediately contact a licensed local wildlife rehabilitator or park ranger. Ultimately, the most effective way to care for 
                    wild animals is to support conservation efforts, protect public lands, and ensure their natural environments remain intact and undisturbed
                    """)

