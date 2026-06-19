import streamlit as st
import cv2
import tempfile
import time
import os
import pandas as pd
from PIL import Image, ImageDraw
from streamlit_option_menu import option_menu

import face_utils # Our clean backend file

st.set_page_config(page_title="Face Attendance System", layout="wide")

# Load model using Streamlit cache so it doesn't reload on every button click
@st.cache_resource
def get_model():
    return face_utils.load_model()

model_data = get_model()

st.title("📸 Automated Attendance System")

# -----------------------------------------
# Sidebar Navigation (Upgraded UI)
# -----------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100) # Optional logo
    st.markdown("### Admin Panel")
    
    choice = option_menu(
        menu_title=None, 
        options=["Dashboard", "Video Attendance", "Live Camera Attendance", "Add New Student"],
        icons=["house", "camera-video", "camera", "person-plus"], 
        menu_icon="cast", 
        default_index=0, 
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#ebebeb", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "16px", 
                "text-align": "left", 
                "margin": "0px", 
                "--hover-color": "#333333" if st.get_option("theme.base") == "dark" else "#f0f2f6"
            },
            "nav-link-selected": {"background-color": "#FF4B4B", "color": "white", "icon-color": "white"},
        }
    )

# -----------------------------------------
# 1. Dashboard
# -----------------------------------------
if choice == "Dashboard":
    st.header("System Overview")
    if model_data:
        st.success("✅ Model is loaded and active.")
        st.subheader("Currently Trained Students:")
        st.write(", ".join(model_data['target_names']))
    else:
        st.warning("⚠️ No trained model found. Please add students and train the system.")

# -----------------------------------------
# 2. Add New Student (Training)
# -----------------------------------------
elif choice == "Add New Student":
    st.header("Add New Student")
    student_name = st.text_input("Enter Student Name (lowercase, no spaces):").strip()
    uploaded_files = st.file_uploader("Upload photos of the student", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

    if st.button("Save & Train Model"):
        if not student_name or not uploaded_files:
            st.error("Please provide a name and at least one photo.")
        else:
            student_dir = os.path.join('./labels', student_name)
            os.makedirs(student_dir, exist_ok=True)
            
            for i, file in enumerate(uploaded_files):
                with open(os.path.join(student_dir, f"{student_name}_{i}.jpg"), "wb") as f:
                    f.write(file.getbuffer())
            
            st.info("Photos saved. Extracting embeddings and training... This may take a minute.")
            
            with st.spinner("Training in progress..."):
                success, msg = face_utils.train_system()
            
            if success:
                st.success(msg)
                get_model.clear() # Clear cache to load the newly trained model
                st.rerun() 
            else:
                st.error(msg)

# -----------------------------------------
# 3. Video Attendance
# -----------------------------------------
elif choice == "Video Attendance":
    st.header("Process Video for Attendance")
    if not model_data:
        st.error("Please train the model first!")
    else:
        uploaded_video = st.file_uploader("Upload Class Video", type=['mp4', 'mkv', 'avi'])
        
        if uploaded_video and st.button("Process Video"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            st.write(f"**Total Frames:** {total_frames} | **FPS:** {fps}")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            recognition_counts = {}
            frame_count = 0
            frame_skip = 3
            start_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                
                if frame_count % 10 == 0: 
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    status_text.text(f"Processing frame {frame_count}/{total_frames}...")

                if frame_count % frame_skip == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    
                    results = face_utils.predict_face(pil_img, model_data['classifier'], model_data['target_names'])
                    
                    for res in results:
                        name = res['name']
                        recognition_counts[name] = recognition_counts.get(name, 0) + 1

            cap.release()
            end_time = time.time()
            
            progress_bar.progress(1.0)
            status_text.text(f"Processing Complete! Took {round(end_time - start_time, 2)} seconds.")
            
            st.subheader("Attendance Results:")
            present_students = []
            for student, count in recognition_counts.items():
                if count >= 150:
                    present_students.append({'Name': student, 'Status': 'Present'})
                    st.success(f"✔️ {student} (Seen {count} times)")
                else:
                    st.warning(f"❌ {student} ignored (Seen only {count} times, need 15)")
            
            if present_students:
                df = pd.DataFrame(present_students)
                df.to_csv('./attendance_sheet.csv', index=False)
                st.download_button("Download CSV", df.to_csv(index=False), "attendance.csv", "text/csv")

# -----------------------------------------
# 4. Live Camera Attendance
# -----------------------------------------
elif choice == "Live Camera Attendance":
    st.header("Single Frame Attendance")
    if not model_data:
        st.error("Please train the model first!")
    else:
        input_method = st.radio("Choose Input Method:", ("Use Camera", "Upload Photo"), horizontal=True)
        
        image_to_process = None
        
        if input_method == "Use Camera":
            image_to_process = st.camera_input("Take a picture to mark attendance")
        else:
            image_to_process = st.file_uploader("Upload an image file", type=['png', 'jpg', 'jpeg'])
        
        if image_to_process:
            img = Image.open(image_to_process).convert('RGB')
            results = face_utils.predict_face(img, model_data['classifier'], model_data['target_names'])
            
            draw = ImageDraw.Draw(img)
            
            if results:
                st.success(f"Found {len(results)} recognizable faces!")
                for res in results:
                    name = res['name']
                    box = res['box']
                    draw.rectangle(box, outline="green", width=3)
                    st.write(f"✔️ Marked Present: **{name}** ({res['prob']*100:.1f}%)")
                
                st.image(img, caption="Processed Image", use_container_width=True)
            else:
                st.error("No recognizable faces found in this picture.")
