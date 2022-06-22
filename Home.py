import av
import numpy as np
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
import streamlit as st
from PIL import Image
import numpy as np

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

main_page_markdown = f"""
    ### This site is used for demonstration project of corn disease classification
    **LIBRARY USED**
    - Streamlit and Streamlit WebRTC
    - Conda environment
    - Pytorch for transfer learning
    - Albumentation used for image augmentation

    **HOW TO USE**
    - Clone project from Github Repositories
    - Just install Streamlit and Streamlit WebRTC then run the `app.py` using syntax `streamlit run app.py`
  """


def app_corn_disease_predictor():
    class VideoPredictor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            print(img)

    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoPredictor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def main():
    st.title('Welcome to Corn Disease Classification Demo Application')

    activities = ["Live Classification", "Upload Image", "Camera Input"]
    choice = st.sidebar.selectbox('Select Activity', activities)

    if choice == "Live Classification":
        st.markdown(main_page_markdown)
        app_corn_disease_predictor()
        st.markdown(f'''Press Start 👈 to start the show!''')

    elif choice == 'Upload Image':
        st.subheader('Inference model using image')
        image_file = st.file_uploader('Upload Image', type=["jpg", "png", "jpeg"])
        if image_file is not None:
            our_static_image = Image.open(image_file)
            st.image(our_static_image, width=400)

            # Predict here
    elif choice == "Camera Input":
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            # To read image file buffer as a PIL Image:
            img = Image.open(img_file_buffer)
            st.image(img, width=400)

            # To convert PIL Image to numpy array:
            img_array = np.array(img)

            # Check the type of img_array:
            # Should output: <class 'numpy.ndarray'>
            st.write(type(img_array))

            # Check the shape of img_array:
            # Should output shape: (height, width, channels)
            st.write(img_array.shape)


if __name__ == '__main__':
    main()
