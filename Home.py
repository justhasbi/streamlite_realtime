import av
from streamlit_webrtc import (RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer)
import streamlit as st
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os
import cv2

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


def prediction(image_name, transform, inference_model, prediction_type=None):
    if prediction_type == "realtime":
        image_prediction = Image.fromarray(image_name)
    else:
        image_prediction = Image.open(image_name)

    image_prediction = transform(image_prediction).float()
    image_prediction = image_prediction.unsqueeze(0)
    loader = torch.utils.data.DataLoader(image_prediction, batch_size=1, shuffle=True)
    data_iteration = iter(loader).next()

    with torch.no_grad():
        inference_model.eval()
        output = inference_model(data_iteration)
        sm = torch.nn.Sigmoid()
        sm_output = (sm(output).squeeze(0)).tolist()
        index = sm_output.index(max(sm_output))

    return index, sm_output


def app_corn_disease_predictor():
    model_name = 'mobilenetv2'
    num_classes = 4
    feature_extract = False
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transAug = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std),
         transforms.Resize((256, 256))])
    idx_to_class = {
        0: 'Karat Daun',
        1: 'Bercak Daun',
        2: 'bulai',
        3: 'Hawar Daun'
    }

    model_ft, _ = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    class VideoPredictor(VideoProcessorBase):
        def __init__(self):
            pass
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            idx, output = prediction(img, transAug, model_ft, prediction_type="realtime")

            print(f"Prediction: {idx} - {idx_to_class[idx]}")
            print(f"Probability : {output[idx]}")
            print(f"Overall prediction : {output}")

            cv2.putText(img, f'{idx} - {idx_to_class[idx]}, {output}', (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.9, (255, 255, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="corn-disease-classification",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoPredictor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


@st.cache(allow_output_mutation=True)
def retrieve_model(pretrained):
    model = models.mobilenet_v2(pretrained=pretrained)
    return model


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == 'mobilenetv2':
        model_ft = retrieve_model(use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_features, num_classes)
        model_state_dict = os.path.join(os.path.dirname(__file__), 'model (1).pth')
        model_ft.load_state_dict(torch.load(model_state_dict, map_location=torch.device('cpu')))

    else:
        print("invalid model name")
        exit()

    return model_ft, input_size


def main():
    st.title('Welcome to Corn Disease Classification Demo Application')

    activities = ["Live Classification", "Upload Image", "Camera Input", "About", "Contact"]
    choice = st.sidebar.selectbox('Select Activity', activities)

    model_name = 'mobilenetv2'
    num_classes = 4
    feature_extract = False
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    idx_to_class = {
        0: 'Karat Daun',
        1: 'Bercak Daun',
        2: 'bulai',
        3: 'Hawar Daun'
    }

    transAug = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std),
         transforms.Resize((256, 256))])
    model_ft, _ = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)


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
            idx, output = prediction(image_file, transAug, model_ft)
            output = [float("{:.2f}".format(i)) for i in output]

            st.write(f""" ### Prediction: {idx} - {idx_to_class[idx]}""")
            st.write(f"Probability : {output[idx]}")
            st.write(f"Overall prediction : **{listOutput}**")

    elif choice == "Camera Input":
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            # To read image file buffer as a PIL Image:
            img = Image.open(img_file_buffer)
            st.image(img, width=400)
            output = [float("{:.2f}".format(i)) for i in output]

            st.write(f""" ### Prediction: {idx} - {idx_to_class[idx]}""")
            st.write(f"Probability : {output[idx]}")
            st.write(f"Overall prediction : **{output}**")

    elif choice == "About":
        st.markdown("#About")

    elif choice == "Contact":
        st.markdown("#Contact")


if __name__ == '__main__':
    main()
