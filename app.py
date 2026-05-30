import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np

st.set_page_config(page_title="Blood Cell Counter")

CLASSES = ["Platelets","RBC","WBC"]

##############################################
# LOAD MODEL
##############################################

@st.cache_resource
def load_model():

    model=models.resnet18()

    model.fc=torch.nn.Linear(
        model.fc.in_features,
        3
    )

    model.load_state_dict(
        torch.load(
            "blood_cnn.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model=load_model()

##############################################
# TRANSFORM
##############################################

transform=transforms.Compose([

    transforms.Resize((224,224)),
    transforms.ToTensor(),

])


##############################################
# CLASSIFY SINGLE CELL
##############################################

def predict_cell(crop):

    img=Image.fromarray(
        cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )
    )

    x=transform(img).unsqueeze(0)

    with torch.no_grad():

        pred=model(x)

        probs=torch.softmax(
            pred,
            dim=1
        )

        conf=torch.max(probs).item()

        idx=torch.argmax(
            probs,
            dim=1
        ).item()

    if conf<0.75:

        return None

    return CLASSES[idx]


##############################################
# DETECT CELLS
##############################################

def detect_cells(image):

    img=np.array(image)

    img=cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )

    gray=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    blur=cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )

    circles=cv2.HoughCircles(

        blur,

        cv2.HOUGH_GRADIENT,

        dp=1.2,

        minDist=18,

        param1=50,

        param2=20,

        minRadius=8,

        maxRadius=55

    )

    rbc=0
    wbc=0
    plate=0

    found=0

    if circles is not None:

        circles=np.uint16(
            np.around(circles)
        )

        for c in circles[0]:

            x,y,r=c

            x1=max(
                0,
                x-r
            )

            y1=max(
                0,
                y-r
            )

            x2=min(
                img.shape[1],
                x+r
            )

            y2=min(
                img.shape[0],
                y+r
            )

            crop=img[
                y1:y2,
                x1:x2
            ]

            if crop.size==0:

                continue

            result=predict_cell(
                crop
            )

            if result is None:

                continue

            found+=1

            if result=="RBC":

                rbc+=1
                color=(0,255,0)

            elif result=="WBC":

                wbc+=1
                color=(0,0,255)

            else:

                plate+=1
                color=(255,0,0)

            cv2.circle(
                img,
                (x,y),
                r,
                color,
                2
            )

            cv2.putText(

                img,

                result,

                (x,y),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                color,

                1

            )

    if found<3:

        return "INVALID",None,None

    out=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    return {

        "RBC":rbc,
        "WBC":wbc,
        "Platelets":plate

    },out,found


##############################################
# UI
##############################################

st.title(
"Blood Cell Counter"
)

uploaded=st.file_uploader(
    "Upload Microscopic Image",
    type=["png","jpg","jpeg"]
)

if uploaded:

    image=Image.open(uploaded)

    st.image(
        image,
        caption="Input"
    )

    if st.button("Predict"):

        result, output, total = detect_cells(image)

        if isinstance(result, str):

            st.error(
                "Invalid Image. Upload Microscopic Blood Cell Image."
            )

        else:

            st.success(
                f"Total Cells Found : {total}"
            )

            st.write(
                "RBC Count:",
                result["RBC"]
            )

            st.write(
                "WBC Count:",
                result["WBC"]
            )

            st.write(
                "Platelets Count:",
                result["Platelets"]
            )

            st.image(
                output,
                caption="Detected Cells"
            )