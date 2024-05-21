import streamlit as st
from skimage import io, filters
import numpy as np
import torch
from neural_net.unet import UNet
import matplotlib.pyplot as plt
from PIL import Image

def main():
    st.info(__doc__)
    st.markdown("<h1>Sonuç Ekranı</h1>", unsafe_allow_html=True)
    file = st.file_uploader("Upload file", type=["tiff", "png", "jpg"])

    if not file:
        st.info("Please upload an image file")
        return

    file_name = file.name
    with open(file_name, "wb") as f:
        f.write(file.getvalue())
        st.success(f"File '{file_name}' successfully saved.")
    
    result = io.imread(file_name)
    if len(result.shape) == 2:
        result = result[..., np.newaxis]

    # Görüntüyü uygun boyuta taşı
    image = result.transpose((2, 0, 1))
    torch_image = torch.from_numpy(image)
    torch_image = torch_image.unsqueeze(0)
    torch_image = torch_image[:, :12, :, :]

    # Modeli yükle
    model = UNet(n_channels=12, n_classes=2, act='relu')
    model.load_state_dict(torch.load("C:\\Users\\MÜDAFERKAYMAK\\logs\\burned_area_dataset_paper\\binary_unet_dice\\fold001_cyan\\checkpoint.pt", map_location=torch.device('cpu')))
    model.eval()

    # Görüntüyü model ile işle
    with torch.no_grad():
        output = model(torch_image)
    #print(output)
    # Çıktı tensöründen sadece bir kanalı al
    output_single_channel = output[0, 1, :, :].cpu().numpy()

    # Eşik değeri belirle
    threshold = filters.threshold_otsu(output_single_channel)
    # Eşik değerini biraz düzeltme
    threshold = threshold * 0.9  # Örneğin, eşik değerini %90'ına ayarlayabiliriz

    # Çıktı tensörünü eşik değerine göre siyah-beyaz maskeye dönüştür
    binary_mask = (output_single_channel > threshold).astype(np.uint8)* 255
    plt.imsave('binary_mask.png', binary_mask, cmap='gray')

    final_result = Image.open("binary_mask.png")
    st.image(final_result)




main()
