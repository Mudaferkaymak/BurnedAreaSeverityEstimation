import streamlit as st
from skimage import io, filters
import numpy as np
import torch
from neural_net.unet import UNet
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import aspose.words as aw
import rasterio
import os
from matplotlib.colors import ListedColormap
import time
from PIL import Image
import datetime

def create_accuracy_image(grayscale_image_path, rgba_image_path, output_image_path):
    # Load images
    grayscale_image = Image.open(grayscale_image_path).convert('L')
    rgba_image = Image.open(rgba_image_path).convert('RGBA')
    
    # Convert images to numpy arrays
    grayscale_array = np.array(grayscale_image)
    rgba_array = np.array(rgba_image)
    
    # Ensure both images have the same dimensions
    assert grayscale_array.shape == rgba_array.shape[:2], "Images must have the same dimensions"
    
    # Create an empty image for the output
    accuracy_image = Image.new('RGB', grayscale_image.size)
    accuracy_array = np.array(accuracy_image)
    
    # Define conditions
    conditions = [
        ((0, 63), (255, 255, 255, 255)),
        ((64, 127), (255, 165, 0, 255)),
        ((128, 191), (165, 42, 42, 255)),
        ((192, 256), (255, 0, 0, 255))
    ]
    
    # Iterate through each pixel and apply conditions
    for y in range(grayscale_array.shape[0]):
        for x in range(grayscale_array.shape[1]):
            grayscale_value = grayscale_array[y, x]
            rgba_value = tuple(rgba_array[y, x])
            
            correct = False
            for (g_range, rgba_match) in conditions:
                if g_range[0] <= grayscale_value <= g_range[1] and rgba_value == rgba_match:
                    correct = True
                    break
            
            if correct:
                accuracy_array[y, x] = (0, 255, 0)  # Green for correct
            else:
                accuracy_array[y, x] = (255, 0, 0)  # Red for incorrect
    
    # Convert accuracy array back to image and save
    accuracy_image = Image.fromarray(accuracy_array)
    accuracy_image.save(output_image_path)

# Usage example

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Normalize sınıfı
class Normalize(object):
    def __init__(self, t1, t2):
        self.norm = transforms.Normalize(t1, t2)

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        result = {}
        image = self.norm(image)
        result['image'] = image
        result['mask'] = mask
        return result

def main():
    # Load custom CSS
    load_css("style.css")

    st.markdown("<h1>Orman Yangını Hasar Analizi</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        file = st.file_uploader("Orijinal Görsel Yükleme Alanı", type=["tiff"])
    
    with col2:
        mask_file = st.file_uploader("Maske Görseli Yükleme Alanı (Opsiyonel)", type=["png"])

    if not file:
        st.info("Lütfen tiff uzantılı orijinal görselinizi ve maske görselinizi opsiyonel bir şekilde yükleyiniz!")
        return
    print(f"Original image has been uploaded. Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    startTime = time.time()
    file_name = file.name
    mask_file_name = mask_file.name if mask_file else None

    with open(file_name, "wb") as f:
        f.write(file.getvalue())
        st.success(f"Orijinal görsel '{file_name}' başarıyla kaydedildi.")
    
    if mask_file:
        with open(mask_file_name, "wb") as f:
            f.write(mask_file.getvalue())
            st.success(f"Maske görseli '{mask_file_name}' başarıyla kaydedildi.")

    with rasterio.open(file_name) as src:
        # Bantları oku
        red = src.read(4)    # 4. bant
        green = src.read(3)  # 3. bant
        blue = src.read(2)   # 2. bant

    def normalize(band):
        band_min, band_max = (band.min(), band.max())
        return ((band-band_min)/((band_max - band_min)))

    def gammacorr(band):
        gamma = 3
        return np.power(band, 1 / gamma)

    red_g = gammacorr(red)
    blue_g = gammacorr(blue)
    green_g = gammacorr(green)

    red_gn = normalize(red_g)
    green_gn = normalize(green_g)
    blue_gn = normalize(blue_g)

    rgb_composite_gn = np.dstack((red_gn, green_gn, blue_gn))
    
    result = io.imread(file_name)
    if len(result.shape) == 2:
        result = result[..., np.newaxis]

    # Görüntüyü uygun boyuta taşı
    image = result.transpose((2, 0, 1)).astype(np.float32)  # float32 olarak çevir
    torch_image = torch.from_numpy(image).float()
    torch_image = torch_image.unsqueeze(0)
    torch_image = torch_image[:, :12, :, :]

    # Normalizasyon: Eğitim sırasında kullanılan aynı değerler
    mean = [0.5] * 12  # 12 kanal için ortalama değeri
    std = [0.5] * 12  # 12 kanal için standart sapma değeri
    normalize = Normalize(mean, std)

    # Normalize görüntü
    sample = {'image': torch_image.squeeze(0), 'mask': None}  # Mask kullanmıyorsanız None olarak bırakın
    normalized_sample = normalize(sample)
    torch_image = normalized_sample['image'].unsqueeze(0)

    # Modeli yükle
    model = UNet(n_channels=12, n_classes=4, act='relu')
    model.load_state_dict(torch.load('limev1.pt', map_location=torch.device('cpu')))
    model.eval()

    # Görüntüyü model ile işle
    with torch.no_grad():
        output = model(torch_image)
        probabilities = F.softmax(output, dim=1)  # Softmax uygulaması
        predicted_classes = torch.argmax(probabilities, dim=1)  # En yüksek olasılıklı sınıfı seç

    # Çıktı tensöründen sadece bir kanalı al
    output_single_channel = predicted_classes[0].cpu().numpy()
    colors = ['white', 'orange', 'brown', 'red']
    cmap = ListedColormap(colors)

    # Görüntüyü 4 sınıflı renk haritası ile göster

    plt.imsave('ornektorchOut_multiclass.png', output_single_channel, cmap=cmap)

    unique, counts = np.unique(output_single_channel, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print(class_counts)
    final_result = Image.open("ornektorchOut_multiclass.png")

    # İlk anahtarı al
    firs_class = next(iter(class_counts))

    # İlk anahtarı haricindekileri al
    other_class = {key: value for key, value in class_counts.items() if key != firs_class}
    class_sum = sum(other_class.values())
    class_percent = {key: round((value / class_sum) * 100, 1) for key, value in other_class.items()}

    if mask_file:
        create_accuracy_image(mask_file_name, 'ornektorchOut_multiclass.png', 'output_pixel_values.png')
        comparison = Image.open("output_pixel_values.png")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(rgb_composite_gn, caption='Orijinal Görsel', use_column_width=True)

        
        with col2:
            mask_image = io.imread(mask_file_name)
            st.image(mask_image, caption='Maske Görseli', use_column_width=True)
            coll1, coll2, coll3 = st.columns([1,1,2])
            #with coll3:
            #   st.write("    ")
            #   st.write(f"  %{class_percent[1]}")
            #   st.write(f"  %{class_percent[2]}")
            #   st.write(f"  %{class_percent[3]}")
            with coll3:
                st.write("    ")
                st.write(f"Düşük")
                st.write(f"Orta")
                st.write(f"Yüksek")
            with coll2:
                st.write("    ")
                white = Image.open("Colors/KoyuGri.jpg")
                white = white.resize((10, 10))  # Genişlik 300, yükseklik 200 piksel olarak yeniden boyutlandır
                st.image(white)
                #st.write("    ")
                orange = Image.open("Colors/AGri.jpg")
                orange = orange.resize((10, 10))  # Genişlik 300, yükseklik 200 piksel olarak yeniden boyutlandır
                st.image(orange)
                #st.write("    ")
                red = Image.open("Colors/Beyaz.jpg")
                red = red.resize((10, 10))  # Genişlik 300, yükseklik 200 piksel olarak yeniden boyutlandır
                st.image(red)
            
        with col3:
            st.image(final_result, caption='Sonuç Görseli', use_column_width=True)
            coll1, coll2, coll3, coll4 = st.columns([1,1,2,2])
            with coll4:
                st.write("    ")
                st.write(f"  %{class_percent[1]}")
                st.write(f"  %{class_percent[2]}")
                st.write(f"  %{class_percent[3]}")
            with coll3:
                st.write("    ")
                st.write(f"Düşük")
                st.write(f"Orta")
                st.write(f"Yüksek")
            with coll2:
                st.write("    ")
                white = Image.open("Colors/Turuncu.png")
                white = white.resize((10, 10))  # Genişlik 300, yükseklik 200 piksel olarak yeniden boyutlandır
                st.image(white)
                #st.write("    ")
                orange = Image.open("Colors/Kahverengi.jpg")
                orange = orange.resize((10, 10))  # Genişlik 300, yükseklik 200 piksel olarak yeniden boyutlandır
                st.image(orange)
                #st.write("    ")
                red = Image.open("Colors/Kirmizi.jpg")
                red = red.resize((10, 10))  # Genişlik 300, yükseklik 200 piksel olarak yeniden boyutlandır
                st.image(red)

        with col4:
            st.image(comparison, caption='Karşılaştırma Görseli', use_column_width=True)
            coll1, coll2, coll3 = st.columns([1, 1, 2])
            #with coll3:
            #   st.write("    ")
            #   st.write(f"  %{class_percent[1]}")
            #   st.write(f"  %{class_percent[2]}")
            #   st.write(f"  %{class_percent[3]}")
            with coll3:
                st.write("    ")
                st.write(f"Eşleşen")
                st.write(f"Eşleşmeyen")
                #st.write(f"Yüksek")
            with coll2:
                st.write("    ")
                white = Image.open("Colors/yesil.jpg")
                white = white.resize((10, 10))  # Genişlik 300, yükseklik 200 piksel olarak yeniden boyutlandır
                st.image(white)
                #st.write("    ")
                orange = Image.open("Colors/Kirmizi.jpg")
                orange = orange.resize((10, 10))  # Genişlik 300, yükseklik 200 piksel olarak yeniden boyutlandır
                st.image(orange)
                #st.write("    ")
        maskEndtime = time.time()
        print(f"All outputs sent to frontend in {maskEndtime - startTime} seconds. Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb_composite_gn, caption='Orijinal Görsel', use_column_width=True)


        with col2:
            st.image(final_result, caption='Sonuç Görseli', use_column_width=True)
            coll1, coll2, coll3, coll4 = st.columns([1,1,2,2])
            with coll4:
                st.write("    ")
                st.write(f"  %{class_percent[1]}")
                st.write(f"  %{class_percent[2]}")
                st.write(f"  %{class_percent[3]}")
            with coll3:
                st.write("    ")
                st.write(f"Düşük")
                st.write(f"Orta")
                st.write(f"Yüksek")
            with coll2:
                st.write("    ")
                white = Image.open("Colors/Turuncu.png")
                white = white.resize((10, 10))  # Genişlik 300, yükseklik 200 piksel olarak yeniden boyutlandır
                st.image(white)
                #st.write("    ")
                orange = Image.open("Colors/Kahverengi.jpg")
                orange = orange.resize((10, 10))  # Genişlik 300, yükseklik 200 piksel olarak yeniden boyutlandır
                st.image(orange)
                #st.write("    ")
                red = Image.open("Colors/Kirmizi.jpg")
                red = red.resize((10, 10))  # Genişlik 300, yükseklik 200 piksel olarak yeniden boyutlandır
                st.image(red)
        outmaskendtime = time.time()
        print(f"Model output sent to frontend in {outmaskendtime - startTime} seconds. Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")       


if __name__ == "__main__":
    main()
