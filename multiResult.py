from skimage import io
import numpy as np
import torch
import torch.nn.functional as F
from neural_net.unet import UNet
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

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

# Giriş görüntüsünü yükle
path = 'C:\\Users\\MÜDAFERKAYMAK\\Desktop\\Ara Proje\\Implementation\\burned-area-baseline\\OrnekResimler\\orn2.tiff'
result = io.imread(path)
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
model.load_state_dict(torch.load('C:\\Users\\MÜDAFERKAYMAK\\Desktop\\Ara Proje\\Implementation\\burned-area-baseline\\multi.pt', map_location=torch.device('cpu')))
model.eval()

# Görüntüyü model ile işle
with torch.no_grad():
    output = model(torch_image)
    probabilities = F.softmax(output, dim=1)  # Softmax uygulaması
    predicted_classes = torch.argmax(probabilities, dim=1)  # En yüksek olasılıklı sınıfı seç

# Çıktı tensöründen yalnızca bir kanalı al
output_single_channel = predicted_classes[0].cpu().numpy()

# Görüntüyü 4 sınıflı renk haritası ile göster
plt.imshow(output_single_channel, cmap='tab10')
plt.axis('off')
plt.show()

plt.imsave('ornek3torchOut_multiclass.png', output_single_channel, cmap='tab10')

# Sınıf dağılımını hesapla
unique, counts = np.unique(output_single_channel, return_counts=True)
class_counts = dict(zip(unique, counts))

print(class_counts)