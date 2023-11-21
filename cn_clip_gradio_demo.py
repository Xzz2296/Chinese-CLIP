import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import gradio
print("Available models:", available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()
with open('label_cn.txt', 'r', encoding='utf-8') as file:
    chinese_classes = file.readlines()

def classify(image, Model = model, Classes = chinese_classes):
    # Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
    model_clip =Model
    classes_clip = Classes
    image_pil =Image.fromarray(image)
    image = preprocess(image_pil).unsqueeze(0).to(device)

    text = torch.cat([clip.tokenize(f" {c}") for c in chinese_classes]).to(device)
    with torch.no_grad():
        image_features = model_clip.encode_image(image)
        text_features = model_clip.encode_text(text)
        # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity =(100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)
        # return classes_clip[indices], values.item()
        return classes_clip[indices]

interface = gradio.Interface(fn=classify,
                             inputs="image",
                             outputs="label",
                             title="基于Chinese-CLIP的开放词汇物体识别"
                             )
interface.launch()
# print(chinese_classes[indices],values.item())