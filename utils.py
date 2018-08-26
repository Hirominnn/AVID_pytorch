import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def convert_tensor_to_PIL(tensor, out_size=None):
    out = transforms.ToPILImage()(tensor.cpu())
    if out_size is not None:
        out = out.resize(out_size)
    return out


def show_imgs(imgs, idxs=[0]):
    n = len(imgs)
    
    for idx in idxs:
        plt.figure(figsize=(20, 5))
        for i, (k, v) in enumerate(imgs.items()):
            ax = plt.subplot(1, n, i+1)
            ax.set_title(k)
            ax.imshow(convert_tensor_to_PIL(v[idx]), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])    

        plt.show()
        
        
