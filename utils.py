import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns

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
            if k[0] == 'x':
                ax.imshow(convert_tensor_to_PIL(v[idx]), cmap='gray')
            else:
                sns.heatmap(
                    v[idx][0].data.cpu().numpy(), vmin=0, vmax=1, 
                    cmap='gray', square=True, cbar=False)
            ax.set_title(k)
            ax.set_xticks([])
            ax.set_yticks([])    
        plt.show()
        
        
def tanh2sigmoid(x):
    return (x + 1) / 2 


def torch_log(x, eps=1e-12):
    return torch.log(torch.clamp(x, eps, 1.))


def l2_BCE(y, t, eps=1e-12):
    return -(t*torch_log(y**2) + (1-t)*torch_log((1-y)**2)).mean()

