from torchvision import transforms
import torch

def get_transform(type='clip', keep_ratio=True, image_size=224):
    if type == 'clip':
        transform = []
        if keep_ratio:
            transform.extend([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ])
        else:
            transform.append(transforms.Resize((image_size, image_size)))
        transform.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        return transforms.Compose(transform)
    else:
        raise NotImplementedError

# 反标准化操作
def un_transform(tensor):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    tensor = tensor.clone()  # 避免改变原始tensor
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor

transform = get_transform(type='clip', keep_ratio=False, image_size=224)