# https://github.com/braindotai/Watermark-Removal-Pytorch

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid

def pil_to_np_array(pil_image):
    ar = np.array(pil_image)
    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]
    return ar.astype(np.float32) / 255.

def np_to_torch_array(np_array):
    return torch.from_numpy(np_array)[None, :]

def torch_to_np_array(torch_array):
    return torch_array.detach().cpu().numpy()[0]

def read_image(path, image_size = -1):
    pil_image = Image.open(path)
    return pil_image

def crop_image(image, crop_factor = 64):
    shape = (image.size[0] - image.size[0] % crop_factor, image.size[1] - image.size[1] % crop_factor)
    bbox = [int((image.shape[0] - shape[0])/2), int((image.shape[1] - shape[1])/2), int((image.shape[0] + shape[0])/2), int((image.shape[1] + shape[1])/2)]
    return image.crop(bbox)

def get_image_grid(images, nrow = 3):
    torch_images = [torch.from_numpy(x) for x in images]
    grid = make_grid(torch_images, nrow)
    return grid.numpy()
    
def visualize_sample(*images_np, nrow = 3, size_factor = 10):
    c = max(x.shape[0] for x in images_np)
    images_np = [x if (x.shape[0] == c) else np.concatenate([x, x, x], axis = 0) for x in images_np]
    grid = get_image_grid(images_np, nrow)
    plt.figure(figsize = (len(images_np) + size_factor, 12 + size_factor))
    plt.axis('off')
    plt.imshow(grid.transpose(1, 2, 0))
    plt.show()

def max_dimension_resize(image_pil, mask_pil, max_dim):
    w, h = image_pil.size
    aspect_ratio = w / h
    if w > max_dim:
        h = int((h / w) * max_dim)
        w = max_dim
    elif h > max_dim:
        w = int((w / h) * max_dim)
        h = max_dim
    return image_pil.resize((w, h)), mask_pil.resize((w, h))

def preprocess_images(image_path, mask_path, max_dim):
    image_pil = read_image(image_path).convert('RGB')
    mask_pil = read_image(mask_path).convert('RGB')

    image_pil, mask_pil = max_dimension_resize(image_pil, mask_pil, max_dim)

    image_np = pil_to_np_array(image_pil)
    mask_np = pil_to_np_array(mask_pil)

    print('Visualizing mask overlap...')

    visualize_sample(image_np, mask_np, image_np * mask_np, nrow = 3, size_factor = 10)

    return image_np, mask_np

import torch
import numpy as np
from torch import nn

from torch import optim
from tqdm.auto import tqdm

class DepthwiseSeperableConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super(DepthwiseSeperableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(input_channels, input_channels, groups = input_channels, **kwargs)
        self.pointwise = nn.Conv2d(input_channels, output_channels, kernel_size = 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, bias = False):
        super(Conv2dBlock, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size - 1) / 2)),
            DepthwiseSeperableConv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = 0, bias = bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.model(x)

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                diff3 = (inp.size(3) - target_shape3) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)

class SkipEncoderDecoder(nn.Module):
    def __init__(self, input_depth, num_channels_down = [128] * 5, num_channels_up = [128] * 5, num_channels_skip = [128] * 5):
        super(SkipEncoderDecoder, self).__init__()

        self.model = nn.Sequential()
        model_tmp = self.model

        for i in range(len(num_channels_down)):

            deeper = nn.Sequential()
            skip = nn.Sequential()

            if num_channels_skip[i] != 0:
                model_tmp.add_module(str(len(model_tmp) + 1), Concat(1, skip, deeper))
            else:
                model_tmp.add_module(str(len(model_tmp) + 1), deeper)
            
            model_tmp.add_module(str(len(model_tmp) + 1), nn.BatchNorm2d(num_channels_skip[i] + (num_channels_up[i + 1] if i < (len(num_channels_down) - 1) else num_channels_down[i])))

            if num_channels_skip[i] != 0:
                skip.add_module(str(len(skip) + 1), Conv2dBlock(input_depth, num_channels_skip[i], 1, bias = False))
                
            deeper.add_module(str(len(deeper) + 1), Conv2dBlock(input_depth, num_channels_down[i], 3, 2, bias = False))
            deeper.add_module(str(len(deeper) + 1), Conv2dBlock(num_channels_down[i], num_channels_down[i], 3, bias = False))

            deeper_main = nn.Sequential()

            if i == len(num_channels_down) - 1:
                k = num_channels_down[i]
            else:
                deeper.add_module(str(len(deeper) + 1), deeper_main)
                k = num_channels_up[i + 1]

            deeper.add_module(str(len(deeper) + 1), nn.Upsample(scale_factor = 2, mode = 'nearest'))

            model_tmp.add_module(str(len(model_tmp) + 1), Conv2dBlock(num_channels_skip[i] + k, num_channels_up[i], 3, 1, bias = False))
            model_tmp.add_module(str(len(model_tmp) + 1), Conv2dBlock(num_channels_up[i], num_channels_up[i], 1, bias = False))

            input_depth = num_channels_down[i]
            model_tmp = deeper_main

        self.model.add_module(str(len(self.model) + 1), nn.Conv2d(num_channels_up[0], 3, 1, bias = True))
        self.model.add_module(str(len(self.model) + 1), nn.Sigmoid())
    
    def forward(self, x):
        return self.model(x)


def input_noise(INPUT_DEPTH, spatial_size, scale = 1./10):
    shape = [1, INPUT_DEPTH, spatial_size[0], spatial_size[1]]
    return torch.rand(*shape) * scale

def remove_watermark(image_path, mask_path, max_dim, reg_noise, input_depth, lr, show_step, training_steps, tqdm_length=100):
    DTYPE = torch.FloatTensor
    has_set_device = False
    if torch.cuda.is_available():
        device = 'cuda'
        has_set_device = True
        print("Setting Device to CUDA...")
    try:
        if torch.backends.mps.is_available():
            device = 'mps'
            has_set_device = True
            print("Setting Device to MPS...")
    except Exception as e:
        print(f"Your version of pytorch might be too old, which does not support MPS. Error: \n{e}")
        pass
    if not has_set_device:
        device = 'cpu'
        print('\nSetting device to "cpu", since torch is not built with "cuda" or "mps" support...')
        print('It is recommended to use GPU if possible...')

    image_np, mask_np = preprocess_images(image_path, mask_path, max_dim)

    print('Building the model...')
    generator = SkipEncoderDecoder(
        input_depth,
        num_channels_down = [128] * 5,
        num_channels_up = [128] * 5,
        num_channels_skip = [128] * 5
    ).type(DTYPE).to(device)

    objective = torch.nn.MSELoss().type(DTYPE).to(device)
    optimizer = optim.Adam(generator.parameters(), lr)

    image_var = np_to_torch_array(image_np).type(DTYPE).to(device)
    mask_var = np_to_torch_array(mask_np).type(DTYPE).to(device)

    generator_input = input_noise(input_depth, image_np.shape[1:]).type(DTYPE).to(device)

    generator_input_saved = generator_input.detach().clone()
    noise = generator_input.detach().clone()

    print('\nStarting training...\n')

    progress_bar = tqdm(range(training_steps), desc='Completed', ncols=tqdm_length)

    for step in progress_bar:
        optimizer.zero_grad()
        generator_input = generator_input_saved

        if reg_noise > 0:
            generator_input = generator_input_saved + (noise.normal_() * reg_noise)

        output = generator(generator_input)

        loss = objective(output * mask_var, image_var * mask_var)
        loss.backward()

        if step % show_step == 0:
            output_image = torch_to_np_array(output)
            visualize_sample(image_np, output_image, nrow = 2, size_factor = 10)

        progress_bar.set_postfix(Loss = loss.item())

        optimizer.step()

    output_image = torch_to_np_array(output)
    visualize_sample(output_image, nrow = 1, size_factor = 10)

    pil_image = Image.fromarray((output_image.transpose(1, 2, 0) * 255.0).astype('uint8'))

    output_path = image_path.split('/')[-1].split('.')[-2] + '-output.jpg'
    print(f'\nSaving final output image to: "{output_path}"\n')

    pil_image.save(output_path)

if __name__ == '__main__':
    remove_watermark(
        image_path = 'images/1.jpg',
        mask_path = 'images/1-mask.jpg',
        max_dim = 512,
        reg_noise = 0.01,
        input_depth = 32,
        lr = 0.001,
        show_step = 100,
        training_steps = 10000,
        tqdm_length = 900
    )