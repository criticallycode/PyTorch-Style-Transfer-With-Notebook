import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image

# function loads image and transforms to PyTorch Tensor, also normalizes
# specify a max size and an optional shape
# normalization numbers come from recommended numbers used on the ImageNet dataset

def image_load(img_path, max_size=800, shape=None):

    # open the image and convert it to RGB
    image = Image.open(img_path).convert('RGB')

    # if the total image size is greater than our specified size,
    # recast to chosen max size
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)


    if shape is not None:
        size = shape


    im_transforms = transforms.Compose([transforms.Resize((size, int(1.5* size))),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])

    # unsqueeze the image after doing transforms
    # unsqueeze takes input, dimensions and output tensor(optional)
    # this discards the transparent alpha channel (:3) and adds batch dimension
    image = im_transforms(image)[:3, :, :].unsqueeze(0)

    return image

# use function to load image
style_image = image_load("style1.jpg")

# let's print the shape to make sure it is what we expect
# should be [1, 3, x, x] - batch dim, color channels, h x w
print(style_image.shape)

# function to convert the tensors back to images
# so that the image can be displayed

def image_convert(tensor):
    # clone the image
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    # convert from a tensor into a numpy array
    image = image.transpose(1, 2, 0)
    # undo normalization
    image = image * np.array((0.229, 0.224, 0.225)) + np.array(
        (0.485, 0.456, 0.406))
    # clip the values to between 0 and 1
    image = image.clip(0, 1)
    return image

def select_features(image, model, layers=None):
    # if no layers are specified, use these layers
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content layer
                  '28': 'conv5_1'}

    # Dict to store the features
    features = {}

    x = image

    # store the feature map responses if the name of the layer matches
    # one of the keys in a given predefined layer dict.

    for name, layer in enumerate(model.features):
        # set x to the layer the image is passing through
        x = layer(x)
        # if the name of the layer is in the chosen layers, get the features from that layer
        if str(name) in layers:
            features[layers[str(name)]] = x

    return features

def gram_matrix(tensor):
    # get the number of filters, height and width (channel doesn't matter)
    _, num_filters, h, w = tensor.size()
    tensor = tensor.view(num_filters, h * w)
    # matrix multiplication against the transpose to get the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram

# load in the VGG model and set requires grad to False

torch.utils.model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', model_dir='./saved_weights/')

vgg_model = models.vgg19()
vgg_model.load_state_dict(torch.load('vgg19-dcbb9e9d.pth'))

for param in vgg_model.parameters():
    param.requires_grad_(False)

# replace max pool with average pooling layers, as they seem to perform better
for i, layer in enumerate(vgg_model.features):
    # if it matches a max pooling layer, replace
    if isinstance(layer, torch.nn.MaxPool2d):
        vgg_model.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model.to(device).eval()

# send both images to the device
content = image_load("content1.jpg").to(device)
style = style_image.to(device)

# extract the feature maps from both of the images
features_content = select_features(content, vgg_model)
features_style = select_features(style, vgg_model)

# compute the gram matrices for the style layers
style_grams = {layer: gram_matrix(features_style[layer]) for layer in features_style}

# Create an image to transform, make random image
target_image = torch.randn_like(content).requires_grad_(True).to(device)

# define the style weights individually
style_image_weights = {'conv1_1': 0.75,
                 'conv2_1': 0.5,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

# track individual loss terms for content and style
content_weight = 1e4
style_weight = 1e2

# declare the optimizer we're going to use

optimizer = optim.Adam([target_image], lr=0.01)

for i in range(1, 6000 + 1):

    optimizer.zero_grad()

    # get the features from the target image and model
    target_features = select_features(target_image, vgg_model)

    # compute contnet loss
    content_loss = torch.mean((target_features['conv4_2'] - features_content['conv4_2']) ** 2)

    # set inital style loss to zero
    style_loss = 0

    # get the individual target features
    for layer in style_image_weights:
        target_feature = target_features[layer]
        # compute the gram matrix
        target_gram = gram_matrix(target_feature)
        # find the shape of the target feature
        _, d, h, w = target_feature.shape
        # get the style gram of the current layer
        style_gram = style_grams[layer]
        # compute the loss for the current style layer
        layer_style_loss = style_image_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        # update the total style loss
        style_loss += layer_style_loss / (d * h * w)

    content_loss = content_weight * content_loss
    style_loss = style_weight * style_loss
    total_loss = content_loss + style_loss
    # do backprop and optimize
    total_loss.backward(retain_graph=True)
    optimizer.step()

    # every 50 iterations, print out statistics
    if i % 50 == 0:
        total_loss_rounded = round(total_loss.item(), 2)
        # proportion of loss belonging to content
        content_fraction = round(content_weight*content_loss.item()/total_loss.item(), 2)
        # proportion of loss belonging to style
        style_fraction = round(content_weight*content_loss.item()/total_loss.item(), 2)
        # Print the current iteration and both the content and style loss
        print('Current Iteration: {}, Total loss: {} - (content: {}, style {})'.format(i, total_loss_rounded, content_fraction, style_fraction))

# now we can carry out our training and create the final image
created_img = image_convert(target_image)

# now let's visualize the image
fig = plt.figure()
plt.imshow(created_img)
plt.axis('off')
plt.savefig('shinkawa-lastofus.png')