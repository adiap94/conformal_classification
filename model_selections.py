import torchvision


def select_model(model_str="resnet152"):
    if model_str == "resnet152":
        model = torchvision.models.resnet152(pretrained=True, progress=True)

    elif model_str == "resnet18":
        model = torchvision.models.resnet18(pretrained=True, progress=True)

    elif model_str == "resnet50":
        model = torchvision.models.resnet18(pretrained=True, progress=True)

    elif model_str == "densenet161":
        model = torchvision.models.densenet161(pretrained=True, progress=True)

    elif model_str == "vgg16":
        model = torchvision.models.vgg16(pretrained=True, progress=True)

    elif model_str == "inception_v3":
        model = torchvision.models.inception_v3(pretrained=True, progress=True)

    elif model_str == "shufflenetv2":
        model = torchvision.models.shufflenetv2(pretrained=True, progress=True)

    return model


if __name__ == "__main__":
    select_model()