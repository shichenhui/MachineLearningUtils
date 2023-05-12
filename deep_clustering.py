def extract_feature(extractor, data_loader):
    device = next(extractor.parameters()).device # use same device as model parameters
    extractor.eval()
    feature_list = []
    label_list = []
    for x, y in tqdm(data_loader):
        x = x.to(device)
        y = y.to(device)
        feature = extractor(x)
        feature = feature.cpu().detach()
        y = y.cpu().detach()
        feature_list.append(feature)
        label_list.append(y)
    
    feature_list = torch.cat(feature_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    feature_list = feature_list.cpu().detach().numpy()
    label_list = label_list.cpu().detach().numpy()
    
    return feature_list, label_list
