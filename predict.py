from utils import process_image

from model_constants import MEAN, STD

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # our model takes in batches so introduce a new shape
    # [3, 224, 224] -> [1, 3, 224, 224]
    image = process_image(test_dir+"/1/image_06743.jpg", MEAN, STD)\
            .unsqueeze(0).to(device)
    model.eval()
    logps = model(image)
    
    # Calculate accuracy
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    
    top_class = [cat_to_name[class_to_idx[i.item()]] for i in top_class[0]]
    top_p = [i.item() for i in top_p[0]]
    return top_p, top_class


