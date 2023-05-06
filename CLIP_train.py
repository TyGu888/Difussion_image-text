import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from glob import glob
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
from transformers import AutoModel, AutoProcessor
from transformers import CLIPProcessor, CLIPModel,CLIPImageProcessor
from sentence_transformers import SentenceTransformer
import gc
clip_processor =  CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
#clip_processor =  AutoProcessor.from_pretrained(".\clip_processor")
BATCHSIZE=128
SAVE_OPT_CKP = True
SAVE_MODEL_CKP = True
UNFREEZE_START = 20# set it to lower number when significantly more samples are included.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

run_name = f'clip224-l18'


def cosine_similarity_loss(pred, target):
    cos = nn.CosineSimilarity(dim=1)
    output = -cos(pred, target).mean()
    return output


def get_train_test_split():
    """add your image paths and embedding labels here"""
    #encoder = SentenceTransformer('all-MiniLM-L6-v2')
    #encoder = encoder.to('cuda')
    #train data is df_filtered.csv, train_images'path is in column 'filepath', train_labels is in column 'prompt'
    #test data is test.csv, test_images'path is in column 'image_path', test_labels is in column 'Prompt'
    
    #train_data = pd.read_csv('combined_dataframe.csv')
    #test_data = pd.read_csv('test.csv')
    #train_images = train_data['file_path']
    #train_labels = train_data['prompt']
    #train_labels=encoder.encode(train_data['prompt'].tolist(), batch_size=512, show_progress_bar=True, device="cuda", convert_to_tensor=True)
    #test_images = test_data['image_path']
    #test_labels = test_data['Prompt']
    #test_labels=encoder.encode(test_data['Prompt'].tolist(), batch_size=512, show_progress_bar=True, device="cuda", convert_to_tensor=True)
    #del encoder
    #gc.collect()
    #torch.cuda.empty_cache()
    #train_labels = torch.load('train_targets.pt').cpu().numpy()
    #test_labels = torch.load('test_targets.pt').cpu().numpy()
    train_data = pd.read_csv('combined_dataframe.csv')
    all_train_images = train_data['file_path'].to_list()
    all_train_labels = torch.load('train_targets.pt').cpu().numpy()

    # Split the data into train and validation sets (9:1 ratio)
    train_images, test_images, train_labels, test_labels = train_test_split(
        all_train_images, all_train_labels, test_size=0.1, random_state=42
    )

    return train_images, train_labels, test_images, test_labels


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        clip = AutoModel.from_pretrained("H:\clip224-l18")
        self.vision = clip.vision_model
        self.fc = nn.Linear(1024, 384)

    def forward(self, x):
        out = self.vision(x)['pooler_output']
        return self.fc(out)


def load_pretrained_model():
    model = Net()

    trainable_model_weights = False
    for name, child in model.named_children():
        if name == 'vision':
            for pn, p in child.named_parameters():
                if str(UNFREEZE_START) in pn:
                    """start unfreezing layer , the weights are trainable"""
                    trainable_model_weights = True
                p.requires_grad = trainable_model_weights
                if p.requires_grad:
                    print(f"{pn} is set to be trainable.")

    return model.to(device)


class IMGDataset:
    def __init__(self, image_paths, targets, clip_processor=clip_processor):
        self.images = image_paths
        self.labels = targets
        self.input_processor = clip_processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        image = self.input_processor(image)['pixel_values']
        #image = torch.tensor(image)
        target = self.labels[item]
       # target = torch.tensor(target, dtype=torch.float)
        return image[0], target
    
if __name__ == "__main__":
    """main training"""
    Path(f"./{run_name}").mkdir(exist_ok=True)
    

    NEPOCH=25
    BestEpoch=0
    BestSim = 0
    train_images, train_targets, test_images, test_targets = get_train_test_split()

    print(f"test size: {len(test_images)}, train size: {len(train_images)}")
    #train_targets=np.loadtxt('train_targets.csv', delimiter=',')
    #test_targets=np.loadtxt('test_targets.csv', delimiter=',')
    #gc.collect()
    #torch.cuda.empty_cache()
    nn_model = load_pretrained_model()
    #nn_model = torch.compile(nn_model)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, nn_model.parameters()), lr=1e-4, fused=True)
    optimizer.zero_grad()
    test_dataloader = DataLoader(IMGDataset(test_images, test_targets),
                                 batch_size=BATCHSIZE, shuffle=False, num_workers=8)
    train_dataloader = DataLoader(IMGDataset(train_images, train_targets),
                                 batch_size=BATCHSIZE, shuffle=True, num_workers=10)



    print('load')

    for epoch in range(NEPOCH):
        epoch_loss = 0
        print(f'Entering epoch {epoch}')
        for s, batch_data in enumerate(tqdm(train_dataloader)):
            batch_images, batch_targets = batch_data

            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)

            pred = nn_model(batch_images)
            cosine_loss = cosine_similarity_loss(pred, batch_targets)
            loss = cosine_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += -cosine_loss.item()
        epoch_loss /= len(train_dataloader)
        print(f"epoch: {epoch}, training loss: {epoch_loss}")
        
        """test loss"""
        epoch_loss = 0
        with torch.no_grad():
            for batch_images, batch_targets in tqdm(test_dataloader):
                batch_images, batch_targets = batch_images.to(device), batch_targets.to(device)
                pred = nn_model(batch_images)
                loss = -cosine_similarity_loss(pred, batch_targets)
                epoch_loss += loss.item()
            epoch_loss /= len(test_dataloader)
        print(f"epoch: {epoch}, test loss: {epoch_loss}")

        if epoch_loss > BestSim:
            BestSim = epoch_loss
            BestEpoch = epoch + 1
            print(f"save best model at {BestSim} with epoch {BestEpoch}")
            if SAVE_MODEL_CKP:
                torch.save(nn_model.state_dict(), f"{run_name}.pt")
            if SAVE_OPT_CKP:
                torch.save(optimizer.state_dict(), f"{run_name}_opt.pt")

        if epoch - 3 > BestEpoch:
            print(f"early stop at {epoch+1} with best epoch {BestEpoch} and test similarity {BestSim}.")
            break