from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import gc
from transformers import GPT2LMHeadModel, GPT2Tokenizer,GPT2Config, AdamW,AutoTokenizer

class EmbeddingMapper(nn.Module):
    def __init__(self, input_dim, output_sequence_length, output_dim):
        super(EmbeddingMapper, self).__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_sequence_length * output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), -1, self.output_dim)
        return x

class GPT2Decoder(nn.Module):
    def __init__(self, model):
        super(GPT2Decoder, self).__init__()
        self.embedding_mapper = EmbeddingMapper(384, 128, model.config.n_embd)
        self.gpt2 = model

    def forward(self, input_embeddings, **kwargs):
        hidden_states = self.embedding_mapper(input_embeddings)
        return self.gpt2(inputs_embeds=hidden_states, **kwargs)
    
# Prepare the dataset
class CustomDataset(Dataset):
    def __init__(self, embeddings, target_texts, tokenizer, max_length=None):
        self.embeddings = embeddings
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        input_embedding = self.embeddings[idx]
        input_embedding = input_embedding#.unsqueeze(0)#.repeat(self.max_length, 1)  # Repeat the embedding along the sequence dimension
        target_text = self.target_texts[idx]
       
        target = self.tokenizer.encode(target_text, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True).squeeze(0)
        return input_embedding, target


if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
    tokenizer.pad_token = tokenizer.eos_token 
    embeddings = torch.load("train_targets.pt")
    df = pd.read_csv("combined_dataframe.csv")
    target_texts = df["prompt"].tolist()

    max_length = 128  # Set this according to your dataset
    dataset = CustomDataset(embeddings, target_texts, tokenizer,max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,num_workers=8)

    # Set up the model, optimizer, and loss function
    gpt2_decoder = GPT2Decoder(model)
    gpt2_decoder.to("cuda")

    from tqdm import tqdm
    embedding_mapper_params = list(gpt2_decoder.embedding_mapper.parameters())
    gpt2_params = list(gpt2_decoder.gpt2.parameters())

    learning_rate_embedding_mapper = 1e-3
    learning_rate_gpt2 = 5e-5

    optimizer = torch.optim.AdamW([
        {'params': embedding_mapper_params, 'lr': learning_rate_embedding_mapper},
        {'params': gpt2_params, 'lr': learning_rate_gpt2}
    ])

    loss_function = torch.nn.CrossEntropyLoss()
    num_epochs = 8
    print_every_n_batches = 100  # Adjust this value as needed
    batch_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(dataloader, desc="Training", ncols=100)

        for batch in progress_bar:
            input_embeddings, targets = batch
            input_embeddings = input_embeddings.to("cuda")
            targets = targets.to("cuda")

            optimizer.zero_grad()
            outputs = gpt2_decoder(input_embeddings)
            logits = outputs.logits

         # Reshape logits and targets to calculate loss
            #print(logits.shape)
            #print(targets.shape)
            logits = logits.view(-1, logits.size(-1))
            targets_t = targets.view(-1)
            #print(logits.shape)
            #print(targets.shape)
            loss = loss_function(logits, targets_t)

            #outputs = gpt2_decoder(input_embeddings, labels=targets)
            #loss = outputs.loss

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"Loss": loss.item()})
            batch_counter += 1

            # if batch_counter % print_every_n_batches == 0:
            #     print("\n")
            #     print("Generated sequence:", tokenizer.decode(outputs.logits.argmax(-1)[0].detach().cpu(), skip_special_tokens=True))
            #     print("Target text:", tokenizer.decode(targets[-1].detach().cpu(), skip_special_tokens=True))
            #     print("\n")
        #save model after each epoch
        torch.save(gpt2_decoder.state_dict(), f"decoder_epoch_{epoch+1}.pt")
        del targets,input_embeddings,targets_t
        gc.collect()
        torch.cuda.empty_cache()