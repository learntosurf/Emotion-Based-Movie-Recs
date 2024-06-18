import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_preprocessing import load_data, preprocess_data
from model import BERTClassifier, BERTDataset, initialize_model
from utils import calc_accuracy
from transformers import AdamW, get_cosine_schedule_with_warmup

def train():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    data = load_data('./dataset/5차년도_2차.csv')
    dataset_train, _ = preprocess_data(data)

    model, tokenizer = initialize_model(device)
    
    max_len = 64
    batch_size = 64

    data_train = BERTDataset(dataset_train, 0, 1, tokenizer, max_len, True, False)
    train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=5)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    t_total = len(train_dataloader) * 5
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total)

    for e in range(5):
        train_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            train_acc += calc_accuracy(out, label)
            if batch_id % 200 == 0:
                print(f"epoch {e+1} batch id {batch_id+1} loss {loss.data.cpu().numpy()} train acc {train_acc / (batch_id+1)}")
        print(f"epoch {e+1} train acc {train_acc / len(train_dataloader)}")

    torch.save(model.state_dict(), 'kobert_model.pth')
    print("Model saved to 'kobert_model.pth'")
