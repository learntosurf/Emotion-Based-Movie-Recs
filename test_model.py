import torch
import numpy as np
from torch.utils.data import DataLoader
from data_preprocessing import preprocess_data, load_data
from model import BERTClassifier, BERTDataset, initialize_model

def load_model():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model, tokenizer = initialize_model(device)
    model.load_state_dict(torch.load('kobert_model.pth', map_location=device))
    model.eval()
    return model, tokenizer, device

def predict(predict_sentence, model, tokenizer, device):
    max_len = 64
    batch_size = 64
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, max_len, True, False)
    test_dataloader = DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("기쁨(Joy)")
            elif np.argmax(logits) == 1:
                test_eval.append("슬픔(Sadness)")
            elif np.argmax(logits) == 2:
                test_eval.append("버럭(Anger)")
            elif np.argmax(logits) == 3:
                test_eval.append("까칠(Disgust)")
            elif np.argmax(logits) == 4:
                test_eval.append("소심(Fear)")

        print(">> 당신의 감정은 " + test_eval[0] + "(이)와 유사하게 느껴집니다.")

if __name__ == "__main__":
    model, tokenizer, device = load_model()

    end = 1
    while end == 1:
        sentence = input("하고싶은 말을 입력해주세요 : ")
        if sentence == "0":
            break
        predict(sentence, model, tokenizer, device)
        print("\n")
