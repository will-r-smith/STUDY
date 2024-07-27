import torch


def generate_outputs(self, model, X, y, requires_grad, get_accuracy):

    #my_batch_size = len(X)

    input_ids = self.tokenizer(X, return_tensors="pt", padding="longest", truncation=True).to(self.device)

    answer_ids = self.tokenizer(y, return_tensors="pt", padding="longest", truncation=True).input_ids.to(self.device)
    #may need to select the 0 index here ^^^

    # Compute the lengths of the original input sequences
    input_lengths = [len(self.tokenizer.encode(x, truncation=True)) for x in X]

    # Extract the last token index for each sequence before padding
    answer_positions = torch.tensor([input_lengths[i] - 1 for i in range(len(X))]).to(self.device)


    answer_ids = answer_ids[:, -1]

    print(input_ids['input_ids'].shape)
    print(answer_ids.shape)

    torch.cuda.empty_cache()
    
    if requires_grad == False:
        with torch.no_grad():
            logits = model(**input_ids).logits
    else:
        logits = model(**input_ids).logits

    print(logits)

    answer_logits = logits[torch.arange(logits.size(0)), answer_positions]


    loss = torch.nn.CrossEntropyLoss()(answer_logits, answer_ids)


    if get_accuracy == True:
        top_tokens = torch.topk(logits[:, -1, :], 10, dim=-1).indices  # shape: (batch_size, top_k)

        top1_predictions = top_tokens[:, 0]
        top1_correct = (top1_predictions == answer_ids).sum().item()
        top10_correct = sum([answer_ids[j].item() in top_tokens[j].tolist() for j in range(len(answer_ids))])

        return loss.item(), top1_correct, top10_correct

    else: 

        return loss

    